"""Runs pipelines on worker threads and fans their output out to WebSocket clients.

The pipeline is synchronous and CPU/GPU bound, so it lives on a thread and
publishes events back into the event loop. One job per video; a second request
for a video already running just subscribes to it.

Closing the tab pauses the job (decision Q24). It does not cancel it and does
not let it run to completion in the background: checkpoints make resuming cheap,
and nobody wants their GPU quietly finishing a video they stopped watching. The
grace period exists so that an F5 or a fullscreen toggle does not count as
leaving.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .models import Cue, Progress, Stage
from .pipeline.orchestrator import Cancelled, JobResult, Orchestrator
from .store import Store

log = logging.getLogger(__name__)


def cue_json(cue: Cue) -> dict[str, Any]:
    return {
        "s": round(cue.start, 3),
        "e": round(cue.end, 3),
        "zh": cue.target,
        "en": cue.source,
        "t": cue.translated,
        **({"sp": cue.speaker} if cue.speaker else {}),
    }


@dataclass
class JobHandle:
    video_id: str
    thread: threading.Thread | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    #: Everything sent so far, replayed to any client that connects late.
    cues: list[Cue] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=lambda: {"type": "progress", "stage": "queued"})
    meta: dict[str, Any] = field(default_factory=dict)
    result: JobResult | None = None
    _reaper: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class JobManager:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self._handles: dict[str, JobHandle] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def get(self, video_id: str) -> JobHandle | None:
        return self._handles.get(video_id)

    def start(
        self, video_id: str, *, force_retranslate: bool = False
    ) -> JobHandle:
        with self._lock:
            handle = self._handles.get(video_id)
            if handle and handle.running:
                return handle
            handle = JobHandle(video_id=video_id)
            self._handles[video_id] = handle

        loop = asyncio.get_running_loop()
        handle.cancel.clear()
        handle.thread = threading.Thread(
            target=self._run,
            args=(handle, loop, force_retranslate),
            name=f"dualsub-{video_id}",
            daemon=True,
        )
        handle.thread.start()
        return handle

    def pause(self, video_id: str) -> None:
        handle = self._handles.get(video_id)
        if handle:
            handle.cancel.set()

    # ------------------------------------------------------------------
    def _run(self, handle: JobHandle, loop: asyncio.AbstractEventLoop, retranslate: bool) -> None:
        def publish(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._fanout, handle, event)

        def on_progress(progress: Progress) -> None:
            publish(
                {
                    "type": "progress",
                    "stage": progress.stage.value,
                    "fraction": progress.fraction,
                    "message": progress.message,
                }
            )

        def on_cues(cues: list[Cue], lo: float, hi: float) -> None:
            handle.cues = _splice(handle.cues, cues, lo, hi)
            publish(
                {
                    "type": "cues",
                    "window": [lo, hi],
                    "cues": [cue_json(c) for c in cues],
                }
            )

        try:
            result = Orchestrator(self.settings, self.store).run(
                handle.video_id,
                on_progress=on_progress,
                on_cues=on_cues,
                should_cancel=handle.cancel.is_set,
                force_retranslate=retranslate,
            )
            handle.result = result
            publish(
                {
                    "type": "done",
                    "title": result.title,
                    "duration_s": result.duration_s,
                    "cue_count": len(result.cues),
                    "elapsed_s": round(result.elapsed_s, 1),
                    "used_manual_captions": result.used_manual_captions,
                }
            )
        except Cancelled:
            publish({"type": "paused", "message": "Paused — progress was saved."})
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed", handle.video_id)
            publish({"type": "error", "message": str(exc)})

    def _fanout(self, handle: JobHandle, event: dict[str, Any]) -> None:
        if event["type"] in ("progress", "done", "error", "paused"):
            handle.state = event
        for queue in list(handle.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - a stalled client
                log.warning("dropping event for a slow subscriber of %s", handle.video_id)

    # ------------------------------------------------------------------
    async def subscribe(self, handle: JobHandle) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        handle.subscribers.add(queue)
        if handle._reaper is not None:
            handle._reaper.cancel()
            handle._reaper = None
        return queue

    async def unsubscribe(self, handle: JobHandle, queue: asyncio.Queue) -> None:
        handle.subscribers.discard(queue)
        if handle.subscribers or not handle.running:
            return
        handle._reaper = asyncio.create_task(self._reap(handle))

    async def _reap(self, handle: JobHandle) -> None:
        try:
            await asyncio.sleep(self.settings.server.orphan_grace_s)
        except asyncio.CancelledError:
            return
        if not handle.subscribers and handle.running:
            log.info("no clients for %s; pausing", handle.video_id)
            handle.cancel.set()


def _splice(existing: list[Cue], incoming: list[Cue], lo: float, hi: float) -> list[Cue]:
    """Replace every cue inside ``[lo, hi]`` with ``incoming``, keeping order."""
    kept = [c for c in existing if not (lo <= c.start <= hi)]
    kept.extend(incoming)
    kept.sort(key=lambda c: c.start)
    return kept


def initial_events(handle: JobHandle) -> list[dict[str, Any]]:
    """What a newly connected client needs in order to catch up."""
    events: list[dict[str, Any]] = []
    if handle.meta:
        events.append({"type": "meta", **handle.meta})
    if handle.cues:
        events.append(
            {
                "type": "cues",
                "window": [0.0, handle.cues[-1].end],
                "cues": [cue_json(c) for c in handle.cues],
                "replace_all": True,
            }
        )
    events.append(handle.state)
    return events


def stage_label(stage: str) -> str:
    return {
        Stage.QUEUED.value: "Queued",
        Stage.AUDIO.value: "Fetching audio",
        Stage.VOCALS.value: "Isolating vocals",
        Stage.ASR.value: "Transcribing",
        Stage.SENTENCES.value: "Rebuilding sentences",
        Stage.CONTEXT.value: "Reading the transcript",
        Stage.TRANSLATE.value: "Translating",
        Stage.SHAPE.value: "Shaping subtitles",
        Stage.DONE.value: "Done",
        Stage.PAUSED.value: "Paused",
        Stage.FAILED.value: "Failed",
    }.get(stage, stage)
