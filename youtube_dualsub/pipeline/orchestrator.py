"""Chains the stages, checkpoints between them, and streams results out.

The pipeline is strictly serial and each stage releases its VRAM before the
next begins, which is what keeps the peak under ~9 GB on a 16 GB card even
though Demucs, Whisper and a 12B model together would want far more.

Resumption is keyed on two independent fingerprints, so the expensive half and
the cheap half of the work invalidate separately: changing the LLM or the
glossary re-runs translation only; changing the Whisper model or the vocal
isolation switch re-runs everything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import Settings
from ..export import write_ass, write_srt
from ..glossary import load_glossary
from ..llm import LlmError, OllamaClient
from ..models import Cue, Progress, Sentence, Stage, Translation, VideoContext
from ..store import Store
from . import asr as asr_stage
from . import audio as audio_stage
from . import context as context_stage
from . import postprocess, sentences as sentences_stage, shape, translate as translate_stage

log = logging.getLogger(__name__)

ProgressCb = Callable[[Progress], None]
#: ``(cues, window_start, window_end)`` — replace every cue in that time window.
CuesCb = Callable[[list[Cue], float, float], None]
CancelCb = Callable[[], bool]


class Cancelled(RuntimeError):
    """The client went away; the job pauses and keeps its checkpoints."""


class JobFailed(RuntimeError):
    pass


@dataclass(slots=True)
class JobResult:
    video_id: str
    title: str
    duration_s: float
    sentences: list[Sentence] = field(default_factory=list)
    translations: list[Translation] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    context: VideoContext = field(default_factory=VideoContext)
    used_manual_captions: bool = False
    elapsed_s: float = 0.0


class Orchestrator:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------
    def run(
        self,
        video_id: str,
        *,
        on_progress: ProgressCb | None = None,
        on_cues: CuesCb | None = None,
        should_cancel: CancelCb | None = None,
        force_retranslate: bool = False,
    ) -> JobResult:
        started = time.time()
        settings, store = self.settings, self.store
        asr_fp = settings.asr_fingerprint
        translation_fp = settings.translation_fingerprint

        def emit(stage: Stage, fraction: float | None, message: str) -> None:
            store.update_job(video_id, stage=stage, fraction=fraction, message=message)
            if on_progress:
                on_progress(Progress(stage=stage, fraction=fraction, message=message))

        def check() -> None:
            if should_cancel and should_cancel():
                raise Cancelled(video_id)

        store.ensure_job(video_id, stage=Stage.QUEUED, error=None, message="")
        result = JobResult(video_id=video_id, title=video_id, duration_s=0.0)

        try:
            sentences, result.used_manual_captions = self._get_sentences(
                video_id, asr_fp, result, emit, check
            )
            result.sentences = sentences
            if not sentences:
                raise JobFailed("No speech was recognised in this video.")

            check()
            result.context = self._get_context(video_id, asr_fp, sentences, emit, check)

            check()
            result.translations = self._translate(
                video_id, asr_fp, translation_fp, sentences, result.context,
                emit, check, on_cues, force_retranslate,
            )

            emit(Stage.SHAPE, None, "Shaping subtitles")
            result.cues = shape.build_cues(
                sentences, result.translations, settings,
                protect=protected_terms(settings, result.context),
            )
            if on_cues and result.cues:
                on_cues(result.cues, 0.0, result.cues[-1].end)

            result.elapsed_s = time.time() - started
            emit(Stage.DONE, 1.0, f"{len(result.cues)} subtitles in {result.elapsed_s:.0f}s")
            return result

        except Cancelled:
            store.update_job(video_id, stage=Stage.PAUSED, message="Paused; progress kept")
            log.info("job %s paused with checkpoints intact", video_id)
            raise
        except Exception as exc:  # noqa: BLE001
            store.update_job(video_id, stage=Stage.FAILED, message=str(exc), error=str(exc))
            log.exception("job %s failed", video_id)
            raise

    # ------------------------------------------------------------------
    def _get_sentences(
        self,
        video_id: str,
        asr_fp: str,
        result: JobResult,
        emit: Callable[[Stage, float | None, str], None],
        check: CancelCb,
    ) -> tuple[list[Sentence], bool]:
        settings, store = self.settings, self.store

        cached = store.load_sentences(video_id, asr_fp)
        job = store.get_job(video_id) or {}
        if cached:
            result.title = job.get("title") or video_id
            result.duration_s = job.get("duration_s") or 0.0
            emit(Stage.ASR, 1.0, f"Reusing {len(cached)} cached transcript lines")
            return cached, False

        emit(Stage.AUDIO, None, "Looking up the video")
        info = audio_stage.probe(video_id, settings)
        result.title, result.duration_s = info.title, info.duration_s
        store.update_job(
            video_id, title=info.title, uploader=info.uploader, duration_s=info.duration_s
        )

        fragments = None
        used_manual = False
        if settings.audio.prefer_manual_captions:
            emit(Stage.AUDIO, None, "Checking for human-uploaded captions")
            cues = audio_stage.get_manual_captions(video_id, settings)
            if cues:
                fragments = asr_stage.fragments_from_cues(cues)
                used_manual = True
                emit(Stage.ASR, 1.0, f"Using {len(cues)} human-uploaded captions")

        if fragments is None:
            check()
            asset = audio_stage.get_audio(
                video_id,
                settings,
                progress=lambda f, m: emit(Stage.AUDIO, f, m),
            )
            check()
            track = asset.path
            if settings.vocals.enabled:
                from . import vocals as vocals_stage

                emit(Stage.VOCALS, None, "Isolating vocals")
                track = vocals_stage.isolate_vocals(
                    asset.path, settings, progress=lambda f, m: emit(Stage.VOCALS, f, m)
                )

            check()
            fragments = asr_stage.transcribe(
                track,
                settings,
                duration_s=info.duration_s,
                progress=lambda f, m: emit(Stage.ASR, f, m),
            )

        emit(Stage.SENTENCES, None, "Rebuilding sentences")
        sentences = sentences_stage.build_sentences(fragments, settings)
        store.save_sentences(video_id, asr_fp, sentences)
        emit(Stage.SENTENCES, 1.0, f"{len(sentences)} sentences")
        return sentences, used_manual

    # ------------------------------------------------------------------
    def _get_context(
        self,
        video_id: str,
        asr_fp: str,
        sentences: list[Sentence],
        emit: Callable[[Stage, float | None, str], None],
        check: CancelCb,
    ) -> VideoContext:
        settings, store = self.settings, self.store
        if not settings.context.enabled:
            return VideoContext()

        context_key = f"{asr_fp}|{settings.context.model or settings.translate.model}"
        cached = store.load_context(video_id, context_key)
        if cached is not None:
            emit(Stage.CONTEXT, 1.0, f"Reusing cached context ({len(cached.terms)} terms)")
            return cached

        emit(Stage.CONTEXT, 0.0, "Reading the whole transcript")
        context = context_stage.build_context(
            sentences, settings, progress=lambda f, m: emit(Stage.CONTEXT, f, m)
        )
        store.save_context(video_id, context_key, context)
        return context

    # ------------------------------------------------------------------
    def _translate(
        self,
        video_id: str,
        asr_fp: str,
        translation_fp: str,
        sentences: list[Sentence],
        context: VideoContext,
        emit: Callable[[Stage, float | None, str], None],
        check: CancelCb,
        on_cues: CuesCb | None,
        force_retranslate: bool,
    ) -> list[Translation]:
        settings, store = self.settings, self.store

        if force_retranslate:
            store.clear_translations(video_id)
        done = {
            t.index: t
            for t in store.load_translations(video_id, asr_fp, translation_fp)
        }
        pending = [s for s in sentences if s.index not in done]

        protect = protected_terms(settings, context)
        if done and on_cues:
            # Resume: show what was already translated before doing more work.
            existing = shape.build_cues(
                sentences, list(done.values()), settings, protect=protect
            )
            if existing:
                on_cues(existing, 0.0, existing[-1].end)

        if not pending:
            emit(Stage.TRANSLATE, 1.0, f"All {len(done)} lines already translated")
            return [done[i] for i in sorted(done)]

        glossary = load_glossary(settings, context.terms)
        client = OllamaClient(
            settings.translate.model,
            num_ctx=settings.translate.num_ctx,
            temperature=settings.translate.temperature,
            timeout_s=settings.translate.request_timeout_s,
            think=settings.translate.think,
        )
        try:
            client.ensure_available()
        except LlmError as exc:
            # Degradation, not failure: English-only subtitles still beat none,
            # and Whisper's English beats YouTube's auto-captions (decision Q19).
            log.error("translation unavailable: %s", exc)
            emit(Stage.TRANSLATE, 1.0, f"No translation ({exc}); showing English only")
            return [done[i] for i in sorted(done)]

        emit(Stage.TRANSLATE, 0.0, f"Translating {len(pending)} lines")

        def on_batch(batch: list[Translation]) -> None:
            postprocess.clean_batch(batch, settings, glossary)
            store.save_translations(video_id, asr_fp, translation_fp, batch)
            for t in batch:
                done[t.index] = t
            if on_cues:
                window = shape.build_cues(
                    sentences, list(done.values()), settings, protect=protect
                )
                lo = min(s.start for s in sentences if s.index in {b.index for b in batch})
                hi = max(s.end for s in sentences if s.index in {b.index for b in batch})
                on_cues([c for c in window if lo <= c.start <= hi], lo, hi)
            check()

        translate_stage.translate_sentences(
            pending,
            context,
            settings,
            glossary=glossary.prompt_terms,
            client=client,
            on_batch=on_batch,
            progress=lambda f, m: emit(Stage.TRANSLATE, f, m),
        )
        client.unload()
        return [done[i] for i in sorted(done)]


# ----------------------------------------------------------------------


def protected_terms(settings: Settings, context: VideoContext) -> list[str]:
    """Renderings that must never be cut in half when a line is split.

    Longest first, so 終界傳送門 wins over 終界 when both would match.
    """
    glossary = load_glossary(settings, context.terms)
    return sorted({t for t in glossary.prompt_terms.values() if len(t) > 1}, key=len, reverse=True)


def export_result(result: JobResult, settings: Settings, formats: list[str]) -> list[str]:
    from ..config import EXPORT_DIR

    written: list[str] = []
    stem = _safe_stem(result.title, result.video_id)
    for fmt in formats:
        path = EXPORT_DIR / f"{stem}.{fmt}"
        if fmt == "srt":
            written.append(str(write_srt(result.cues, path, settings)))
        elif fmt == "ass":
            written.append(str(write_ass(result.cues, path, settings, title=result.title)))
        else:
            raise ValueError(f"unknown export format: {fmt}")
    return written


def _safe_stem(title: str, video_id: str) -> str:
    cleaned = "".join(c for c in title if c.isalnum() or c in " -_()[]").strip()
    return f"{cleaned[:80] or video_id}.{video_id}"
