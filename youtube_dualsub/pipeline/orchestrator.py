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

import json
import logging
import time
from dataclasses import dataclass, field

from typing import Callable

from ..config import TERMS_VERSION, Settings
from ..export import write_ass, write_srt
from ..glossary import load_glossary
from ..llm import LlmError, OllamaClient
from ..models import Cue, Progress, Sentence, Stage, Translation, VideoContext
from ..store import Store
from . import asr as asr_stage
from . import audio as audio_stage
from . import context as context_stage
from . import diarize as diarize_stage
from . import gapfill
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
    #: The spoken language, e.g. "ja": from YouTube, the captions, or Whisper.
    language: str | None = None
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
            sentences = self._get_sentences(video_id, asr_fp, result, emit, check)
            result.sentences = sentences
            if not sentences:
                raise JobFailed("No speech was recognised in this video.")

            check()
            result.context = self._get_context(
                video_id, asr_fp, sentences, result.language, emit, check
            )

            check()
            result.translations = self._translate(
                video_id, asr_fp, translation_fp, sentences, result.context,
                result.language, emit, check, on_cues, force_retranslate,
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
    ) -> list[Sentence]:
        settings, store = self.settings, self.store

        cached = store.load_sentences(video_id, asr_fp)
        job = store.get_job(video_id) or {}
        if cached:
            result.title = job.get("title") or video_id
            result.duration_s = job.get("duration_s") or 0.0
            result.language = job.get("language")
            emit(Stage.ASR, 1.0, f"Reusing {len(cached)} cached transcript lines")
            return cached

        emit(Stage.AUDIO, None, "Looking up the video")
        info = audio_stage.probe(video_id, settings)
        result.title, result.duration_s = info.title, info.duration_s
        language = audio_stage.spoken_language(info, settings)
        store.update_job(
            video_id,
            title=info.title,
            uploader=info.uploader,
            duration_s=info.duration_s,
            description=info.description,
            tags=json.dumps(list(info.tags), ensure_ascii=False),
        )

        sentences: list[Sentence] | None = None
        if settings.audio.prefer_manual_captions:
            emit(Stage.AUDIO, None, "Checking for human-uploaded captions")
            found = audio_stage.get_manual_captions(info, settings)
            if found:
                language, cues = found
                result.used_manual_captions = True
                emit(Stage.ASR, 1.0, f"Using {len(cues)} human-uploaded captions")
                # A human already decided where each line breaks; keep it.
                sentences = sentences_stage.sentences_from_captions(
                    asr_stage.fragments_from_cues(cues), settings
                )
                if settings.diarization.enabled:
                    # Captions say what was said, not who said it; that
                    # needs the audio after all.
                    check()
                    asset = audio_stage.get_audio(
                        info, settings, progress=lambda f, m: emit(Stage.AUDIO, f, m)
                    )
                    turns = self._diarize(asset.path, emit)
                    if turns:
                        diarize_stage.label_sentences(sentences, turns)

        if sentences is None:
            check()
            asset = audio_stage.get_audio(
                info,
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
            transcription = asr_stage.transcribe(
                track,
                settings,
                language=language,
                prompt=asr_stage.build_prompt(info.title, info.description, info.tags, language),
                duration_s=info.duration_s,
                progress=lambda f, m: emit(Stage.ASR, f, m),
            )
            if settings.asr.youtube_gap_fill:
                check()
                emit(Stage.ASR, 1.0, "Filling gaps from YouTube captions")
                added = gapfill.fill_gaps(
                    transcription.fragments, audio_stage.get_auto_captions(info, settings)
                )
                if added:
                    log.info("added %d auto-caption line(s) Whisper missed", len(added))
                    if settings.asr.youtube_gap_retime:
                        check()
                        emit(Stage.ASR, 1.0, "Retiming YouTube lines")
                        heard = asr_stage.decode_windows(
                            track,
                            settings,
                            gapfill.retime_windows(added),
                            language=transcription.language or language,
                        )
                        unmoved = gapfill.retime(added, heard)
                        check()
                        forced = gapfill.retime_forced(
                            unmoved,
                            asr_stage.align_texts(
                                track,
                                settings,
                                [
                                    (lo, hi, f.text)
                                    for f, (lo, hi) in zip(
                                        unmoved, gapfill.retime_windows(unmoved)
                                    )
                                ],
                                language=transcription.language or language,
                            ),
                        )
                        log.info(
                            "retimed %d of %d auto-caption line(s), %d by forced alignment",
                            len(added) - len(unmoved) + forced,
                            len(added),
                            forced,
                        )
                    transcription.fragments = sorted(
                        transcription.fragments + added, key=lambda f: f.start
                    )
            # After Whisper, not alongside it: run concurrently, the two fought
            # over the CPU and transcription slowed by more than diarization
            # takes on its own (~5 s for a 7.5-minute clip).
            if settings.diarization.enabled:
                check()
                turns = self._diarize(track, emit)
                if turns:
                    diarize_stage.label_fragments(transcription.fragments, turns)
            language = transcription.language or language
            emit(Stage.SENTENCES, None, "Rebuilding sentences")
            sentences = sentences_stage.build_sentences(transcription.fragments, settings)

        result.language = language
        store.update_job(video_id, language=language)
        store.save_sentences(video_id, asr_fp, sentences)
        emit(Stage.SENTENCES, 1.0, f"{len(sentences)} sentences ({language or 'unknown'})")
        return sentences

    # ------------------------------------------------------------------
    def _diarize(self, track, emit) -> list:
        """Speaker turns, or none: diarization is an enhancement (Q19)."""
        if emit:
            emit(Stage.DIARIZE, None, "Telling speakers apart")
        try:
            turns = diarize_stage.diarize(
                track,
                self.settings,
                progress=(lambda f, m: emit(Stage.DIARIZE, f, m)) if emit else None,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("diarization failed, continuing without speakers: %s", exc)
            return []
        speakers = len({t.speaker for t in turns})
        log.info("diarization: %d turns, %d speakers", len(turns), speakers)
        return turns

    # ------------------------------------------------------------------
    def _get_context(
        self,
        video_id: str,
        asr_fp: str,
        sentences: list[Sentence],
        language: str | None,
        emit: Callable[[Stage, float | None, str], None],
        check: CancelCb,
    ) -> VideoContext:
        settings, store = self.settings, self.store
        if not settings.context.enabled:
            return VideoContext()

        key = context_key(settings)
        cached = store.load_context(video_id, key)
        if cached is not None:
            emit(Stage.CONTEXT, 1.0, f"Reusing cached context ({len(cached.terms)} terms)")
            return cached

        job = store.get_job(video_id) or {}
        try:
            tags = tuple(json.loads(job.get("tags") or "[]"))
        except json.JSONDecodeError:
            tags = ()
        metadata = context_stage.metadata_text(
            job.get("title") or "", job.get("uploader") or "", job.get("description") or "", tags
        )

        emit(Stage.CONTEXT, 0.0, "Reading the whole transcript")
        context = context_stage.build_context(
            sentences,
            settings,
            metadata=metadata,
            language=language,
            lookup_cache=store,
            progress=lambda f, m: emit(Stage.CONTEXT, f, m),
        )
        store.save_context(video_id, key, context)
        return context

    # ------------------------------------------------------------------
    def _translate(
        self,
        video_id: str,
        asr_fp: str,
        translation_fp: str,
        sentences: list[Sentence],
        context: VideoContext,
        language: str | None,
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
            # Degradation, not failure: source-only subtitles still beat none,
            # and Whisper's transcript beats YouTube's auto-captions (Q19).
            log.error("translation unavailable: %s", exc)
            emit(Stage.TRANSLATE, 1.0, f"No translation ({exc}); showing the original only")
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
            language=language,
            glossary=glossary.prompt_terms,
            client=client,
            on_batch=on_batch,
            progress=lambda f, m: emit(Stage.TRANSLATE, f, m),
        )
        client.unload()
        return [done[i] for i in sorted(done)]


# ----------------------------------------------------------------------


def context_key(settings: Settings) -> str:
    """What the cached summary and terminology depend on."""
    return "|".join(
        [
            settings.asr_fingerprint,
            settings.context.model or settings.translate.model,
            f"wiki{int(settings.context.wiki_lookup)}r{TERMS_VERSION}",
        ]
    )


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
