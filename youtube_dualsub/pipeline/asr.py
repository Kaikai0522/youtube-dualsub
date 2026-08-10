"""Speech recognition with faster-whisper.

Three settings here are load-bearing for noisy multi-speaker content and should
not be "tidied up":

``condition_on_previous_text=False``
    With it on, Whisper feeds its own last output back in as a prompt. One
    hallucination during a music sting then seeds the next window, and the
    damage cascades for minutes. Off costs a little coherence and buys back
    the whole video.

``vad_filter=True``
    Not optional. Whisper invents speech in silence and under music — the
    classic "Thanks for watching!" over an outro. The VAD never shows it those
    regions in the first place.

``word_timestamps=True``
    Sentence rebuilding needs pause lengths, and cue shaping needs to split a
    long sentence at a real word boundary.
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Callable

from .._cuda import ensure_cuda_dlls
from ..config import Settings
from ..models import Fragment, Word

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

#: Phrases Whisper emits when it has nothing to transcribe. Only ever dropped
#: when the model itself also reports a high no-speech probability, so real
#: occurrences in the dialogue survive.
_FILLER_HALLUCINATIONS = {
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "you",
    "bye",
    "♪",
}


class AsrFailed(RuntimeError):
    pass


def transcribe(
    audio_path: Path,
    settings: Settings,
    *,
    duration_s: float | None = None,
    progress: ProgressCb | None = None,
) -> list[Fragment]:
    ensure_cuda_dlls()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise AsrFailed("faster-whisper is not installed. Run: uv sync") from exc

    cfg = settings.asr
    if progress:
        progress(None, f"Loading Whisper {cfg.model}")

    try:
        model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except Exception as exc:  # noqa: BLE001
        raise AsrFailed(_explain_load_failure(exc)) from exc

    fragments: list[Fragment] = []
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=cfg.language,
            beam_size=cfg.beam_size,
            word_timestamps=cfg.word_timestamps,
            condition_on_previous_text=cfg.condition_on_previous_text,
            temperature=list(cfg.temperature),
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            no_speech_threshold=cfg.no_speech_threshold,
            vad_filter=cfg.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": cfg.vad_min_silence_ms,
                "speech_pad_ms": cfg.vad_speech_pad_ms,
            },
        )
        total = duration_s or getattr(info, "duration", None) or 0.0
        if progress:
            detected = getattr(info, "language", cfg.language)
            progress(0.0, f"Transcribing ({detected})")

        recent: list[str] = []
        dropped = 0
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            if _is_hallucination(text, recent, getattr(seg, "no_speech_prob", 0.0)):
                dropped += 1
                continue
            recent.append(text.lower())
            del recent[:-3]

            fragments.append(
                Fragment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    words=[
                        Word(
                            start=float(w.start),
                            end=float(w.end),
                            text=w.word,
                            probability=float(getattr(w, "probability", 1.0)),
                        )
                        for w in (seg.words or [])
                        if w.start is not None and w.end is not None
                    ],
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                )
            )
            if progress and total:
                progress(min(1.0, float(seg.end) / total), "Transcribing")

        if dropped:
            log.info("dropped %d hallucinated segment(s)", dropped)
    finally:
        del model
        _release_cuda()

    if not fragments:
        raise AsrFailed(
            "Whisper produced no speech at all. The audio file may be silent or corrupt."
        )

    if progress:
        progress(1.0, f"Transcribed {len(fragments)} fragments")
    return fragments


def fragments_from_cues(cues: list[tuple[float, float, str]]) -> list[Fragment]:
    """Adapt human-uploaded captions into the shape the rest of the pipeline expects."""
    return [
        Fragment(start=start, end=end, text=text)
        for start, end, text in cues
        if text.strip()
    ]


# --------------------------------------------------------------------------

_REPEAT = re.compile(r"\b(\w[\w']*)(?:\W+\1\b){3,}", re.IGNORECASE)


def _is_hallucination(text: str, recent: list[str], no_speech_prob: float) -> bool:
    normalized = text.strip().lower()
    if normalized in _FILLER_HALLUCINATIONS and no_speech_prob > 0.5:
        return True
    # The same short line three times running is the model looping, not four
    # people saying "go go go" — that has different timings and longer text.
    if len(normalized) < 60 and recent.count(normalized) >= 2:
        return True
    if _REPEAT.search(text):
        return True
    return False


def _explain_load_failure(exc: Exception) -> str:
    message = str(exc)
    if "cudnn" in message.lower() or "cublas" in message.lower():
        return (
            f"{message}\n\n"
            "This is the cuDNN/cuBLAS DLL problem. Run `.\\setup.ps1` again, or:\n"
            "  uv pip install 'nvidia-cudnn-cu12>=9.1,<10' 'nvidia-cublas-cu12>=12.4'"
        )
    return message


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
