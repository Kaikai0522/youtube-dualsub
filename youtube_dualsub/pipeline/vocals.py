"""Optional Demucs vocal isolation.

The target content is game commentary: four people shouting over background
music and explosions. Whisper's word error rate and its hallucination rate
both climb sharply with music in the mix, so stripping the backing track is
plausibly the cheapest accuracy win available — but only plausibly, which is
why this is a switch to be A/B tested rather than a fixed part of the pipeline
(decision Q30).

Peak VRAM here is ~4 GB and it is fully released before ASR starts.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

from ..config import CACHE_DIR, Settings

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]


class VocalsUnavailable(RuntimeError):
    pass


def isolate_vocals(
    audio_path: Path,
    settings: Settings,
    *,
    progress: ProgressCb | None = None,
) -> Path:
    """Return a vocals-only WAV, or ``audio_path`` unchanged if isolation is off
    or unavailable. Failure here degrades rather than fails the job (Q19)."""
    cfg = settings.vocals
    if not cfg.enabled:
        return audio_path

    out_path = CACHE_DIR / f"{audio_path.stem}.vocals.wav"
    if out_path.is_file() and out_path.stat().st_size > 0:
        if progress:
            progress(1.0, "Using cached vocal track")
        return out_path

    try:
        return _separate(audio_path, out_path, settings, progress)
    except Exception as exc:  # noqa: BLE001
        if cfg.required:
            raise VocalsUnavailable(str(exc)) from exc
        log.warning("vocal isolation unavailable (%s); using the raw mix instead", exc)
        if progress:
            progress(1.0, f"Skipping vocal isolation: {exc}")
        return audio_path


def _separate(
    audio_path: Path, out_path: Path, settings: Settings, progress: ProgressCb | None
) -> Path:
    cfg = settings.vocals
    try:
        from demucs.api import Separator, save_audio
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise VocalsUnavailable(
            "demucs is not installed. Run: uv sync --extra vocals"
        ) from exc

    if progress:
        progress(None, f"Loading Demucs ({cfg.model})")

    separator = Separator(
        model=cfg.model,
        device=cfg.device,
        segment=cfg.segment_s,
        progress=False,
    )
    try:
        if progress:
            progress(None, "Separating vocals from background")
        _origin, stems = separator.separate_audio_file(str(audio_path))
        if "vocals" not in stems:
            raise VocalsUnavailable(f"model {cfg.model} produced no 'vocals' stem")
        save_audio(stems["vocals"], str(out_path), samplerate=separator.samplerate)
    finally:
        del separator
        _release_cuda()

    if progress:
        progress(1.0, "Vocals isolated")
    return out_path


def _release_cuda() -> None:
    """Hand the ~4 GB back before the next stage loads its model."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch is optional
        pass
