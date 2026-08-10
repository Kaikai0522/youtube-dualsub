"""The only place in the project that knows where audio comes from.

Every other stage receives a `Path` and asks no questions. That is deliberate:
YouTube's SABR migration is actively breaking yt-dlp, and when the browser-side
MediaSource capture replaces it (Phase 5) the change is confined to this file
and ``sources/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ..config import CACHE_DIR, Settings
from ..sources import ytdlp_source
from ..sources.ytdlp_source import AudioAsset, AudioUnavailable, VideoInfo, VideoTooLong

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

__all__ = [
    "AudioAsset",
    "AudioUnavailable",
    "VideoInfo",
    "VideoTooLong",
    "get_audio",
    "get_manual_captions",
    "probe",
]


def probe(video_id: str, settings: Settings) -> VideoInfo:
    return ytdlp_source.probe(
        video_id, cookies_from_browser=settings.audio.cookies_from_browser
    )


def get_audio(
    video_id: str,
    settings: Settings,
    *,
    progress: ProgressCb | None = None,
) -> AudioAsset:
    if settings.audio.source == "mse":
        from ..sources import mse_source

        path = mse_source.fetch_audio(video_id, CACHE_DIR)
        return AudioAsset(path=path, info=probe(video_id, settings), format_spec="mse")

    return ytdlp_source.fetch_audio(
        video_id,
        CACHE_DIR,
        format_chain=settings.audio.format_chain,
        max_duration_s=settings.audio.max_duration_s,
        cookies_from_browser=settings.audio.cookies_from_browser,
        progress=progress,
    )


def get_manual_captions(
    video_id: str, settings: Settings
) -> list[tuple[float, float, str]] | None:
    """Human-uploaded captions only, or ``None``.

    When this returns something, the whole audio + Demucs + Whisper chain is
    skipped: a human transcript beats anything we can produce locally, is
    already time-aligned, and costs zero GPU seconds.
    """
    language = settings.asr.language or "en"
    try:
        return ytdlp_source.fetch_manual_subtitles(
            video_id,
            language,
            cookies_from_browser=settings.audio.cookies_from_browser,
        )
    except Exception as exc:  # noqa: BLE001 - never let a caption probe kill the job
        log.warning("manual caption lookup failed, falling back to ASR: %s", exc)
        return None
