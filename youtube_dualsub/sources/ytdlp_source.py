"""yt-dlp backed audio + manual-caption acquisition.

Two things live here:

``fetch_audio``
    Walks a chain of format specs. The first entry asks for a real audio-only
    track; the later entry accepts a muxed HLS stream and strips the audio with
    ffmpeg. HLS is what YouTube serves to TVs and iOS, so it survives the SABR
    migration that is killing the DASH audio-only URLs.

``fetch_manual_subtitles``
    Only ever returns *human-uploaded* captions. YouTube's auto-captions are
    deliberately ignored (decision Q2): they carry no punctuation and no
    casing, and feeding that to an LLM turns small ASR errors into confident
    Chinese nonsense. Better to spend 3 GPU-minutes on Whisper.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]


class AudioUnavailable(RuntimeError):
    """No format chain entry produced a usable audio file."""


class VideoTooLong(RuntimeError):
    pass


@dataclass(slots=True)
class VideoInfo:
    video_id: str
    title: str
    uploader: str
    duration_s: float
    has_manual_subs: bool
    manual_sub_langs: tuple[str, ...]


@dataclass(slots=True)
class AudioAsset:
    path: Path
    info: VideoInfo
    format_spec: str


def _ytdl():
    """Imported lazily: yt-dlp is slow to import and only the download path needs it."""
    from yt_dlp import YoutubeDL

    return YoutubeDL


def _download_error() -> type[Exception]:
    from yt_dlp.utils import DownloadError

    return DownloadError


def _url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _base_opts(cookies_from_browser: str | None) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 10,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def probe(video_id: str, *, cookies_from_browser: str | None = None) -> VideoInfo:
    opts = _base_opts(cookies_from_browser) | {"skip_download": True}
    with _ytdl()(opts) as ydl:
        raw = ydl.extract_info(_url(video_id), download=False)
    manual = raw.get("subtitles") or {}
    return VideoInfo(
        video_id=video_id,
        title=raw.get("title") or video_id,
        uploader=raw.get("uploader") or "",
        duration_s=float(raw.get("duration") or 0.0),
        has_manual_subs=bool(manual),
        manual_sub_langs=tuple(sorted(manual.keys())),
    )


def fetch_audio(
    video_id: str,
    dest_dir: Path,
    *,
    format_chain: Iterable[str],
    max_duration_s: int,
    cookies_from_browser: str | None = None,
    progress: ProgressCb | None = None,
) -> AudioAsset:
    dest_dir.mkdir(parents=True, exist_ok=True)

    info = probe(video_id, cookies_from_browser=cookies_from_browser)
    if info.duration_s > max_duration_s:
        raise VideoTooLong(
            f"{info.duration_s / 60:.0f} minutes exceeds the {max_duration_s / 60:.0f} minute limit."
        )

    cached = _find_cached(dest_dir, video_id)
    if cached is not None:
        if progress:
            progress(1.0, f"Using cached audio ({cached.name})")
        return AudioAsset(path=cached, info=info, format_spec="cache")

    def hook(d: dict) -> None:
        if not progress or d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        done = d.get("downloaded_bytes") or 0
        progress(done / total if total else None, "Downloading audio")

    failures: list[str] = []
    for spec in format_chain:
        opts = _base_opts(cookies_from_browser) | {
            "format": spec,
            "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
            "progress_hooks": [hook],
            # 'best' keeps the source codec and merely strips the video stream,
            # so the muxed-HLS fallback costs a remux rather than a re-encode.
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "best", "nopostoverwrites": False}
            ],
        }
        try:
            with _ytdl()(opts) as ydl:
                ydl.download([_url(video_id)])
        except _download_error() as exc:
            failures.append(f"{spec}: {exc}")
            log.warning("format spec %r failed: %s", spec, exc)
            continue

        produced = _find_cached(dest_dir, video_id)
        if produced is not None:
            if progress:
                progress(1.0, f"Audio ready ({produced.name})")
            return AudioAsset(path=produced, info=info, format_spec=spec)
        failures.append(f"{spec}: download reported success but produced no file")

    raise AudioUnavailable(
        "Could not obtain audio for "
        + video_id
        + ". YouTube may be forcing SABR for every client available to yt-dlp.\n"
        + "Tried:\n  "
        + "\n  ".join(failures)
        + "\nThings that sometimes help: `uv pip install -U yt-dlp`, or setting "
        "audio.cookies_from_browser = \"chrome\" in config.local.json."
    )


_AUDIO_EXTS = (".m4a", ".opus", ".webm", ".mp3", ".aac", ".ogg", ".wav", ".mp4", ".mka")


def _find_cached(dest_dir: Path, video_id: str) -> Path | None:
    for ext in _AUDIO_EXTS:
        candidate = dest_dir / f"{video_id}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


# --------------------------------------------------------------------------
# Manual captions
# --------------------------------------------------------------------------

def fetch_manual_subtitles(
    video_id: str,
    language: str,
    *,
    cookies_from_browser: str | None = None,
) -> list[tuple[float, float, str]] | None:
    """Return ``(start, end, text)`` triples from human-uploaded captions.

    Returns ``None`` when the video has no manual captions in ``language`` —
    auto-generated tracks are never considered.
    """
    opts = _base_opts(cookies_from_browser) | {"skip_download": True}
    with _ytdl()(opts) as ydl:
        raw = ydl.extract_info(_url(video_id), download=False)
        tracks = (raw.get("subtitles") or {})
        chosen = _pick_language(tracks, language)
        if chosen is None:
            return None

        entries = tracks[chosen]
        # json3 carries timing per event and needs no regex archaeology.
        by_ext = {e.get("ext"): e for e in entries}
        for ext in ("json3", "srv3", "vtt"):
            entry = by_ext.get(ext)
            if not entry or not entry.get("url"):
                continue
            try:
                payload = ydl.urlopen(entry["url"]).read().decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001 - any failure just means "no manual subs"
                log.warning("could not download %s captions: %s", ext, exc)
                continue
            cues = _parse_json3(payload) if ext in ("json3", "srv3") else _parse_vtt(payload)
            if cues:
                collapsed = _collapse_rolling(cues)
                log.info(
                    "using %d human-uploaded %s captions (%s, collapsed from %d rolling cues)",
                    len(collapsed), chosen, ext, len(cues),
                )
                return collapsed
    return None


def _collapse_rolling(
    cues: list[tuple[float, float, str]], max_gap: float = 0.4
) -> list[tuple[float, float, str]]:
    """Merge YouTube's karaoke-style repeats back into whole lines.

    YouTube serves captions as a rolling highlight: one line is re-sent every
    time the highlighted word advances, so a single spoken sentence arrives as
    five to ten cues of 0.02-0.5s each, all carrying identical text. Taken
    literally that inflates a 6-minute video from 262 lines to 1158, makes the
    subtitles strobe, and bills the translator for every duplicate.

    Only *adjacent* repeats are merged, so a line genuinely said twice with a
    pause between stays two lines.
    """
    out: list[list] = []
    for start, end, text in cues:
        if out and out[-1][2] == text and start - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end, text])
    return [(s, e, t) for s, e, t in out]


def _pick_language(tracks: dict, language: str) -> str | None:
    if not tracks:
        return None
    if language in tracks:
        return language
    prefix = language.split("-")[0]
    for key in tracks:
        if key.split("-")[0] == prefix:
            return key
    return None


def _parse_json3(payload: str) -> list[tuple[float, float, str]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    cues: list[tuple[float, float, str]] = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        dur = float(event.get("dDurationMs", 0)) / 1000.0
        cues.append((start, start + dur, _clean(text)))
    return cues


_VTT_TIME = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)


def _parse_vtt(payload: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    block: list[str] = []
    timing: tuple[float, float] | None = None

    def flush() -> None:
        if timing and block:
            text = _clean(" ".join(block))
            if text:
                cues.append((timing[0], timing[1], text))

    for line in payload.splitlines():
        m = _VTT_TIME.search(line)
        if m:
            flush()
            block = []
            timing = (_vtt_seconds(m, 0), _vtt_seconds(m, 4))
        elif not line.strip():
            flush()
            block = []
            timing = None
        elif timing is not None:
            block.append(line.strip())
    flush()
    return cues


def _vtt_seconds(m: re.Match[str], offset: int) -> float:
    hours = float(m.group(offset + 1) or 0)
    return (
        hours * 3600
        + float(m.group(offset + 2)) * 60
        + float(m.group(offset + 3))
        + float(m.group(offset + 4)) / 1000.0
    )


_TAG = re.compile(r"</?[^>]+>")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", text)).strip()
