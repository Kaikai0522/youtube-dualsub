"""Phase 5: audio captured from the browser's own MediaSource buffer.

The idea, in one paragraph: a hidden offscreen player plays the video muted at
``playbackRate = 16`` while a MAIN-world hook on
``SourceBuffer.prototype.appendBuffer`` copies every audio segment the real
YouTube player feeds to the decoder. Because the bytes are still *encoded*,
the 16x playback rate does not distort them — it only makes the player fetch
them sixteen times sooner. YouTube cannot block this without blocking its own
player, which is what makes it the durable answer to SABR.

Nothing here is wired up yet. It exists so that ``get_audio`` already has two
implementations to choose between, and switching costs one config value.
"""

from __future__ import annotations

from pathlib import Path


class MseCaptureUnavailable(RuntimeError):
    pass


def fetch_audio(video_id: str, dest_dir: Path) -> Path:  # pragma: no cover - stub
    raise MseCaptureUnavailable(
        "The MediaSource capture source is not implemented yet (Phase 5). "
        "Set audio.source = 'ytdlp' in config.local.json."
    )
