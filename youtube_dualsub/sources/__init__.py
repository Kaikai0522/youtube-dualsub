"""Ways of getting audio out of YouTube.

Everything in here is quarantined behind ``pipeline.audio.get_audio`` because
this is the part of the project most likely to be broken by YouTube: as of
2026 the WEB client's player response no longer carries adaptiveFormats
playback URLs, only a SABR URL, and yt-dlp is in a running battle over it.
When that battle is lost, only this package changes.
"""
