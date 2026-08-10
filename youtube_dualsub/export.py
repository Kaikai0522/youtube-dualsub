"""Write cues out as bilingual subtitle files.

Exports exist mostly so the translation can be *inspected*: a .srt opened in a
text editor is the fastest way to check whether the glossary held, whether the
model slipped into Simplified, and whether it invented anything. That it also
lets the subtitles be used in mpv, Plex or on a phone is a bonus.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .models import Cue

_ASS_HEADER = """[Script Info]
Title: {title}
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dual,Microsoft JhengHei,{fs_primary},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,46,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_srt(cues: list[Cue], path: Path, settings: Settings) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i, cue in enumerate(cues, start=1):
        lines = _ordered_lines(cue, settings)
        if not lines:
            continue
        blocks.append(
            f"{i}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n" + "\n".join(lines) + "\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def write_ass(cues: list[Cue], path: Path, settings: Settings, *, title: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    style = settings.style
    # The overlay sizes are CSS pixels against a ~720p player; ASS is authored
    # against PlayResY 1080, so scale to keep the two renderings comparable.
    scale = 1.8
    primary = round(style.font_size_zh * scale) if style.zh_on_top else round(
        style.font_size_en * scale
    )
    secondary = round(style.font_size_en * scale) if style.zh_on_top else round(
        style.font_size_zh * scale
    )

    out = [_ASS_HEADER.format(title=title or path.stem, fs_primary=primary)]
    for cue in cues:
        lines = _ordered_lines(cue, settings)
        if not lines:
            continue
        if len(lines) == 2:
            text = f"{_ass_escape(lines[0])}\\N{{\\fs{secondary}}}{_ass_escape(lines[1])}"
        else:
            text = _ass_escape(lines[0])
        out.append(
            f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Dual,,0,0,0,,{text}\n"
        )
    path.write_text("".join(out), encoding="utf-8")
    return path


# --------------------------------------------------------------------------


def _ordered_lines(cue: Cue, settings: Settings) -> list[str]:
    prefix = ""
    if settings.style.show_speaker_prefix and cue.speaker:
        prefix = f"{cue.speaker}: "

    zh = cue.target.strip()
    en = cue.source.strip()
    if not zh and not en:
        return []
    if not zh:
        return [prefix + en]
    if not en:
        return [prefix + zh]
    return [prefix + zh, en] if settings.style.zh_on_top else [prefix + en, zh]


def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
