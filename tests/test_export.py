"""Bilingual subtitle file output."""

from __future__ import annotations

from youtube_dualsub.config import Settings, _merge
from youtube_dualsub.export import _srt_time, write_ass, write_srt
from youtube_dualsub.models import Cue


def cues() -> list[Cue]:
    return [
        Cue(start=1.0, end=2.5, source="we're cooked", target="我們完蛋了"),
        Cue(start=3.0, end=4.0, source="run", target=""),
    ]


def test_srt_puts_chinese_on_top_by_default(tmp_path):
    path = write_srt(cues(), tmp_path / "out.srt", Settings())
    body = path.read_text("utf-8")
    lines = body.splitlines()
    assert lines[1] == "00:00:01,000 --> 00:00:02,500"
    assert lines[2] == "我們完蛋了"
    assert lines[3] == "we're cooked"


def test_srt_can_put_english_on_top(tmp_path):
    settings = _merge(Settings(), {"style": {"zh_on_top": False}})
    body = write_srt(cues(), tmp_path / "out.srt", settings).read_text("utf-8")
    lines = body.splitlines()
    assert lines[2] == "we're cooked"
    assert lines[3] == "我們完蛋了"


def test_an_untranslated_cue_emits_a_single_line(tmp_path):
    body = write_srt(cues(), tmp_path / "out.srt", Settings()).read_text("utf-8")
    block = body.split("\n\n")[1].splitlines()
    assert block[2:] == ["run"]


def test_ass_has_a_style_and_one_dialogue_per_cue(tmp_path):
    body = write_ass(cues(), tmp_path / "out.ass", Settings(), title="Test").read_text("utf-8")
    assert "[V4+ Styles]" in body
    assert body.count("Dialogue:") == 2
    assert "我們完蛋了\\N" in body


def test_srt_timestamps():
    assert _srt_time(0) == "00:00:00,000"
    assert _srt_time(3661.5) == "01:01:01,500"
