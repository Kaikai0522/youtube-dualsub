"""Writing-system helpers: the reason Japanese is not spaced like English."""

from __future__ import annotations

from youtube_dualsub.lang import (
    ends_sentence,
    is_cjk,
    join,
    join_tokens,
    language_name,
    visible_length,
)


def test_whisper_japanese_tokens_are_joined_without_spaces():
    """The bug that motivated this module: そ こ に 人 間 いた."""
    assert join_tokens(["そ", "こ", "に", "人", "間", "いた", "い", "わ", "。"]) == "そこに人間いたいわ。"


def test_whisper_english_tokens_keep_their_own_spaces():
    assert join_tokens([" Wait", " he's", " got", " gapples."]) == "Wait he's got gapples."


def test_latin_tokens_that_lost_their_space_at_a_segment_boundary_get_one():
    assert join_tokens([" Oh", "my", " God"]) == "Oh my God"


def test_mixed_japanese_and_latin():
    assert join_tokens(["OK", "よ", "！"]) == "OKよ！"


def test_join_adds_a_space_only_between_spaced_scripts():
    assert join("Hi,", "there.") == "Hi, there."
    assert join("待って", "待って") == "待って待って"
    assert join("OK！", "何？") == "OK！何？"


def test_is_cjk_tolerates_embedded_latin():
    assert is_cjk("今日はMinecraftをやります")
    assert is_cjk("OKよ！")
    assert not is_cjk("Minecraft is fun")
    assert not is_cjk("")


def test_sentence_ends_see_through_closing_quotes():
    assert ends_sentence("「行こう。」")
    assert ends_sentence("Really?")
    assert ends_sentence("何これ？")
    assert not ends_sentence("そこに、")


def test_visible_length_counts_wide_characters_twice():
    assert visible_length("abc") == 3
    assert visible_length("日本") == 4


def test_language_names():
    assert language_name("ja") == "Japanese"
    assert language_name("en-US") == "English"
    assert language_name(None) == "the original language"
