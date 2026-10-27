"""The parts of the ASR stage that do not need a GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from youtube_dualsub.models import Fragment, Word
from youtube_dualsub.pipeline.asr import (
    _is_hallucination,
    _looks_degenerate,
    build_prompt,
    snap_to_speech,
)


def test_the_prompt_carries_the_names_bracketed_in_the_title():
    prompt = build_prompt("【Vspo精華】多人連動【橘ひなの】【猫汰つな】【hendy】", "", (), "ja")
    for name in ("橘ひなの", "猫汰つな", "hendy"):
        assert name in prompt
    assert "。" in prompt, "the style sentence is what switches Japanese punctuation on"


def test_clip_channel_labels_are_not_names():
    """A long prompt of 精華/翻譯/中文 hashtags sent the decoder into loops."""
    prompt = build_prompt("【Vspo精華】【切り抜き】【Vtuber中文】【橘ひなの】", "#翻譯 #烤肉", (), "ja")
    for label in ("Vspo精華", "切り抜き", "Vtuber中文", "翻譯", "烤肉"):
        assert label not in prompt


def test_the_prompt_stays_short():
    title = "".join(f"【name{i}】" for i in range(20))
    tags = tuple(f"tag{i}" for i in range(200))
    prompt = build_prompt(title, "#a " * 100, tags, "en")
    assert prompt.count("name") == 6
    assert "tag" not in prompt


def frag(text, start=10.0, nsp=0.1, logprob=-0.2):
    return Fragment(start=start, end=start + 1, text=text, no_speech_prob=nsp, avg_logprob=logprob)


class TestHallucinations:
    def test_the_japanese_outro_hallucination_is_dropped_mid_video(self):
        assert _is_hallucination(frag("ご視聴ありがとうございました。", start=120), [], "ja", 196)

    def test_a_real_sign_off_at_the_very_end_survives(self):
        assert not _is_hallucination(frag("ご視聴ありがとうございました。", start=180), [], "ja", 196)

    def test_subtitle_credits_are_dropped(self):
        text = "この動画の字幕は、ご覧の方によって作成されました。"
        assert _is_hallucination(frag(text), [], "ja", 196)

    def test_a_runaway_loop_is_dropped_but_real_repetition_is_not(self):
        assert _is_hallucination(frag("もう1個探せよ" * 12), [], "ja", 600)
        assert not _is_hallucination(frag("待って待って待って待って"), [], "ja", 600)

    def test_stray_low_confidence_latin_in_a_japanese_video(self):
        assert _is_hallucination(frag("insurance calcula", logprob=-0.9), [], "ja", 600)
        assert not _is_hallucination(frag("OK！OK！", logprob=-0.9), [], "ja", 600)
        assert not _is_hallucination(frag("insurance calcula", logprob=-0.9), [], "en", 600)


def test_a_looping_batched_chunk_is_sent_back_for_a_careful_decode():
    looped = SimpleNamespace(text="もう1個探せよ" * 8)
    fine = SimpleNamespace(text="ツナ見て、ほら、金の玉あったよ。")
    assert _looks_degenerate(looped, 2.4)
    assert not _looks_degenerate(fine, 2.4)


class TestRepeatedLines:
    def test_several_people_saying_the_same_short_thing_is_kept(self):
        """Two speakers' "え？" three times over is a conversation, not a loop."""
        recent = ["え?", "え?"]
        assert not _is_hallucination(frag("え?", start=111.0), recent, "ja", 165)

    def test_a_longer_line_repeated_three_times_running_is_a_loop(self):
        line = "これ、あのー、人員募集なんですか?"
        assert _is_hallucination(frag(line, start=300.0), [line.lower()] * 2, "ja", 452)

    def test_text_with_no_time_to_be_spoken_is_dropped(self):
        zero = Fragment(start=319.71, end=319.71, text="感染部なんですか?")
        assert _is_hallucination(zero, [], "ja", 452)
        assert not _is_hallucination(Fragment(start=10.0, end=10.0, text="え?"), [], "ja", 452)


def word(start, end, text):
    return Word(start=start, end=end, text=text)


class TestSnapToSpeech:
    def test_a_word_that_swallowed_the_silence_before_it_is_clamped(self):
        fragment = Fragment(79.7, 87.6, "", [word(79.7, 84.9, " So"), word(84.9, 85.3, " then")])
        [out] = snap_to_speech([fragment], [(76.0, 79.3), (84.5, 88.0)])
        assert out.words[0].start == pytest.approx(84.5)
        assert out.start == pytest.approx(84.5)

    def test_a_word_outside_all_speech_keeps_its_timing(self):
        fragment = Fragment(10.0, 10.5, "", [word(10.0, 10.5, " hm")])
        [out] = snap_to_speech([fragment], [(0.0, 5.0), (20.0, 25.0)])
        assert (out.words[0].start, out.words[0].end) == (10.0, 10.5)

    def test_a_kana_left_behind_in_the_previous_speakers_audio_rejoins_its_word(self):
        """シ at 79.72 s, ャル、スマブラ… from 84.44 s: one word, one speaker."""
        words_ = [word(79.72, 80.08, "シ"), word(84.44, 84.50, "ャ"), word(84.50, 84.52, "ル、")]
        fragment = Fragment(79.72, 84.52, "シャル、", words_)
        [out] = snap_to_speech([fragment], [(76.0, 80.2), (84.3, 88.0)])
        assert out.start == pytest.approx(84.34)
        assert out.words[0].end == pytest.approx(84.44)

    def test_a_kana_left_behind_before_a_number_rejoins_its_line(self):
        """v-V9NpCvPDk 3:49: で at 228.13 s, 1ラウンド… from 235.04 s; spoken at 3:56."""
        words_ = [word(228.13, 228.53, "で"), word(235.04, 235.44, "1"), word(235.44, 235.62, "ラ")]
        fragment = Fragment(228.13, 235.62, "で1ラ", words_)
        [out] = snap_to_speech([fragment], [(227.96, 228.23), (235.29, 239.27)])
        assert out.start == pytest.approx(235.19)

    def test_a_token_stretched_over_the_previous_voice_is_cut_back(self):
        """v-V9NpCvPDk 9:58: なん 597.95 s, で 598.53-600.41 s, すぐ…; spoken at 10:00."""
        words_ = [word(597.95, 598.45, "なん"), word(598.53, 600.41, "で"), word(600.41, 600.49, "す")]
        fragment = Fragment(597.95, 600.49, "なんです", words_)
        [out] = snap_to_speech([fragment], [(596.0, 598.5), (598.78, 604.1)])
        assert out.start == pytest.approx(600.21)

    def test_a_held_vowel_keeps_its_length(self):
        words_ = [word(10.0, 11.2, "えー"), word(11.2, 11.4, "と")]
        fragment = Fragment(10.0, 11.4, "えーと", words_)
        [out] = snap_to_speech([fragment], [(9.8, 12.0)])
        assert out.start == pytest.approx(10.0)

    def test_a_pause_after_punctuation_is_left_alone(self):
        words_ = [word(97.31, 97.63, "で、"), word(101.78, 101.86, "ス")]
        fragment = Fragment(97.31, 101.86, "で、ス", words_)
        [out] = snap_to_speech([fragment], [(97.0, 98.0), (101.5, 103.0)])
        assert out.start == pytest.approx(97.31)



def test_decode_windows_groups_never_add_up_to_more_than_the_audio():
    """faster-whisper asserts total clip length <= audio length (d1MzjzSTwvc)."""
    from youtube_dualsub.pipeline.asr import _within

    regions = [(k * 5.0, k * 5.0 + 17.0) for k in range(10)]
    groups = _within(regions, 60.0)

    assert sorted(k for g in groups for k in g) == list(range(10))
    assert all(sum(regions[k][1] - regions[k][0] for k in g) <= 60.0 for g in groups)
