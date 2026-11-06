"""The scoring rules of the ASR harness: a wrong metric would bless a regression."""

from __future__ import annotations

import pytest

from youtube_dualsub.asr_eval import (
    clean,
    error_rate,
    found_lines,
    letters,
    parse_set,
    score,
    start_offsets,
    words,
)


class TestNormalising:
    def test_notes_speakers_and_invisible_marks_are_not_speech(self):
        assert letters("​ ぼたん：えー（笑）[拍手] 本当に？ ​") == "えー本当に"

    def test_width_and_case_are_folded(self):
        assert letters("ＡＢＣ　ｱｲｳ") == "abcアイウ"

    def test_a_time_is_not_taken_for_a_speaker(self):
        assert letters("10:30 start") == "1030start"

    def test_english_words_keep_apostrophes(self):
        assert words("Phil: I can’t — DON'T scream!") == ["i", "can't", "don't", "scream"]

    def test_a_drawn_out_english_word_is_the_word(self):
        assert words("Nooo, howww?! all good") == ["no", "how", "all", "good"]

    def test_clean_lowercases(self):
        assert clean("WORK TO DO") == "work to do"


class TestErrorRate:
    def test_characters(self):
        assert error_rate("本当にやばい", "本当やばい") == (1, 6)

    def test_words(self):
        assert error_rate(["a", "b", "c"], ["a", "x", "c", "d"]) == (2, 3)


class TestFoundLines:
    def test_a_line_said_nearby_is_found(self):
        ref = [(10.0, 12.0, "いや、マジでね")]
        hyp = [(11.0, 14.0, "いやマジでねほんとに")]
        assert found_lines(ref, hyp) == (1, 1)

    def test_a_line_said_far_away_is_not(self):
        ref = [(10.0, 12.0, "いや、マジでね")]
        hyp = [(40.0, 42.0, "いやマジでね")]
        assert found_lines(ref, hyp) == (0, 1)

    def test_an_invented_line_lowers_precision(self):
        ref = [(10.0, 12.0, "おはよう")]
        hyp = [(10.0, 12.0, "おはよう"), (11.0, 12.0, "ごちそうさまでした")]
        assert found_lines(hyp, ref) == (1, 2)

    def test_one_letter_lines_are_not_counted(self):
        assert found_lines([(1.0, 2.0, "え")], []) == (0, 0)


class TestStartOffsets:
    def test_the_nearest_matching_start_is_used(self):
        """いや recurs: the line two seconds away is not the one that was said."""
        ref = [(20.0, 22.0, "いやいや待って")]
        hyp = [(16.5, 17.0, "いやいや待ってって"), (20.4, 22.0, "いやいや待って")]
        assert start_offsets(ref, hyp) == [pytest.approx(0.4)]

    def test_a_line_opened_mid_sentence_has_no_timing(self):
        ref = [(20.0, 22.0, "待ってよ")]
        hyp = [(19.0, 22.0, "いや待ってよ")]
        assert start_offsets(ref, hyp) == []


def test_score_uses_words_for_english_and_letters_for_japanese():
    ref = [(0.0, 2.0, "That was clean.")]
    assert score(ref, [(0.0, 2.0, "that was clean")], "en").edits == 0
    assert score([(0.0, 2.0, "本当に")], [(0.0, 2.0, "本当")], "ja").edits == 1


def test_overrides_from_the_command_line():
    assert parse_set(["asr.beam_size=1", "asr.language=ja", "asr.batched=false"]) == {
        "asr": {"beam_size": 1, "language": "ja", "batched": False}
    }
