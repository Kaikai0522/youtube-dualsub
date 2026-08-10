"""Cue shaping rules — the part of the pipeline with no GPU and no excuses."""

from __future__ import annotations

import pytest

from youtube_dualsub.config import Settings
from youtube_dualsub.models import Sentence, Translation
from youtube_dualsub.pipeline import shape


def settings(**shape_overrides) -> Settings:
    from youtube_dualsub.config import _merge

    return _merge(Settings(), {"shape": shape_overrides}) if shape_overrides else Settings()


def sentence(index: int, start: float, end: float, text: str = "hello there") -> Sentence:
    return Sentence(index=index, start=start, end=end, text=text)


def translation(index: int, text: str = "你好") -> Translation:
    return Translation(index=index, text=text)


class TestClamping:
    def test_a_cue_never_outlives_the_next_ones_start(self):
        """The rule the whole design hangs on: subtitles track the speaker.

        Two sentences 0.4s apart. The 1.0s readability floor would push the
        first cue to 1.0s and cover the second speaker; it must not.
        """
        sentences = [sentence(0, 0.0, 0.3), sentence(1, 0.4, 3.0)]
        translations = [translation(0), translation(1)]

        cues = shape.build_cues(sentences, translations, settings(merge_below_s=0.0))

        assert len(cues) == 2
        assert cues[0].end <= cues[1].start
        assert cues[0].end == pytest.approx(0.4, abs=1e-6)

    def test_min_duration_applies_when_there_is_room(self):
        sentences = [sentence(0, 0.0, 0.3), sentence(1, 10.0, 12.0)]
        cues = shape.build_cues(
            sentences, [translation(0), translation(1)], settings(merge_below_s=0.0)
        )
        assert cues[0].duration == pytest.approx(1.0, abs=1e-6)

    def test_nothing_loiters_past_the_maximum(self):
        sentences = [sentence(0, 0.0, 30.0)]
        cues = shape.build_cues(sentences, [translation(0)], settings())
        assert cues[0].duration <= Settings().shape.max_duration_s + 1e-6

    def test_overlapping_asr_timings_are_pulled_apart(self):
        sentences = [sentence(0, 0.0, 5.0), sentence(1, 3.0, 6.0)]
        cues = shape.build_cues(
            sentences, [translation(0), translation(1)], settings(merge_below_s=0.0)
        )
        assert cues[0].end <= cues[1].start
        assert all(c.duration > 0 for c in cues)


class TestMerging:
    def test_flicker_is_fixed_by_merging_not_by_holding(self):
        """Four 0.3s fragments become one readable cue, still ending on time."""
        sentences = [
            sentence(0, 0.0, 0.3, "go"),
            sentence(1, 0.3, 0.6, "go"),
            sentence(2, 0.6, 0.9, "go"),
            sentence(3, 5.0, 7.0, "he's behind you"),
        ]
        translations = [translation(i, t) for i, t in enumerate(["衝", "衝", "衝", "他在你後面"])]

        cues = shape.build_cues(sentences, translations, settings())

        assert len(cues) == 2
        assert cues[0].target == "衝衝衝"
        assert cues[0].source == "go go go"
        assert cues[0].end <= cues[1].start

    def test_a_trailing_stub_folds_backwards(self):
        sentences = [sentence(0, 0.0, 4.0, "that was close"), sentence(1, 4.0, 4.2, "yeah")]
        cues = shape.build_cues(
            sentences, [translation(0, "好險"), translation(1, "對啊")], settings()
        )
        assert len(cues) == 1
        assert cues[0].target == "好險對啊"


class TestSplitting:
    def test_a_long_line_becomes_several_cues_sharing_the_time(self):
        long_zh = "這是一段非常長的中文字幕，" * 4
        cues = shape.build_cues(
            [sentence(0, 0.0, 20.0, "a very long english line " * 4)],
            [translation(0, long_zh)],
            settings(),
        )
        assert len(cues) > 1
        assert all(len(c.target) <= Settings().shape.max_chars_zh + 2 for c in cues)
        assert all(c.source for c in cues), "the English must be split alongside the Chinese"
        assert cues[0].start == pytest.approx(0.0)
        for earlier, later in zip(cues, cues[1:]):
            assert earlier.end <= later.start + 1e-6

    def test_a_slight_overflow_splits_evenly_instead_of_orphaning_a_tail(self):
        """23 characters under a 20-character limit must not become 20 + 3.

        Greedy filling left three characters with their own half-second cue.
        """
        zh = "速通者通常會得到一個關於最近興趣點方向的提示。"
        assert len(zh) == 23
        cues = shape.build_cues(
            [sentence(0, 0.0, 4.0, "a b c d e f g h i j k l m n o p")],
            [translation(0, zh)],
            settings(),
        )
        assert len(cues) == 2
        shorter, longer = sorted(len(c.target) for c in cues)
        assert shorter >= 8, f"orphaned tail: {[c.target for c in cues]}"
        assert longer - shorter <= 3, "the two halves should be close in length"
        assert all(c.source.strip() for c in cues), "the English must be split too"

    def test_every_piece_still_respects_the_ceiling(self):
        cues = shape.build_cues(
            [sentence(0, 0.0, 30.0, "x " * 60)],
            [translation(0, "字" * 97)],
            settings(),
        )
        assert all(len(c.target) <= Settings().shape.max_chars_zh for c in cues)
        assert sum(len(c.target) for c in cues) == 97, "no characters may be lost"

class TestProtectedTerms:
    """Chinese has no spaces, so an index-based split will sever a word unless
    it is told which character runs belong together."""

    def test_a_glossary_term_is_never_cut_in_half(self):
        zh = "這樣如果探查者因為特定原因選了這個種子碼，跑者實際上會實現那個原因。"
        cues = shape.build_cues(
            [sentence(0, 0.0, 6.0, "x " * 30)],
            [translation(0, zh)],
            settings(),
            protect=["種子碼", "探查者"],
        )
        joined = [c.target for c in cues]
        assert len(cues) > 1
        for term in ("種子碼", "探查者"):
            assert any(term in part for part in joined), f"{term} was split across {joined}"

    def test_latin_words_survive_intact_without_being_listed(self):
        cues = shape.build_cues(
            [sentence(0, 0.0, 6.0, "x " * 20)],
            [translation(0, "這個伺服器執行的是Minecraft最新版本而且非常穩定喔")],
            settings(),
        )
        assert any("Minecraft" in c.target for c in cues), [c.target for c in cues]

    def test_numbers_are_not_severed(self):
        cues = shape.build_cues(
            [sentence(0, 0.0, 6.0, "x " * 20)],
            [translation(0, "他們總共花了1234567秒才終於抵達那個地方真是誇張")],
            settings(),
        )
        assert any("1234567" in c.target for c in cues), [c.target for c in cues]

    def test_a_word_severed_upstream_is_repaired(self):
        """Both halves fit the limit, so splitting never runs — the join must.

        The translator ends one line with "Minecraft M" and starts the next with
        "anhunt"; nothing downstream would otherwise notice.
        """
        cues = shape.build_cues(
            [sentence(0, 0.0, 2.0, "Thanks everyone."), sentence(1, 2.0, 4.0, "How it is made.")],
            [translation(0, "謝謝大家。Minecraft M"), translation(1, "anhunt影片是如何製作的？")],
            settings(),
        )
        assert not any(
            c.target.endswith("M") for c in cues
        ), [c.target for c in cues]
        assert any("Manhunt" in c.target for c in cues), [c.target for c in cues]

    def test_a_repair_does_not_glue_across_a_silence(self):
        cues = shape.build_cues(
            [sentence(0, 0.0, 2.0, "a"), sentence(1, 40.0, 42.0, "b")],
            [translation(0, "測試M"), translation(1, "anhunt")],
            settings(),
        )
        assert len(cues) == 2

    def test_a_term_longer_than_the_line_budget_still_terminates(self):
        """Something has to give, but it must not loop forever."""
        cues = shape.build_cues(
            [sentence(0, 0.0, 6.0, "x")],
            [translation(0, "字" * 60)],
            settings(),
            protect=["字" * 60],
        )
        assert cues and sum(len(c.target) for c in cues) == 60


class TestSplittingPunctuation:
    def test_split_prefers_punctuation(self):
        cues = shape.build_cues(
            [sentence(0, 0.0, 10.0, "one two three four five six seven eight")],
            [translation(0, "第一句話已經講完了而且講得很長，第二句話從這裡開始講起也很長")],
            settings(),
        )
        assert len(cues) == 2
        assert cues[0].target == "第一句話已經講完了而且講得很長，"

    def test_a_stub_does_not_merge_across_a_silence(self):
        """Merging is for flicker, not for gluing together separate utterances."""
        cues = shape.build_cues(
            [sentence(0, 0.0, 0.4, "huh"), sentence(1, 6.0, 8.0, "he found the portal")],
            [translation(0, "蛤"), translation(1, "他找到傳送門了")],
            settings(),
        )
        assert len(cues) == 2
        assert cues[0].target == "蛤"


class TestUntranslated:
    def test_missing_translation_still_yields_the_english(self):
        """Decision Q19: English beats nothing, and beats YouTube's auto-captions."""
        cues = shape.build_cues([sentence(0, 0.0, 2.0, "we're cooked")], [], settings())
        assert len(cues) == 1
        assert cues[0].source == "we're cooked"
        assert cues[0].target == ""
        assert cues[0].translated is False
