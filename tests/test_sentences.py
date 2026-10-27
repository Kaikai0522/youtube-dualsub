"""Rebuilding whole sentences out of Whisper's fragments."""

from __future__ import annotations

import pytest

from youtube_dualsub.config import Settings
from youtube_dualsub.models import Fragment, Word
from youtube_dualsub.pipeline.sentences import build_sentences, sentences_from_captions


def words(spec: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start=s, end=e, text=t) for s, e, t in spec]


def test_fragments_are_glued_back_into_one_thought():
    """The motivating case: three fragments, one sentence, one translation unit."""
    fragments = [
        Fragment(0.0, 2.1, " Wait he's got", words(
            [(0.0, 0.4, " Wait"), (0.5, 0.9, " he's"), (1.0, 2.1, " got")])),
        Fragment(2.1, 4.4, " gapples. We're", words(
            [(2.1, 3.0, " gapples."), (3.2, 4.4, " We're")])),
        Fragment(4.4, 5.0, " cooked.", words([(4.4, 5.0, " cooked.")])),
    ]

    sentences = build_sentences(fragments, Settings())

    assert [s.text for s in sentences] == ["Wait he's got gapples.", "We're cooked."]
    assert sentences[0].start == pytest.approx(0.0)
    assert sentences[0].end == pytest.approx(3.0)
    assert [s.index for s in sentences] == [0, 1]


def test_a_long_pause_ends_a_sentence_without_punctuation():
    fragments = [
        Fragment(0.0, 4.0, " no way", words([(0.0, 0.5, " no"), (0.6, 1.0, " way")])),
        Fragment(4.0, 5.0, " run", words([(4.0, 5.0, " run")])),
    ]
    sentences = build_sentences(fragments, Settings())
    assert len(sentences) == 2


def test_a_runaway_monologue_is_cut_at_the_ceiling():
    spec = [(i * 0.5, i * 0.5 + 0.4, f" word{i}") for i in range(60)]
    fragments = [Fragment(0.0, 30.0, " ".join(w[2] for w in spec), words(spec))]

    sentences = build_sentences(fragments, Settings())

    assert len(sentences) > 1
    assert all(s.duration <= Settings().sentences.max_sentence_s + 1.0 for s in sentences)


def test_captions_without_word_timestamps_still_work():
    """Human-uploaded captions arrive as coarse blocks and must flow through."""
    fragments = [
        Fragment(0.0, 2.0, "Welcome back everyone."),
        Fragment(2.0, 4.0, "Today we're doing a manhunt."),
    ]
    sentences = build_sentences(fragments, Settings())
    assert [s.text for s in sentences] == [
        "Welcome back everyone.",
        "Today we're doing a manhunt.",
    ]


def test_indices_are_contiguous_after_stubs_are_absorbed():
    fragments = [
        Fragment(0.0, 1.0, " Hi.", words([(0.0, 1.0, " Hi.")])),
        Fragment(1.0, 1.1, " a", words([(1.0, 1.1, " a")])),
        Fragment(2.0, 3.0, " Bye.", words([(2.0, 3.0, " Bye.")])),
    ]
    sentences = build_sentences(fragments, Settings())
    assert [s.index for s in sentences] == list(range(len(sentences)))


# --------------------------------------------------------------------------
# Scored segmentation: cases taken from real transcripts
# --------------------------------------------------------------------------


def frag(spec: list[tuple[float, float, str]]) -> Fragment:
    return Fragment(spec[0][0], spec[-1][1], "".join(t for _, _, t in spec), words(spec))


def test_a_tail_after_a_question_mark_joins_the_sentence_it_starts():
    """Whisper put the pause after "Oh", not after "Dream?". Grammar wins."""
    fragments = [
        frag([(158.7, 159.0, " You"), (159.0, 159.3, " know"), (159.3, 160.3, " Dream?"),
              (160.5, 160.9, " Oh")]),
        frag([(162.5, 162.8, " my"), (162.8, 163.1, " God,"), (163.1, 163.6, " I'm"),
              (163.6, 164.0, " catching"), (164.0, 164.4, " you.")]),
    ]

    texts = [s.text for s in build_sentences(fragments, Settings())]

    assert texts == ["You know Dream?", "Oh my God, I'm catching you."]


def test_a_lone_word_before_a_lowercase_continuation_is_not_a_sentence():
    fragments = [
        frag([(157.5, 157.8, " You")]),
        frag([(158.7, 159.0, " know"), (159.0, 159.4, " that"), (159.4, 160.0, " grass.")]),
    ]

    texts = [s.text for s in build_sentences(fragments, Settings())]

    assert texts == ["You know that grass."]


def test_a_very_long_pause_always_ends_a_sentence():
    fragments = [
        frag([(0.0, 0.3, " The")]),
        frag([(4.0, 4.5, " gap"), (4.5, 5.0, " closes.")]),
    ]
    assert len(build_sentences(fragments, Settings())) == 2


def test_unpunctuated_speech_is_cut_at_its_longest_pause_not_at_a_ceiling():
    spec = [(i * 0.4, i * 0.4 + 0.3, f" w{i}") for i in range(40)]
    # One real pause in the middle of sixteen seconds of speech.
    spec = spec[:20] + [(s + 0.9, e + 0.9, t) for s, e, t in spec[20:]]

    sentences = build_sentences([frag(spec)], Settings())

    assert [s.text.split()[0] for s in sentences][:2] == ["w0", "w20"]


def test_japanese_is_joined_without_spaces_and_cut_at_punctuation():
    fragments = [
        frag([(3.1, 3.4, "ねえ"), (3.4, 3.6, "ねえ"), (3.6, 3.7, "、"), (3.7, 4.1, "こっち来て。")]),
        frag([(4.2, 4.6, "ほら"), (4.6, 4.7, "、"), (4.7, 6.8, "人間いたって言ってるよ。")]),
    ]

    texts = [s.text for s in build_sentences(fragments, Settings())]

    assert texts == ["ねえねえ、こっち来て。", "ほら、人間いたって言ってるよ。"]


def test_a_pause_measured_inside_a_japanese_word_is_ignored():
    """Whisper put 3.7 s between そ and こに — an alignment artefact, not a pause."""
    fragments = [frag([(7.07, 7.39, "そ"), (11.06, 11.5, "こに"), (11.5, 12.14, "人間いたいわ。")])]

    texts = [s.text for s in build_sentences(fragments, Settings())]

    assert texts == ["そこに人間いたいわ。"]


class TestCaptions:
    def test_a_human_line_break_is_kept_even_mid_sentence(self):
        fragments = [
            Fragment(0.0, 2.0, "so we're going to head"),
            Fragment(2.0, 4.0, "to the red tower"),
        ]
        sentences = sentences_from_captions(fragments, Settings())
        assert [s.text for s in sentences] == ["so we're going to head", "to the red tower"]
        assert [s.index for s in sentences] == [0, 1]

    def test_only_an_overlong_line_is_split_and_at_its_punctuation(self):
        fragments = [Fragment(0.0, 16.0, "今日は大冒険ってことですね。めちゃくちゃ大冒険です")]
        sentences = sentences_from_captions(fragments, Settings())
        assert [s.text for s in sentences] == ["今日は大冒険ってことですね。", "めちゃくちゃ大冒険です"]
        assert sentences[0].end == pytest.approx(sentences[1].start)


def test_a_new_voice_ends_the_sentence():
    """No punctuation, no pause — but two people."""
    spec = [(0.0, 0.3, " no"), (0.3, 0.6, " way"), (0.65, 0.9, " yes"), (0.9, 1.2, " way")]
    ws = words(spec)
    for w, sp in zip(ws, ["1", "1", "2", "2"]):
        w.speaker = sp
    sentences = build_sentences([Fragment(0.0, 1.2, "no way yes way", ws)], Settings())
    assert [(s.text, s.speaker) for s in sentences] == [("no way", "1"), ("yes way", "2")]


def test_whispers_spaces_split_fast_japanese_cross_talk():
    """VlaTbwhSSqk: four voices in four seconds, marked only by Whisper's spaces
    and with word timings a uniform 0.1 s apart."""
    spec = [(31.96 + 0.1 * i, 32.06 + 0.1 * i, t) for i, t in enumerate(
        ["繋", "い", "で", "遊", "ぶ", " 初", "めて", "話", "して", "る",
         " 話", "した", "こと", "ない", "よ", " 全", "然", "私", "友", "達", "じゃない", "やん"])]

    texts = [s.text for s in build_sentences([frag(spec)], Settings())]

    assert texts == ["繋いで遊ぶ", "初めて話してる", "話したことないよ", "全然私友達じゃないやん"]


def test_an_unpunctuated_japanese_segment_end_cuts_only_with_a_pause_behind_it():
    """The sequential decoder writes no punctuation; its segment ends are the only
    marks. One after a pause ends the sentence; one after a dangling が does not."""
    paused = [
        frag([(119.7, 120.2, "今日は"), (120.2, 121.3, "カルビーの"), (121.3, 122.8, "ことを友達って呼んだよ")]),
        frag([(123.5, 124.0, "わか"), (124.0, 125.26, "んないんだよ")]),
    ]
    run_on = [
        frag([(84.25, 85.95, "すいませんね先輩が")]),
        frag([(85.95, 87.01, "ちょっと先に")]),
        frag([(87.01, 88.31, "火つけられます")]),
    ]

    assert [s.text for s in build_sentences(paused, Settings())] == [
        "今日はカルビーのことを友達って呼んだよ",
        "わかんないんだよ",
    ]
    assert [s.text for s in build_sentences(run_on, Settings())] == [
        "すいませんね先輩がちょっと先に火つけられます"
    ]


def test_a_voice_change_inside_a_japanese_word_moves_to_the_nearest_break():
    """Diarization put the turn edge at わかん|ない; the line breaks at the comma."""
    spec = [(0.0, 0.4, "そう"), (0.4, 0.5, "、"), (0.5, 0.9, "なんか"), (0.9, 1.3, "わか"),
            (1.3, 1.5, "ん"), (1.5, 1.8, "ない"), (1.8, 2.0, "、"),
            (2.0, 2.4, "お疲れ"), (2.4, 2.8, "とか"), (2.8, 3.2, "言って"), (3.2, 3.5, "た")]
    ws = words(spec)
    for w in ws:
        w.speaker = "1" if w.start < 1.5 else "2"

    sentences = build_sentences([Fragment(0.0, 3.5, "", ws)], Settings())

    assert [s.text for s in sentences] == ["そう、なんかわかんない、", "お疲れとか言ってた"]


def test_latin_pieces_inside_japanese_stay_one_word():
    """Whisper hands back "VC" as V + C and "1700" as 17 + 00."""
    spec = [(47.2, 47.56, "V"), (47.56, 47.76, "C"), (47.76, 48.14, "繋"), (48.14, 48.5, "がない"),
            (48.6, 48.8, "で"), (270.62, 271.0, "17"), (271.0, 272.67, "00円")]

    texts = [s.text for s in build_sentences([frag(spec)], Settings())]

    assert texts == ["VC繋がないで", "1700円"]
