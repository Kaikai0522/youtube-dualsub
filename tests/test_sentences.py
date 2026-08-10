"""Rebuilding whole sentences out of Whisper's fragments."""

from __future__ import annotations

import pytest

from youtube_dualsub.config import Settings
from youtube_dualsub.models import Fragment, Word
from youtube_dualsub.pipeline.sentences import build_sentences


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
