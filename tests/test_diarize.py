"""Speaker turns -> speakers on words and lines, without the models."""

from __future__ import annotations

from youtube_dualsub.models import Fragment, Sentence, Word
from youtube_dualsub.pipeline.diarize import (
    Turn,
    TurnIndex,
    _drop_minor_speakers,
    _number_by_appearance,
    label_fragments,
    label_sentences,
)


def test_speakers_are_numbered_by_first_appearance():
    turns = _number_by_appearance([(5.0, 6.0, 3), (0.0, 1.0, 7), (2.0, 3.0, 3)])
    assert [(t.start, t.speaker) for t in turns] == [(0.0, "1"), (2.0, "2"), (5.0, "2")]


def test_clusters_that_barely_speak_are_forgotten():
    """Seven clusters on a three-person clip; four were laughter and noise."""
    raw = [(0, 100, 0), (100, 150, 1), (150, 180, 2), (180, 182, 3), (190, 191, 4)]
    kept = _drop_minor_speakers(raw)
    assert {c for _, _, c in kept} == {0, 1, 2}


def test_a_word_takes_the_speaker_it_overlaps_most():
    index = TurnIndex([Turn(0.0, 2.0, "1"), Turn(1.8, 4.0, "2")])
    assert index.speaker_at(0.5, 1.0) == "1"
    assert index.speaker_at(1.7, 2.6) == "2"


def test_a_word_just_outside_every_turn_takes_the_nearest():
    index = TurnIndex([Turn(0.0, 1.0, "1"), Turn(5.0, 6.0, "2")])
    assert index.speaker_at(1.2, 1.4) == "1"
    assert index.speaker_at(3.0, 3.2) is None


def test_long_turns_are_found_even_when_they_start_far_back():
    index = TurnIndex([Turn(0.0, 60.0, "1")] + [Turn(61.0 + i, 61.5 + i, "2") for i in range(50)])
    assert index.speaker_at(59.0, 59.5) == "1"


def test_labelling_words_and_caption_lines():
    turns = [Turn(0.0, 2.0, "1"), Turn(2.2, 4.0, "2")]
    fragment = Fragment(0.0, 3.5, "a b", [Word(0.1, 0.5, " a"), Word(2.5, 3.0, " b")])
    label_fragments([fragment], turns)
    assert [w.speaker for w in fragment.words] == ["1", "2"]

    line = Sentence(index=0, start=1.5, end=3.9, text="はい 熱くなりますので")
    label_sentences([line], turns)
    assert line.speaker == "2", "the line goes to whoever talks longest in it"
