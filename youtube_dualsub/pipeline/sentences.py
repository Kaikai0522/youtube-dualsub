"""Rebuild ASR fragments into whole spoken sentences.

Whisper cuts on its own 30-second windows, not on meaning, so a raw fragment
list looks like::

    [0.0-2.1] "Wait he's got"
    [2.1-4.4] "gapples. We're"
    [4.4-5.0] "cooked."

Translating those three pieces independently produces three pieces of garbage,
because each one is missing the half of the thought that makes it translatable.
Everything downstream — the sliding-window context, the glossary, the batch
alignment — assumes it is working with whole sentences, and this is where that
becomes true.

Sentences are cut at, in priority order: sentence-final punctuation, a pause
longer than ``pause_break_s``, and finally the hard duration/length ceilings so
that a run-on monologue cannot become one 90-second cue.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..models import Fragment, Sentence, Word


def build_sentences(fragments: list[Fragment], settings: Settings) -> list[Sentence]:
    cfg = settings.sentences
    tokens = _tokens(fragments)
    if not tokens:
        return []

    sentences: list[Sentence] = []
    current: list[Word] = []

    for i, token in enumerate(tokens):
        current.append(token)
        if i + 1 >= len(tokens):
            break

        nxt = tokens[i + 1]
        text_so_far = _join(current)
        gap = nxt.start - token.end
        duration = token.end - current[0].start

        ends_sentence = text_so_far.rstrip().endswith(tuple(cfg.sentence_end_chars))
        long_enough = len(text_so_far) >= cfg.min_fragment_chars

        if (
            (ends_sentence and long_enough)
            or gap >= cfg.pause_break_s
            or duration >= cfg.max_sentence_s
            or len(text_so_far) >= cfg.max_sentence_chars
        ):
            sentences.append(_emit(len(sentences), current))
            current = []

    if current:
        sentences.append(_emit(len(sentences), current))

    return _absorb_stubs(sentences, cfg.min_fragment_chars)


def _tokens(fragments: list[Fragment]) -> list[Word]:
    """A single stream of timed units.

    Fragments that carry word timestamps contribute their words; fragments that
    do not (human-uploaded captions, for instance) contribute themselves as one
    coarse unit. Both cases flow through the same grouping logic.
    """
    tokens: list[Word] = []
    for frag in fragments:
        if frag.words:
            tokens.extend(w for w in frag.words if w.text.strip())
        elif frag.text.strip():
            tokens.append(Word(start=frag.start, end=frag.end, text=frag.text.strip()))
    tokens.sort(key=lambda w: (w.start, w.end))
    return tokens


def _join(words: list[Word]) -> str:
    return _tidy("".join(w.text if w.text.startswith(" ") else " " + w.text for w in words))


_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%)\]])")


def _tidy(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()


def _emit(index: int, words: list[Word]) -> Sentence:
    return Sentence(
        index=index,
        start=round(words[0].start, 3),
        end=round(words[-1].end, 3),
        text=_join(words),
    )


def _absorb_stubs(sentences: list[Sentence], min_chars: int) -> list[Sentence]:
    """Glue "Yeah." / "What?" stubs onto their neighbour.

    A one-word sentence has no context to translate against and would become a
    cue that flashes for a fifth of a second. Merging is the honest fix; the
    alternative — holding it on screen longer — desynchronises the subtitle
    from the speaker, which is exactly what must not happen.
    """
    if not sentences:
        return []

    merged: list[Sentence] = []
    for sentence in sentences:
        if merged and len(sentence.text) < min_chars:
            prev = merged[-1]
            prev.text = _tidy(f"{prev.text} {sentence.text}")
            prev.end = sentence.end
            continue
        merged.append(sentence)

    for i, sentence in enumerate(merged):
        sentence.index = i
    return merged
