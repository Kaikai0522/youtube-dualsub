"""Fill the lines Whisper missed with YouTube's own auto-captions.

Whisper loses shouted speech before it ever decodes it. Silero VAD scored a
scream of "サクサマー！やばーい！" at 0.29 at most, under the 0.35 threshold, so
the decoder was never shown it; handed the bare clip, Whisper wrote it down.
Lowering the threshold is not the fix: at 0.2 the VAD glues speech into longer
chunks and Whisper skips more inside them — on a human-captioned clip, lines
mostly missed went 9 -> 16 of 48.

YouTube's recogniser hears those stretches. Against the same human captions its
auto-captions matched Whisper (character recall 84.9 % vs 85.3 %, precision
93.1 % vs 92.0 %) while mostly missing 4 lines instead of 9. So its lines are
added where two things agree that Whisper has nothing:

* **text** — Whisper's words within ``_SLACK_S`` hold under a third of the
  line's characters, and
* **time** — Whisper's words cover under half of the line's time.

Text alone added duplicates written another way (LINE低く beside Whisper's
ライン引く); time alone found nothing, because Whisper usually has *some* word
near the line it skipped. Both together, on the human-captioned clip: 2 lines
added, 2 right. Re-decoding the gaps with Whisper, measured earlier, was right
30 % of the time (AGENTS.md) — this is a different recogniser, not a second try
of the same one.

**The text is YouTube's; the timing is Whisper's** (``retime``). Most auto-caption
lines carry no word offsets, only YouTube's display window, which is not when
the words were said: on ``v-V9NpCvPDk`` 何それ?何それ?やばい。ついてきてる。 was
windowed 433.68–436.68 s and spoken 433.2–440.2 s, so the subtitle left while
the speaker was still talking. Whisper, handed that stretch without the VAD,
heard the line with good word timings. Its words are used only to find where
YouTube's text was spoken — a Whisper hallucination can move a line but never
put words on screen. Where decoding found nothing to match (it loops on shouting:
14 of 20 lines on ``VlaTbwhSSqk``), YouTube's text is force-aligned onto the
audio instead (``retime_forced``); only an alignment that collapsed keeps
YouTube's timing.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..models import Fragment, Word
from ..sources.ytdlp_source import AutoLine

_SLACK_S = 1.0
_MAX_TEXT_FOUND = 0.34
_MAX_TIME_COVERED = 0.5
_MIN_CHARS = 2

#: Sound tags YouTube writes for non-speech: [笑い], [叫び声], [音楽], [Music].
_SOUND_TAG = re.compile(r"\[[^\]]*\]")
_NOT_TEXT = re.compile(r"[\s、。，,.!?！？…ー〜~「」『』()（）・♪\-]")


def fill_gaps(fragments: list[Fragment], lines: list[AutoLine]) -> list[Fragment]:
    """The YouTube lines Whisper missed, as fragments with word timings."""
    heard = [
        (w.start, w.end, w.text)
        for f in fragments
        for w in (f.words or [Word(start=f.start, end=f.end, text=f.text)])
    ]
    spoken = []
    for start, end, pieces in lines:
        pieces = [(t, _SOUND_TAG.sub("", text)) for t, text in pieces]
        pieces = [(t, text) for t, text in pieces if text.strip()]
        if pieces:
            spoken.append((start, end, pieces))

    added: list[Fragment] = []
    for i, (start, end, pieces) in enumerate(spoken):
        # A line's json3 window stays up while the next one rolls in; the next
        # *spoken* line's start is where this one's speech ends. A [笑い] tag
        # rolling in 0.43 s later does not end a 31-character sentence.
        if i + 1 < len(spoken) and start < spoken[i + 1][0] < end:
            end = spoken[i + 1][0]
        wanted = _letters("".join(text for _, text in pieces))
        if len(wanted) < _MIN_CHARS or end <= start:
            continue
        if _covered(heard, start, end) >= _MAX_TIME_COVERED:
            continue
        nearby = _letters(
            "".join(text for s, e, text in heard if s < end + _SLACK_S and e > start - _SLACK_S)
        )
        if _found(wanted, nearby) >= _MAX_TEXT_FOUND:
            continue
        # The first word cannot have been said before the previous line's last.
        earliest = spoken[i - 1][2][-1][0] if i else 0.0
        pieces, end = _drop_heard_edges(pieces, end, heard, earliest)
        if len(_letters("".join(text for _, text in pieces))) < _MIN_CHARS:
            continue
        added.append(_fragment(pieces, end))
    return added


#: A json3 line's first piece carries the line's start, not its own time: it
#: was said up to this long before. On d1MzjzSTwvc これ was stamped 19.40 s and
#: said at 12.72 s, 投げ stamped 87.28 s and said at 84.65 s; the other pieces
#: matched Whisper within 0.3 s. Across four videos 11-73 % of first pieces
#: were over 1 s late, almost none early.
_FIRST_LAG_S = 7.0


def _drop_heard_edges(
    pieces: list[tuple[float, str]], end: float, heard: list[tuple[float, float, str]], earliest: float
) -> tuple[list[tuple[float, str]], float]:
    """Drop the line's first and last pieces when Whisper already has them.

    YouTube glues a word Whisper heard onto the line it missed: the これ said at
    12.72 s headed a line of speech from 20.4 s, and shown there it repeated a
    word six seconds after it was said. The line passed ``fill_gaps``'s checks
    on the words Whisper missed; the words it did not miss are left to Whisper.
    """
    pieces = list(pieces)
    if len(pieces) > 1:
        anchor = pieces[1][0]
        lo = max(earliest, anchor - _FIRST_LAG_S) - _SLACK_S
        before = _letters("".join(text for s, e, text in heard if lo <= s < anchor))
        first = _letters(pieces[0][1])
        if len(first) >= _MIN_CHARS and before.endswith(first):
            pieces.pop(0)
    while len(pieces) > 1:
        t, text = pieces[-1]
        last = _letters(text)
        near = _letters(
            "".join(w for s, e, w in heard if s < end + _SLACK_S and e > t - _SLACK_S)
        )
        if len(last) < _MIN_CHARS or last not in near:
            break
        pieces.pop()
        end = t
    return pieces, end


def _fragment(pieces: list[tuple[float, str]], end: float) -> Fragment:
    words = []
    for k, (t, text) in enumerate(pieces):
        stop = pieces[k + 1][0] if k + 1 < len(pieces) else end
        words.append(Word(start=t, end=max(stop, t + 0.05), text=text))
    return Fragment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.text for w in words).strip(),
        words=words,
    )


#: How far around YouTube's window Whisper listens for the line. The measured
#: overrun was 3.5 s past the window's end.
_REACH_BEFORE_S = 1.5
_REACH_AFTER_S = 5.0
#: Whisper must have heard this share of the line's letters to be trusted with
#: its timing. Kanji against kana (何それ / なにそれ) is why it is not higher.
_MIN_ALIGNED = 0.5
#: Single-character matches are noise in CJK text: い, て and か are everywhere.
_MIN_BLOCK = 2


def retime_windows(added: list[Fragment]) -> list[tuple[float, float]]:
    """Where to listen for each added line: YouTube's window, widened.

    Not back by ``_FIRST_LAG_S``, although the first word may be that early:
    measured (2026-09-30), it found 投げえ at 84.66 s instead of 85.78 s, but the
    line then matched an earlier repeat of its own words — on v-V9NpCvPDk
    何それ?何それ? moved from 433.6 to 426.7 s, こいつやば 9 s early on VlaTbwhSSqk.
    """
    return [(max(0.0, f.start - _REACH_BEFORE_S), f.end + _REACH_AFTER_S) for f in added]


def retime(added: list[Fragment], heard: list[list[Word]]) -> list[Fragment]:
    """Move each line onto the span where Whisper heard its text; returns the rest.

    ``heard[i]`` is Whisper's words (absolute times) in ``retime_windows(added)[i]``.
    The lines returned are left for ``retime_forced``.
    """
    unmoved = []
    for fragment, words in zip(added, heard):
        span = aligned_span(fragment.text, words)
        if span is not None:
            _move(fragment, *span)
        else:
            unmoved.append(fragment)
    return unmoved


#: A forced alignment's first and last word absorb the silence at the window's
#: edges (配 stretched 2.10-2.40 s, 。 234.12-239.16 s), so each is cut to this
#: much per letter, and never under the floor.
_EDGE_PER_LETTER_S = 0.1
_EDGE_FLOOR_S = 0.2
#: Even trimmed, the alignment runs early; YouTube's own start rarely does.
#: Not starting more than this before it: start errors over 1 s 23 % -> 16 %.
_MAX_EARLIER_S = 1.0
#: When the text is not in the window, DTW piles it onto the window's last
#: frames: 全員だよな landed as four zero-length words at 232.62 s, 8 s late.
_COLLAPSED_S = 0.021


def retime_forced(lines: list[Fragment], aligned: list[list[Word]]) -> int:
    """Move each line onto its forced alignment; returns how many moved.

    For the lines ``retime`` could not place: Whisper decoding their window
    looped (ああ ×30) or heard something else. ``aligned[i]`` is
    ``asr.align_texts`` for ``lines[i]`` over ``retime_windows(lines)[i]``.

    Measured (2026-10-28) on 199 YouTube lines Whisper's main pass also heard,
    against Whisper's own word times, for the 29 that decoding could not
    match: ends off by over 1 s 34 % -> 14 % (median 0.71 -> 0.05 s), starts
    28 % -> 17 %. YouTube's end is the next line's start, not the speaker's.
    """
    moved = 0
    for fragment, words in zip(lines, aligned):
        span = forced_span(fragment.start, words)
        if span is not None:
            _move(fragment, *span)
            moved += 1
    return moved


def forced_span(youtube_start: float, words: list[Word]) -> tuple[float, float] | None:
    """(start, end) of a forced alignment, or None when it collapsed."""
    spoken = [w for w in words if _said(w.text)]
    letters = sum(_said(w.text) for w in spoken)
    if not letters:
        return None
    piled = sum(_said(w.text) for w in spoken if w.end - w.start < _COLLAPSED_S)
    if piled * 2 >= letters:
        return None
    first, last = spoken[0], spoken[-1]
    start = max(first.start, first.end - _edge(first), youtube_start - _MAX_EARLIER_S)
    end = min(last.end, last.start + _edge(last))
    return (start, end) if end > start else None


def _edge(word: Word) -> float:
    return max(_EDGE_FLOOR_S, _EDGE_PER_LETTER_S * _said(word.text))


def aligned_span(text: str, words: list[Word]) -> tuple[float, float] | None:
    """(start, end) of the words that spell ``text``, or None when too few do."""
    wanted = _letters(text)
    owner: list[int] = []
    for i, word in enumerate(words):
        owner.extend([i] * len(_letters(word.text)))
    spoken = "".join(_letters(w.text) for w in words)
    if not wanted or not spoken:
        return None

    least = min(_MIN_BLOCK, len(wanted))
    blocks = [
        b
        for b in SequenceMatcher(None, wanted, spoken, autojunk=False).get_matching_blocks()
        if b.size >= least
    ]
    if sum(b.size for b in blocks) / len(wanted) < _MIN_ALIGNED:
        return None

    # Letters of the line that did not match at its edges (何 against なに) were
    # still said, just before the first match or after the last one.
    first, last = blocks[0], blocks[-1]
    lo = max(0, first.b - first.a)
    hi = min(len(spoken) - 1, last.b + last.size - 1 + len(wanted) - last.a - last.size)
    start, end = words[owner[lo]].start, words[owner[hi]].end
    return (start, end) if end > start else None


def _move(fragment: Fragment, start: float, end: float) -> None:
    """Stretch the line's words from YouTube's window onto [start, end]."""
    old_start, old_end = fragment.start, fragment.end
    scale = (end - start) / (old_end - old_start) if old_end > old_start else 0.0
    for word in fragment.words:
        word.start = start + (word.start - old_start) * scale
        word.end = start + (word.end - old_start) * scale
    fragment.start, fragment.end = start, end


#: Punctuation only: unlike ``_letters`` it keeps ー, which is a held vowel
#: (the ケー of オッケー lasts as long as it is held).
_PUNCT = re.compile(r"[\s、。，,.!?！？…「」『』()（）・♪\-]")


def _said(text: str) -> int:
    """How many spoken characters ``text`` holds."""
    return len(_PUNCT.sub("", text))


def _letters(text: str) -> str:
    return _NOT_TEXT.sub("", text)


def _found(wanted: str, nearby: str) -> float:
    blocks = SequenceMatcher(None, wanted, nearby, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(wanted)


def _covered(heard: list[tuple[float, float, str]], start: float, end: float) -> float:
    """Share of [start, end] under some Whisper word."""
    spans = sorted((max(start, s), min(end, e)) for s, e, _ in heard if s < end and e > start)
    total, cursor = 0.0, start
    for s, e in spans:
        s = max(s, cursor)
        if e > s:
            total += e - s
            cursor = e
    return total / (end - start)


__all__ = [
    "aligned_span",
    "fill_gaps",
    "forced_span",
    "retime",
    "retime_forced",
    "retime_windows",
]
