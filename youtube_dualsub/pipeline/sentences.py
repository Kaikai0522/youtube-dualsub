"""Turn a transcript into translation units: one spoken thought each.

Two very different inputs arrive here.

**Human captions** were already cut by a person who watched the video. Their
line breaks are the best segmentation available, so they are kept as they are
(``sentences_from_captions``). A caption line may stop mid-sentence; the
translator sees the lines around it and translates it as the half it is.

**Whisper** hands back a stream of timed word tokens cut on its own 30-second
windows, not on meaning::

    [157.55] "You"  (pause)  [158.74] "know that grass breaking behind you, Dream? Oh"
    (pause) [162.55] "my God, I'm catching you."

Cutting at every pause — the first version of this module — produced exactly
those three subtitles. ``build_sentences`` instead scores every gap between two
tokens as a candidate boundary and picks the set of boundaries that is best
*overall*, by dynamic programming:

* sentence-final punctuation, a long pause and a Whisper segment end all make
  a boundary more attractive;
* a following lowercase word, an article or preposition before the gap, or a
  Japanese case particle before it make it less attractive — those are the
  signs of a sentence that has not finished;
* every sentence pays a fixed price, a stub pays extra, and one that runs past
  the target length pays in proportion, so the optimum is neither one word per
  line nor one line per minute.

The same code handles English and Japanese because lengths are measured in
visible characters (``lang.visible_length``) and tokens are joined by
``lang.join_tokens``, which knows Japanese takes no spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..lang import (
    CLAUSE_BREAK,
    CLOSERS,
    SENTENCE_END,
    ends_clause,
    ends_sentence,
    is_cjk_char,
    join_tokens,
    tidy,
    visible_length,
)

from ..models import Fragment, Sentence, Word

#: Words a spoken English sentence essentially never ends on.
_DANGLING_EN = frozenset(
    "a an the to of in on at with for from by into about my your his her their our its "
    "and but or nor because if that i i'm i've we're you're it's".split()
)
#: Japanese case particles: a clause ending in one is still waiting for its verb.
_DANGLING_JA = "はがをにへとのもや"

_SENTENCE_COST = 2.0
_SPEAKER_CHANGE_GAIN = 4.0
_CJK_SPACE_GAIN = 3.0
_CJK_SEGMENT_END_GAIN = 2.0
#: How far a speaker change may be moved to reach a place a sentence can end.
_VOICE_REACH_S = 0.6

_STUB_CHARS = 4
_STUB_COST = 1.5
_OVERLONG_WEIGHT = 4.0


@dataclass(slots=True)
class _Token:
    start: float
    end: float
    text: str
    segment_end: bool = False
    speaker: str | None = None


def build_sentences(fragments: list[Fragment], settings: Settings) -> list[Sentence]:
    """Segment a Whisper transcript. See the module docstring."""
    tokens = _tokens(fragments)
    if not tokens:
        return []

    cfg = settings.sentences
    n = len(tokens)
    voices = _voice_changes(tokens)
    gains = [_boundary_gain(tokens[i], tokens[i + 1], voices[i]) for i in range(n - 1)]
    forced = [
        tokens[i + 1].start - tokens[i].end >= cfg.hard_pause_s
        and not _inside_cjk_word(tokens[i], tokens[i + 1])
        for i in range(n - 1)
    ]
    widths = [max(1, visible_length(t.text)) for t in tokens]

    inf = float("inf")
    best = [0.0] + [inf] * n
    back = [0] * (n + 1)
    for end in range(n):  # the sentence is tokens[start..end]
        width = 0
        for start in range(end, -1, -1):
            width += widths[start]
            span = tokens[end].end - tokens[start].start
            if start < end and (width > cfg.max_chars or span > cfg.max_sentence_s):
                break
            if best[start] == inf:
                continue
            cost = best[start] + _sentence_cost(width, span, settings)
            if end < n - 1:
                cost -= gains[end]
            if cost < best[end + 1]:
                best[end + 1] = cost
                back[end + 1] = start
            if start > 0 and forced[start - 1]:
                break

    cuts: list[tuple[int, int]] = []
    end = n
    while end > 0:
        start = back[end]
        cuts.append((start, end))
        end = start
    cuts.reverse()

    return [
        Sentence(
            index=i,
            start=round(tokens[a].start, 3),
            end=round(tokens[b - 1].end, 3),
            text=_join(tokens[a:b]),
            speaker=_main_speaker(tokens[a:b]),
        )
        for i, (a, b) in enumerate(cuts)
    ]


def _join(tokens: list[_Token]) -> str:
    """Whisper splits "VC" and "1700" into pieces with no space between them;
    keep them one word instead of letting ``join_tokens`` space them apart."""
    parts: list[str] = []
    for prev, token in zip([None, *tokens], tokens):
        if parts and prev and not prev.segment_end and not token.text[:1].isspace():
            parts[-1] += token.text
        else:
            parts.append(token.text)
    return join_tokens(parts)


def _main_speaker(tokens: list[_Token]) -> str | None:
    talk: dict[str, float] = {}
    for t in tokens:
        if t.speaker:
            talk[t.speaker] = talk.get(t.speaker, 0.0) + max(0.01, t.end - t.start)
    return max(talk, key=talk.get) if talk else None


def sentences_from_captions(fragments: list[Fragment], settings: Settings) -> list[Sentence]:
    """Keep a human's line breaks; only split a line that runs far too long."""
    limit = settings.sentences.caption_max_s
    pieces: list[tuple[float, float, str]] = []
    for frag in sorted(fragments, key=lambda f: f.start):
        text = tidy(frag.text)
        if text:
            pieces.extend(_split_caption(frag.start, frag.end, text, limit))
    return [
        Sentence(index=i, start=round(s, 3), end=round(e, 3), text=t)
        for i, (s, e, t) in enumerate(pieces)
    ]


# --------------------------------------------------------------------------


def _tokens(fragments: list[Fragment]) -> list[_Token]:
    """A single stream of timed units.

    Fragments that carry word timestamps contribute their words; fragments that
    do not contribute themselves as one coarse unit.
    """
    tokens: list[_Token] = []
    for frag in sorted(fragments, key=lambda f: f.start):
        words: list[Word] = [w for w in frag.words if w.text.strip()]
        if not words and frag.text.strip():
            words = [Word(start=frag.start, end=frag.end, text=frag.text.strip())]
        for i, w in enumerate(words):
            tokens.append(
                _Token(
                    start=w.start,
                    end=w.end,
                    text=w.text,
                    segment_end=i == len(words) - 1,
                    speaker=w.speaker or frag.speaker,
                )
            )
    return tokens


def _inside_cjk_word(left: _Token, right: _Token) -> bool:
    """Two Japanese sub-word tokens inside one Whisper segment, with no
    punctuation between them.

    Whisper's word timings are least reliable exactly here: it put a
    3.7-second "pause" between そ and こに人間いたいわ, and 0.8 s between 金 and
    の玉. A gap measured inside a Japanese word is an alignment artefact, not a
    pause anyone made. The same holds for the Latin letters and digits inside
    Japanese speech, which arrive as pieces with no space: V | C, 17 | 00円.
    """
    before, after = left.text.strip(), right.text.strip()
    if left.segment_end or not before or not after:
        return False
    if is_cjk_char(before[-1]) and is_cjk_char(after[0]):
        return not _cjk_space(left, right)
    return not right.text[:1].isspace() and before[-1].isalnum() and after[0].isalnum()


def _cjk_space(left: _Token, right: _Token) -> bool:
    """A space Whisper wrote between two Japanese tokens.

    Japanese tokens arrive without spaces, so a leading one is Whisper's own
    mark, and in fast cross-talk it is often the only mark there is: "繋いで遊ぶ
    初めて話してる 話したことないよ 全然私友達じゃないやん" is several voices in
    4 seconds, with word timings spaced a uniform 0.1 s apart that no pause can
    be read from.
    """
    return right.text[:1].isspace() and _between_cjk(left, right)


def _between_cjk(left: _Token, right: _Token) -> bool:
    before = left.text.strip().rstrip(SENTENCE_END + CLAUSE_BREAK + CLOSERS)
    after = right.text.strip()
    return (
        bool(before)
        and bool(after)
        and is_cjk_char(before[-1])
        and is_cjk_char(after[0])
    )


def _voice_changes(tokens: list[_Token]) -> list[bool]:
    """For each gap, whether a different person speaks after it.

    Turn edges are only word-accurate, and in Japanese the word they land on
    is usually mid-word — わかん|ない, カルビーの|こと — where no sentence may
    end, so the change would be thrown away. It moves instead to the nearest
    gap within ``_VOICE_REACH_S`` where a sentence can end.
    """
    n = len(tokens)
    trailing = SENTENCE_END + CLAUSE_BREAK + CLOSERS
    open_gap = [
        not _inside_cjk_word(tokens[i], tokens[i + 1])
        and tokens[i + 1].text.strip()[:1] not in trailing
        for i in range(n - 1)
    ]
    changes = [False] * (n - 1)
    for i in range(n - 1):
        a, b = tokens[i].speaker, tokens[i + 1].speaker
        if not (a and b and a != b):
            continue
        at = tokens[i + 1].start
        best, best_dist = None, _VOICE_REACH_S
        for step in (-1, 1):
            j = i
            while 0 <= j < n - 1 and abs(tokens[j + 1].start - at) <= best_dist:
                if open_gap[j]:
                    best, best_dist = j, abs(tokens[j + 1].start - at)
                    break
                j += step
        if best is not None:
            changes[best] = True
    return changes


def _boundary_gain(left: _Token, right: _Token, new_voice: bool = False) -> float:
    """How good a place the gap between two tokens is to end a sentence."""
    before, after = left.text.strip(), right.text.strip()
    gap = right.start - left.end
    if _inside_cjk_word(left, right):
        return -3.0
    gain = 3.0 * min(1.5, max(0.0, (gap - 0.3) / 0.7))
    spaced = _cjk_space(left, right)

    if ends_sentence(before):
        gain += 4.0
        if after[:1].isupper():
            gain += 0.5
    elif ends_clause(before):
        gain += 1.0
    elif not new_voice and not spaced:
        # Signs of an unfinished sentence — which only mean anything when the
        # same person keeps talking.
        if after[:1].islower() and after[:1].isascii():
            gain -= 2.5  # "You" | "know that..." — the sentence goes on
        word = before.lower().strip(",.!?\"'")
        if word in _DANGLING_EN:
            gain -= 2.0
        elif before[-1:] in _DANGLING_JA:
            gain -= 1.0
    if spaced and not ends_sentence(before):
        gain += _CJK_SPACE_GAIN
    if left.segment_end:
        gain += 0.5
        if not ends_sentence(before) and not ends_clause(before) and _between_cjk(left, right):
            # The sequential decoder writes no punctuation at all, so its
            # segment ends are the only marks: 呼んだよ | わかんないんだよ. Weaker
            # than a space — alone it does not cut 先輩が | ちょっと先に — but a
            # pause or a new voice on top of it does.
            gain += _CJK_SEGMENT_END_GAIN
    # A new voice ends the sentence about as surely as a full stop. Not a hard
    # cut: a turn boundary is only accurate to a word or so, and forcing it
    # would strand that word as a one-word subtitle.
    if new_voice:
        gain += _SPEAKER_CHANGE_GAIN
    return gain


def _sentence_cost(width: int, span: float, settings: Settings) -> float:
    cfg = settings.sentences
    cost = _SENTENCE_COST
    if width < _STUB_CHARS + 1:
        cost += _STUB_COST
    over = max(0.0, width - cfg.target_chars) / cfg.target_chars
    over += max(0.0, span - cfg.target_s) / cfg.target_s
    return cost + _OVERLONG_WEIGHT * over


def _split_caption(
    start: float, end: float, text: str, limit: float
) -> list[tuple[float, float, str]]:
    if end - start <= limit:
        return [(start, end, text)]
    cut = _caption_cut(text)
    if cut is None:
        return [(start, end, text)]
    left, right = text[:cut].strip(), text[cut:].strip()
    share = visible_length(left) / max(1, visible_length(left) + visible_length(right))
    middle = start + (end - start) * share
    return _split_caption(start, middle, left, limit) + _split_caption(middle, end, right, limit)


def _caption_cut(text: str) -> int | None:
    """The split point nearest the middle, preferring punctuation, then spaces."""
    middle = len(text) / 2
    for accept in (
        lambda i: ends_sentence(text[:i]),
        lambda i: ends_clause(text[:i]),
        lambda i: text[i - 1].isspace() or text[i].isspace(),
    ):
        candidates = [i for i in range(1, len(text)) if accept(i)]
        if candidates:
            return min(candidates, key=lambda i: abs(i - middle))
    return None


__all__ = ["build_sentences", "sentences_from_captions"]
