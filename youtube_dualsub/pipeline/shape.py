"""Turn translated sentences into cues a human can actually read.

Fast, overlapping speech produces sentences that are three words long and last
a fifth of a second. Rendered faithfully they strobe. The rules below exist to
fix that *without* ever letting a subtitle drift away from the person speaking:

1. **Merge** anything shorter than ``merge_below_s`` into a neighbour. Merging
   is the only legitimate anti-flicker tool, because it keeps text and speech
   aligned.

2. **Split** a line longer than ``max_chars_zh`` into several cues, dividing the
   time in proportion to the text. Both languages are split at the same points
   so the pairing survives.

3. **Clamp** the end time. In priority order::

       end <= next cue's start          (absolute - never overlap the next line)
       end <= start + max_duration_s    (nothing loiters)
       end >= start + min_duration_s    (only if there is a gap to grow into)

   The minimum duration is deliberately the *weakest* rule. Holding a line for a
   readable one second sounds harmless right up to the moment it covers the next
   speaker, at which point the subtitle is lying about who is talking. When the
   two rules conflict, synchronisation wins and the line goes early.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..config import Settings
from ..lang import is_cjk, is_cjk_char, join
from ..models import Cue, Sentence, Translation, TranslationStatus

#: Never emit a cue shorter than this, even between two touching sentences.
_EPSILON = 0.12


def build_cues(
    sentences: list[Sentence],
    translations: list[Translation],
    settings: Settings,
    *,
    protect: Sequence[str] = (),
) -> list[Cue]:
    """``protect`` lists strings that must never be cut in half — glossary
    renderings, mostly. Chinese has no spaces, so without a list of known words
    an index-based split happily severs 種子碼 into 種 / 子碼."""
    cfg = settings.shape
    by_index = {t.index: t for t in translations}

    cues: list[Cue] = []
    for sentence in sentences:
        translation = by_index.get(sentence.index)
        if translation is None:
            # Not translated yet (streaming) or dropped: the original still has value.
            cues.append(
                Cue(
                    start=sentence.start,
                    end=sentence.end,
                    source=sentence.text,
                    target="",
                    speaker=sentence.speaker,
                    translated=False,
                )
            )
            continue
        translated = translation.status is TranslationStatus.OK
        cues.append(
            Cue(
                start=sentence.start,
                end=sentence.end,
                source=sentence.text,
                target=translation.text if translated else "",
                speaker=sentence.speaker,
                translated=translated,
            )
        )

    cues.sort(key=lambda c: (c.start, c.end))
    cues = _merge_short(cues, cfg.merge_below_s, cfg.max_duration_s, cfg.merge_max_gap_s)
    cues = _rejoin_severed(cues, protect, cfg.merge_max_gap_s)
    cues = _split_long(
        cues, cfg.max_chars_zh, cfg.max_chars_en, protect, cfg.max_chars_ja,
        min_piece_s=cfg.min_duration_s,
    )
    return _clamp(cues, cfg.max_duration_s, cfg.min_duration_s, cfg.cue_gap_s)


# --------------------------------------------------------------------------
# 1. merge
# --------------------------------------------------------------------------

def _merge_short(
    cues: list[Cue], threshold: float, max_duration: float, max_gap: float
) -> list[Cue]:
    if not cues:
        return []

    out: list[Cue] = [cues[0]]
    for cue in cues[1:]:
        prev = out[-1]
        if prev.duration < threshold and _adjacent(prev, cue, max_duration, max_gap):
            out[-1] = _join(prev, cue)
        else:
            out.append(cue)

    # A final stub has no successor to absorb it, so fold it backwards.
    if (
        len(out) > 1
        and out[-1].duration < threshold
        and _adjacent(out[-2], out[-1], max_duration, max_gap)
    ):
        tail = out.pop()
        out[-1] = _join(out[-1], tail)
    return out


def _adjacent(a: Cue, b: Cue, max_duration: float, max_gap: float) -> bool:
    """Whether two cues are close enough to become one.

    The gap ceiling is the important half. Merging across a silence would put
    the text on screen well before it is spoken — the exact desynchronisation
    the shaping rules exist to prevent.
    """
    return (
        (b.start - a.end) <= max_gap
        and (b.end - a.start) <= max_duration
        and a.speaker == b.speaker
    )


def _join(a: Cue, b: Cue, *, separate: bool = True) -> Cue:
    """One cue out of two.

    ``separate`` keeps two utterances visibly apart: in Chinese and Japanese,
    where nothing else would, a space goes between them unless punctuation
    already does the job (初めて話してる 話したことないよ). Without it the pair
    read as one sentence, and the splitter, seeing no mark, cut mid-word
    (話したことな / いよ). Repairing a severed word joins with no space.
    """
    source, target = join(a.source, b.source), _join_text(a.target, b.target, cjk=True)
    if separate:
        source = _spaced(a.source, b.source) or source
        target = _spaced(a.target, b.target) or target
    return Cue(
        start=a.start,
        end=max(a.end, b.end),
        source=source,
        target=target,
        speaker=a.speaker or b.speaker,
        translated=a.translated and b.translated,
    )


def _spaced(a: str, b: str) -> str | None:
    """``a b`` when both sides are CJK and nothing marks the break, else None."""
    a, b = a.strip(), b.strip()
    if a and b and is_cjk_char(a[-1]) and is_cjk_char(b[0]):
        return f"{a} {b}"
    return None


def _join_text(a: str, b: str, *, cjk: bool) -> str:
    a, b = a.strip(), b.strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a}{b}" if cjk else f"{a} {b}"


def _rejoin_severed(cues: list[Cue], protect: Sequence[str], max_gap: float) -> list[Cue]:
    """Repair word breaks inherited from the boundary between two sentences.

    Splitting only runs on lines that exceed the limit, so a word severed
    *upstream* — the translator ending one line with "Minecraft M" and starting
    the next with "anhunt" — sails straight through untouched. Joining the pair
    back together lets the splitter redo the break in the right place, or leaves
    it as one line if it now fits.
    """
    if not cues:
        return []

    out: list[Cue] = [cues[0]]
    for cue in cues[1:]:
        prev = out[-1]
        # Only temporal adjacency is required here. The duration ceiling that
        # governs anti-flicker merging does not apply: splitting runs straight
        # after this and will re-divide both the text and its time.
        joinable = (cue.start - prev.end) <= max_gap and prev.speaker == cue.speaker
        if joinable and _severs_a_word(prev.target, cue.target, protect):
            out[-1] = _join(prev, cue, separate=False)
        else:
            out.append(cue)
    return out


def _severs_a_word(before: str, after: str, protect: Sequence[str]) -> bool:
    if not before or not after:
        return False
    # A Latin or numeric token running straight across the join.
    if re.search(r"[A-Za-z0-9]$", before) and re.match(r"^[A-Za-z0-9]", after):
        return True
    for term in protect:
        for k in range(1, len(term)):
            if before.endswith(term[:k]) and after.startswith(term[k:]):
                return True
    return False


# --------------------------------------------------------------------------
# 2. split
# --------------------------------------------------------------------------

_CJK_BREAKS = "，。！？；：、,.!?;: "


def _split_long(
    cues: list[Cue],
    max_zh: int,
    max_en: int,
    protect: Sequence[str] = (),
    max_ja: int | None = None,
    min_piece_s: float = 0.0,
) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        cjk_source = is_cjk(cue.source)
        max_src = (max_ja or max_zh) if cjk_source else max_en
        # Whichever language needs more pieces decides; the other follows.
        # A long English line under a short translation used to stay one
        # screen-wide cue (or lose its translation when split alone).
        src_pieces = 1 if cjk_source else len(_split_words(cue.source, max_src))
        parts_zh = (
            _split_cjk(cue.target, max_zh, protect, pieces=src_pieces, words=True)
            if cue.target
            else []
        )
        n = len(parts_zh)
        if n <= 1 and len(cue.source) <= max_src:
            out.append(cue)
            continue
        if n <= 1:
            # Very long source over a short (or no) translation: split the
            # source alone. A translation that fits stays whole on one piece.
            parts_en = (
                _split_cjk(cue.source, max_src, protect)
                if cjk_source
                else _split_words(cue.source, max_src)
            )
            n = len(parts_en)
            if cue.target and n > 1:
                out.append(_replace_text(cue, "\n".join(parts_en), cue.target))
                continue
            parts_zh = [""] * n
        elif cjk_source:
            parts_en = _distribute_chars(cue.source, [len(p) for p in parts_zh], protect)
        else:
            parts_en = _distribute_words(cue.source, [len(p) for p in parts_zh], max_src)

        if n <= 1:
            out.append(cue)
            continue

        # A piece too short to read becomes a second row of the same cue
        # instead of its own flash on screen: the text still fits the width,
        # and nothing outlives its speech.
        groups = max(1, min(n, int(cue.duration / min_piece_s))) if min_piece_s > 0 else n
        bounds = [round(k * n / groups) for k in range(groups + 1)]
        parts_en = ["\n".join(p.strip() for p in parts_en[a:b]) for a, b in zip(bounds, bounds[1:])]
        parts_zh = ["\n".join(p.strip() for p in parts_zh[a:b]) for a, b in zip(bounds, bounds[1:])]
        n = groups
        if n == 1:
            out.append(_replace_text(cue, parts_en[0], parts_zh[0]))
            continue

        weights = [max(1, len(p)) for p in parts_zh] if any(parts_zh) else [1] * n
        total = sum(weights)
        span = cue.duration
        cursor = cue.start
        for i in range(n):
            share = span * weights[i] / total
            end = cue.end if i == n - 1 else cursor + share
            out.append(
                Cue(
                    start=round(cursor, 3),
                    end=round(end, 3),
                    source=parts_en[i].strip(),
                    target=parts_zh[i].strip(),
                    speaker=cue.speaker,
                    translated=cue.translated,
                )
            )
            cursor = end
    return out


def _replace_text(cue: Cue, source: str, target: str) -> Cue:
    return Cue(
        start=cue.start,
        end=cue.end,
        source=source,
        target=target,
        speaker=cue.speaker,
        translated=cue.translated,
    )


#: Latin words, numbers and version-like tokens must survive intact too, and
#: so must katakana words: カー / ドゲーム reads as a mistake.
_ATOMIC = re.compile(r"[A-Za-z][A-Za-z'’\-]*|\d+(?:[.,:]\d+)*%?|[ァ-ヺー]{2,}")


#: A number and the unit after it: 25 / 歲 reads as two thoughts.
_NUMBER_UNIT = re.compile(r"\d[\d.,:%]*\s?[一-鿿]")


def _chinese_words(text: str) -> list[tuple[int, int]]:
    """Spans of multi-character Chinese words, from jieba.

    Without a word list the splitter cuts wherever the balance point falls:
    他 / 們, 專業田 / 徑選手. jieba's dictionary is Simplified, so the text is
    segmented through OpenCC's t2s (one character for one) and the spans mapped
    back; on Traditional text directly it glues words (進田徑, 比賽並).
    """
    try:
        import warnings

        # jieba 0.42.1 (2020) is unmaintained: on Python 3.12 its regexes raise
        # SyntaxWarning, and it imports the deprecated pkg_resources (falling
        # back to a plain open() once setuptools drops it). Both are harmless.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba
        from opencc import OpenCC
    except ImportError:  # pragma: no cover - both are declared dependencies
        return []
    global _T2S
    if _T2S is None:
        import logging

        jieba.setLogLevel(logging.WARNING)
        _T2S = OpenCC("t2s")
    simplified = _T2S.convert(text)
    if len(simplified) != len(text):
        simplified = text
    spans, i = [], 0
    for word in jieba.cut(simplified):
        if len(word) > 1:
            spans.append((i, i + len(word)))
        i += len(word)
    return spans


_T2S = None


def _forbidden_cuts(text: str, protect: Sequence[str], words: bool = False) -> set[int]:
    """Indices ``i`` where cutting after ``text[i]`` would sever a known word.

    ``words`` adds Chinese word boundaries; only for the translation, since
    jieba would cut Japanese at nonsense places."""
    spans: list[tuple[int, int]] = [m.span() for m in _ATOMIC.finditer(text)]
    if words:
        spans += _chinese_words(text)
        spans += [m.span() for m in _NUMBER_UNIT.finditer(text)]
    for term in protect:
        if len(term) < 2:
            continue
        start = text.find(term)
        while start != -1:
            spans.append((start, start + len(term)))
            start = text.find(term, start + 1)

    forbidden: set[int] = set()
    for start, end in spans:
        forbidden.update(range(start, end - 1))
    return forbidden


def _split_cjk(
    text: str,
    limit: int,
    protect: Sequence[str] = (),
    *,
    pieces: int = 1,
    words: bool = False,
) -> list[str]:
    """Split Chinese text into balanced pieces of at most ``limit`` characters.

    Balance matters as much as the ceiling. Filling each piece to the limit and
    letting the remainder fall out leaves orphans: a 23-character line under a
    20-character limit becomes 20 + 3, and those three characters get their own
    half-second cue. Splitting the same line as 12 + 11 reads far better and
    costs nothing.

    Punctuation is still preferred as a cut point, but the one nearest the
    balanced boundary wins rather than the last one that fits — otherwise the
    punctuation search reintroduces the imbalance it was meant to avoid.

    ``pieces`` asks for at least that many parts (the source line is long);
    ``words`` keeps Chinese words whole.
    """
    text = text.strip()
    if not text:
        return []
    # Never ask for pieces shorter than a few characters each.
    pieces = max(1, min(pieces, len(text) // 4))
    if len(text) <= limit and pieces <= 1:
        return [text]

    parts: list[str] = []
    rest = text

    # Recompute the balance point from what is *left* each time. Deciding the
    # number of pieces once up front breaks as soon as a punctuation cut lands
    # shorter than the target: the arithmetic still expects the original number
    # of pieces, and the final remainder overflows the limit.
    while len(rest) > limit or len(parts) + 1 < pieces:
        left = max(-(-len(rest) // limit), pieces - len(parts))  # ceil
        target = -(-len(rest) // left)
        cut = _best_cut(rest, target, limit, _forbidden_cuts(rest, protect, words))
        parts.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].lstrip()
        if not rest:
            break

    if rest:
        parts.append(rest)
    return [p for p in parts if p]


def _best_cut(text: str, target: int, limit: int, forbidden: set[int] = frozenset()) -> int:
    """Index of the character to cut after.

    Preference order: punctuation nearest the balance point, then the nearest
    position that does not sever a known word, then the balance point itself.
    The last case only happens when a single word is longer than the whole
    line budget, where something has to give.
    """
    ceiling = min(len(text) - 1, limit) - 1
    floor = target // 2

    punctuation = [
        i
        for i, ch in enumerate(text[: ceiling + 1])
        if ch in _CJK_BREAKS and i >= floor and i not in forbidden and _is_break(text, i)
    ]
    if punctuation:
        return min(punctuation, key=lambda i: abs(i - (target - 1)))

    allowed = [i for i in range(floor, ceiling + 1) if i not in forbidden]
    if allowed:
        return min(allowed, key=lambda i: abs(i - (target - 1)))

    # Balance is a preference; not severing a word is closer to a requirement.
    # Before giving up, look outside the balanced window — an unbalanced pair of
    # lines reads oddly, but "Minecraft M" / "anhunt" reads as a mistake.
    anywhere = [i for i in range(0, ceiling + 1) if i not in forbidden]
    if anywhere:
        return min(anywhere, key=lambda i: abs(i - (target - 1)))

    return min(target, ceiling + 1) - 1


def _is_break(text: str, i: int) -> bool:
    """A space is a break only between two CJK characters (two utterances
    joined by ``_join``). Around a Latin word or number it is padding the
    translator adds — 這是一位 25 歲 — and cutting there severs the phrase."""
    if text[i] != " ":
        return True
    return 0 < i < len(text) - 1 and is_cjk_char(text[i - 1]) and is_cjk_char(text[i + 1])


def _split_words(text: str, limit: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current)) + 1 + len(word) > limit:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts or [text]


def _distribute_words(text: str, weights: list[int], limit: int | None = None) -> list[str]:
    """Cut the English into ``len(weights)`` pieces sized in proportion to the Chinese."""
    words = text.split()
    n = len(weights)
    if n <= 1 or not words:
        return [text] + [""] * (n - 1)

    parts: list[str] = []
    cursor = 0
    for i, weight in enumerate(weights):
        if i == n - 1 or cursor >= len(words):
            parts.append(" ".join(words[cursor:]))
            cursor = len(words)
            continue
        share = weight / (sum(weights[i:]) or 1)
        ideal = cursor + max(1, round((len(words) - cursor) * share))
        hi = min(len(words), max(cursor + 1, len(words) - (n - i - 1)))

        def cost(end: int) -> tuple:
            # Over the limit is worst; then, within two words of the
            # proportional point, prefer ending on punctuation (two merged
            # utterances split where one ends); then the proportional point.
            far = abs(end - ideal)
            over = limit is not None and (
                len(" ".join(words[cursor:end])) > limit
                or len(" ".join(words[end:])) > limit * (n - i - 1)
            )
            return (
                over,
                far > 2,
                not re.search(r"[.,!?;:]$", words[end - 1]),
                far,
            )

        end = min(range(cursor + 1, hi + 1), key=cost)
        parts.append(" ".join(words[cursor:end]))
        cursor = end
    return parts


def _distribute_chars(text: str, weights: list[int], protect: Sequence[str] = ()) -> list[str]:
    """Cut Japanese source text into pieces sized in proportion to the Chinese.

    There are no spaces to cut at, so each cut goes to the punctuation mark or
    space nearest the proportional point, and never inside a protected word.
    """
    n = len(weights)
    text = text.strip()
    if n <= 1 or len(text) < n:
        return [text] + [""] * (n - 1)

    forbidden = _forbidden_cuts(text, protect)
    total = sum(weights) or n
    cuts: list[int] = []
    consumed = 0
    for weight in weights[:-1]:
        consumed += weight
        ideal = round(len(text) * consumed / total)
        lo = (cuts[-1] + 1) if cuts else 1
        window = range(max(lo, ideal - 4), min(len(text) - 1, ideal + 4) + 1)
        preferred = [
            i for i in window
            if (text[i - 1] in _CJK_BREAKS or text[i - 1].isspace()) and i - 1 not in forbidden
        ]
        allowed = [i for i in window if i - 1 not in forbidden]
        pool = preferred or allowed or [min(max(lo, ideal), len(text) - 1)]
        cuts.append(min(pool, key=lambda i: abs(i - ideal)))

    bounds = [0, *cuts, len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:])]


# --------------------------------------------------------------------------
# 3. clamp
# --------------------------------------------------------------------------


def _clamp(cues: list[Cue], max_duration: float, min_duration: float, gap: float) -> list[Cue]:
    out: list[Cue] = []
    for i, cue in enumerate(cues):
        start = cue.start
        # Guard against overlapping ASR timings: a cue never starts before the
        # previous one ends.
        if out and start < out[-1].end:
            start = out[-1].end

        ceiling = (cues[i + 1].start - gap) if i + 1 < len(cues) else float("inf")
        if ceiling <= start:
            ceiling = start + _EPSILON

        end = min(cue.end, ceiling, start + max_duration)
        # Soft floor: grow into free time only, never into the next cue.
        end = max(end, min(start + min_duration, ceiling))
        end = max(end, start + _EPSILON)

        out.append(
            Cue(
                start=round(start, 3),
                end=round(end, 3),
                source=cue.source,
                target=cue.target,
                speaker=cue.speaker,
                translated=cue.translated,
            )
        )
    return [c for c in out if c.source or c.target]


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()
