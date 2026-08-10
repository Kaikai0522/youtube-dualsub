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
            # Not translated yet (streaming) or dropped: the English still has value.
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
    cues = _split_long(cues, cfg.max_chars_zh, cfg.max_chars_en, protect)
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


def _join(a: Cue, b: Cue) -> Cue:
    return Cue(
        start=a.start,
        end=max(a.end, b.end),
        source=_join_text(a.source, b.source, cjk=False),
        target=_join_text(a.target, b.target, cjk=True),
        speaker=a.speaker or b.speaker,
        translated=a.translated and b.translated,
    )


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
            out[-1] = _join(prev, cue)
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

_CJK_BREAKS = "，。！？；：、,.!?;:"


def _split_long(
    cues: list[Cue], max_zh: int, max_en: int, protect: Sequence[str] = ()
) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        parts_zh = _split_cjk(cue.target, max_zh, protect) if cue.target else []
        n = len(parts_zh)
        if n <= 1 and len(cue.source) <= max_en:
            out.append(cue)
            continue
        if n <= 1:
            # Untranslated but very long English: split on the English alone.
            parts_en = _split_words(cue.source, max_en)
            n = len(parts_en)
            parts_zh = [""] * n
        else:
            parts_en = _distribute_words(cue.source, [len(p) for p in parts_zh])

        if n <= 1:
            out.append(cue)
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


#: Latin words, numbers and version-like tokens must survive intact too.
_ATOMIC = re.compile(r"[A-Za-z][A-Za-z'’\-]*|\d+(?:[.,:]\d+)*%?")


def _forbidden_cuts(text: str, protect: Sequence[str]) -> set[int]:
    """Indices ``i`` where cutting after ``text[i]`` would sever a known word."""
    spans: list[tuple[int, int]] = [m.span() for m in _ATOMIC.finditer(text)]
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


def _split_cjk(text: str, limit: int, protect: Sequence[str] = ()) -> list[str]:
    """Split Chinese text into balanced pieces of at most ``limit`` characters.

    Balance matters as much as the ceiling. Filling each piece to the limit and
    letting the remainder fall out leaves orphans: a 23-character line under a
    20-character limit becomes 20 + 3, and those three characters get their own
    half-second cue. Splitting the same line as 12 + 11 reads far better and
    costs nothing.

    Punctuation is still preferred as a cut point, but the one nearest the
    balanced boundary wins rather than the last one that fits — otherwise the
    punctuation search reintroduces the imbalance it was meant to avoid.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    rest = text

    # Recompute the balance point from what is *left* each time. Deciding the
    # number of pieces once up front breaks as soon as a punctuation cut lands
    # shorter than the target: the arithmetic still expects the original number
    # of pieces, and the final remainder overflows the limit.
    while len(rest) > limit:
        pieces = -(-len(rest) // limit)  # ceil
        target = -(-len(rest) // pieces)
        cut = _best_cut(rest, target, limit, _forbidden_cuts(rest, protect))
        parts.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].lstrip()

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
        if ch in _CJK_BREAKS and i >= floor and i not in forbidden
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


def _distribute_words(text: str, weights: list[int]) -> list[str]:
    """Cut the English into ``len(weights)`` pieces sized in proportion to the Chinese."""
    words = text.split()
    n = len(weights)
    if n <= 1 or not words:
        return [text] + [""] * (n - 1)

    total = sum(weights) or n
    parts: list[str] = []
    cursor = 0
    for i, weight in enumerate(weights):
        if i == n - 1:
            parts.append(" ".join(words[cursor:]))
            break
        take = max(1, round(len(words) * weight / total))
        take = min(take, len(words) - cursor - (n - i - 1))
        take = max(take, 1)
        parts.append(" ".join(words[cursor : cursor + take]))
        cursor += take
    return parts


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
