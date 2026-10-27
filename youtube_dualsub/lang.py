"""What the rest of the pipeline needs to know about writing systems.

English separates words with spaces and ends sentences with ``.!?``. Japanese
does neither: Whisper hands back sub-word tokens with no spaces between them,
and a sentence ends in 。！？ — or, in casual speech, in nothing at all. Every
stage that joins, splits or measures text asks this module instead of assuming
English, because the first version assumed English and turned そこに人間いた
into "そ こ に 人 間 いた".

Script is decided from the text itself rather than threaded through as a
language code. A Japanese video still contains "OK" and "Minecraft", and a line
knows what it is made of better than the video's metadata does.
"""

from __future__ import annotations

import re

#: Language codes the pipeline is tuned for. Anything else still runs, with
#: generic prompts and English-style segmentation for Latin scripts.
LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
}

SENTENCE_END = ".!?。！？…‼⁉"
#: May trail a sentence-ending mark without ending anything themselves: 「行こう。」
CLOSERS = "\"'”’」』）)】]〉》"
CLAUSE_BREAK = ",，、;；:："

_CJK = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\uac00-\ud7af]"
)
_LATIN_ALNUM = re.compile(r"[A-Za-z0-9]")
_LETTER = re.compile(r"\w", re.UNICODE)


def language_name(code: str | None) -> str:
    if not code:
        return "the original language"
    return LANGUAGE_NAMES.get(code.split("-")[0].lower(), code)


def base_language(code: str | None) -> str | None:
    return code.split("-")[0].lower() if code else None


def is_cjk_char(ch: str) -> bool:
    return bool(_CJK.match(ch))


def is_cjk(text: str) -> bool:
    """Whether ``text`` is written mostly in a script that uses no spaces.

    "OKよ！" counts as CJK: one kana is enough to make spacing wrong, so the
    threshold is deliberately low.
    """
    letters = _LETTER.findall(text)
    if not letters:
        return False
    cjk = sum(1 for ch in letters if is_cjk_char(ch))
    return cjk / len(letters) >= 0.3


def join(left: str, right: str) -> str:
    """Concatenate two pieces of text with a space only where the script wants one."""
    left, right = left.rstrip(), right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if is_cjk_char(left[-1]) or is_cjk_char(right[0]):
        return left + right
    # Whisper word tokens carry their own leading space; captions do not.
    return f"{left} {right}"


def join_tokens(tokens: list[str]) -> str:
    """Rebuild text from Whisper word tokens.

    English tokens arrive as " word" and concatenate as-is. Japanese tokens
    arrive as bare fragments ("そ", "こ", "に") and must also concatenate as-is.
    The only space ever *added* is between two Latin tokens that lost theirs at
    a segment boundary.
    """
    out = ""
    for token in tokens:
        if not token:
            continue
        if out and not token[0].isspace() and not out[-1].isspace():
            if _LATIN_ALNUM.match(out[-1]) and _LATIN_ALNUM.match(token[0]):
                out += " "
        out += token
    return tidy(out)


_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%)\]、。！？，」』])")


def tidy(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()


def ends_sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip(CLOSERS)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def ends_clause(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in CLAUSE_BREAK


def visible_length(text: str) -> int:
    """Characters as a reader experiences them: a kanji is as wide as two letters."""
    return sum(2 if is_cjk_char(ch) else 1 for ch in text)
