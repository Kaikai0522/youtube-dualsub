"""Deterministic clean-up applied to every finished translation.

Two passes, in this order:

1. **OpenCC ``s2twp``** — Simplified to Taiwan-Traditional *including vocabulary*
   (视频 -> 影片, 软件 -> 軟體). Cheap, deterministic, and it does not care how
   confident the model was. It is a switch rather than a fixture so the
   acceptance run can measure whether gemma4 actually needs it (decision Q9/Q30);
   if it turns out not to, turning it off costs nothing.

2. **Glossary substitution** — rewrites known-wrong renderings to the canonical
   term. This runs *after* OpenCC so that the glossary's own Traditional
   Chinese targets are never fed through the converter.
"""

from __future__ import annotations

import functools
import logging
import re

from ..config import Settings
from ..glossary import Glossary, apply_glossary
from ..models import Translation, TranslationStatus

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _converter(config: str):
    try:
        from opencc import OpenCC
    except ImportError:  # pragma: no cover
        log.warning("opencc is not installed; skipping Simplified->Traditional conversion")
        return None
    try:
        return OpenCC(config)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load OpenCC config %r: %s", config, exc)
        return None


def clean(text: str, settings: Settings, glossary: Glossary) -> str:
    if settings.postprocess.opencc_enabled:
        converter = _converter(settings.postprocess.opencc_config)
        if converter is not None:
            text = converter.convert(text)
    if glossary:
        text = apply_glossary(text, glossary)
    return _tidy(text)


def clean_batch(
    translations: list[Translation], settings: Settings, glossary: Glossary
) -> list[Translation]:
    for t in translations:
        # A source-fallback line is English on purpose; running it through a
        # Chinese converter and glossary would only corrupt it.
        if t.status is TranslationStatus.SOURCE_FALLBACK:
            continue
        t.text = clean(t.text, settings, glossary)
    return translations


_WS = re.compile(r"[ \t]+")
_STRIPPABLE = "「」『』\"'"


def _tidy(text: str) -> str:
    text = _WS.sub(" ", text.replace("\n", " ")).strip()
    # Models like to wrap a whole subtitle line in quotes it was never given.
    if len(text) > 2 and text[0] in _STRIPPABLE and text[-1] in _STRIPPABLE:
        text = text[1:-1].strip()
    return text


SIMPLIFIED_PROBE = re.compile(
    "[这后个么们说时来对现开关点问题实发经动华语让觉记认识边过还应该"
    "网络软视频质学习尽体验专业务员导长图书馆车间队伍强调节约"
    "]"
)


def simplified_ratio(texts: list[str]) -> float:
    """Fraction of lines containing at least one obviously Simplified character.

    Used by the acceptance run to answer "does gemma4 actually need OpenCC?"
    with a number instead of a hunch.
    """
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if SIMPLIFIED_PROBE.search(t))
    return hits / len(texts)
