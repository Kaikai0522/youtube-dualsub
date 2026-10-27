"""Terminology control, in two layers of decreasing authority.

    <file>.yaml          optional, hand-written overrides (user.yaml); always win
    auto-extracted       found in this video, grounded in wiki titles (terms.py)

Nothing here has to be written by hand any more. The auto layer used to be
the model's own guesses, which is why a hand-made Minecraft glossary once
shipped: asked for "creeper", a general-purpose model confidently produced
爬行者, a name no Taiwanese player uses. The wiki grounding in ``terms.py``
fixed that without a list, for any subject. ``user.yaml`` remains for the rare
name you want rendered your own way.

Auto-extracted terms are scoped to a single video and never persisted across
videos (decision Q28): a term guessed wrong once should not quietly poison
every future translation. (Wiki *lookups* are cached across videos; they are
facts about the wiki, not guesses about a video.)


Each entry may list ``aliases`` — renderings the model is *known* to reach for.
Those get rewritten to the canonical term after translation, which catches the
case where the glossary in the prompt was simply ignored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import GLOSSARY_DIR, Settings

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Term:
    source: str
    target: str
    aliases: tuple[str, ...] = ()


@dataclass(slots=True)
class Glossary:
    #: source -> target, for injection into the translation prompt.
    prompt_terms: dict[str, str] = field(default_factory=dict)
    #: (compiled pattern, replacement) applied to finished translations.
    substitutions: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.prompt_terms or self.substitutions)


def load_glossary(settings: Settings, auto_terms: dict[str, str] | None = None) -> Glossary:
    """Build the merged glossary. Later files override earlier ones."""
    terms: dict[str, Term] = {}

    if auto_terms and settings.postprocess.use_auto_terms:
        for source, target in auto_terms.items():
            terms[source.lower()] = Term(source=source, target=target)

    for filename in settings.postprocess.glossary_files:
        for term in _read_file(GLOSSARY_DIR / filename):
            terms[term.source.lower()] = term

    prompt_terms = {t.source: t.target for t in terms.values()}
    substitutions = _compile_substitutions(terms.values())
    return Glossary(prompt_terms=prompt_terms, substitutions=substitutions)


def _read_file(path: Path) -> list[Term]:
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text("utf-8")) or {}
    except yaml.YAMLError as exc:
        log.error("glossary %s is not valid YAML, ignoring it: %s", path.name, exc)
        return []

    out: list[Term] = []
    for raw in data.get("terms") or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if not source or not target:
            continue
        aliases = tuple(
            str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip()
        )
        out.append(Term(source=source, target=target, aliases=aliases))
    return out


def _compile_substitutions(terms) -> list[tuple[re.Pattern[str], str]]:
    subs: list[tuple[re.Pattern[str], str]] = []
    for term in terms:
        # Longest first so "netherite" is not eaten by "nether".
        for variant in sorted({term.source, *term.aliases}, key=len, reverse=True):
            if variant == term.target:
                continue
            subs.append((_pattern(variant), term.target))
    # And again globally, so a long CJK alias wins over a short one.
    subs.sort(key=lambda pair: len(pair[0].pattern), reverse=True)
    return subs


_LATIN = re.compile(r"^[\w\s'\-]+$", re.ASCII)


def _pattern(variant: str) -> re.Pattern[str]:
    """Word boundaries for Latin text; bare substring for CJK, which has none."""
    if _LATIN.match(variant):
        return re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE)
    return re.compile(re.escape(variant))


def apply_glossary(text: str, glossary: Glossary) -> str:
    for pattern, replacement in glossary.substitutions:
        text = pattern.sub(replacement, text)
    return text
