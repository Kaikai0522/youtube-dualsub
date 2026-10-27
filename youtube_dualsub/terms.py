"""Grounding auto-extracted terminology in what Chinese speakers actually call things.

A 12B model knows *a* Chinese name for most things and the Taiwanese one for
far fewer. Asked for Minecraft's vocabulary it answered 下界 (mainland) for
Nether, 苦力機 for creeper and 豬頭人 for hoglin: 55 of 99 right. The names it
gets wrong are exactly the ones a viewer notices.

The fix is to stop asking the model to *remember* and let it only *recognise*.
The model lists the names and jargon in the transcript together with the
canonical name to look up; this module then asks MediaWiki. An English (or
Japanese) wiki page links to its Chinese counterpart, and Chinese MediaWiki
sites convert titles between variants, so ``varianttitles["zh-tw"]`` is the
name a Taiwanese reader of that wiki sees. On the same 99 terms the lookup got
76 right in 1.8 seconds, and most of its disagreements were cases where the
hand-made glossary itself was wrong (鐵魔像, 凋零怪, 螢光石 are the zh-TW
client's names).

Two kinds of wiki are consulted:

* **Wikipedia**, for people, places, groups and titles. Never for ordinary
  nouns: "chest" in Wikipedia is anatomy.
* **A domain wiki**, when the video's subject matches one configured in
  ``context.domain_wikis``. Game wikis know every item, mob and mechanic.

Lookups are rejected when they land on a disambiguation page, a page in another
namespace (教學:速通), a redirect into a *section* of a broader page ("crit" ->
"Melee attack#Critical hit"), or a redirect to a page with a different name
altogether (猫汰つな -> her agency, ぶいすぽっ!) — each of those means the wiki
has no page for this term, and its title would be a wrong answer. Only the
model's canonical name is looked up, never the spoken form: a nickname like
ツナ is also a fish.

Everything is cached in SQLite, including misses, so a second video on the
same subject costs no network at all. When the network is down the model's own
guesses are used unchanged: terminology is an enhancement, never a dependency.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol

from .config import TERMS_VERSION
from .lang import is_cjk


log = logging.getLogger(__name__)

_USER_AGENT = "youtube-dualsub/0.2 (local subtitle tool; terminology lookup)"
_BATCH = 50

#: Term kinds worth asking Wikipedia about. Items, products and jargon are
#: not: their Wikipedia namesakes are usually a different sense of the word
#: (a game's タバコ item became 菸草, the plant).
WIKIPEDIA_KINDS = frozenset({"person", "group", "place", "work", "organization"})

_WIKIPEDIA = {
    "en": "https://en.wikipedia.org/w/api.php",
    "ja": "https://ja.wikipedia.org/w/api.php",
}
_ZH_WIKIPEDIA = "https://zh.wikipedia.org/w/api.php"

_PARENTHETICAL = re.compile(r"\s*[（(][^）)]*[）)]\s*$")


@dataclass(slots=True)
class Candidate:
    """One term the model found in the transcript."""

    source: str          # as it appears in the transcript
    lookup: str          # canonical name to search for
    kind: str = "other"
    guess: str = ""      # the model's own zh-TW rendering


class LookupCache(Protocol):
    def get_lookups(self, api: str, titles: Iterable[str]) -> dict[str, str | None]: ...
    def save_lookups(self, api: str, results: dict[str, str | None]) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str | None] = {}

    def get_lookups(self, api, titles):
        return {t: self._data[(api, t)] for t in titles if (api, t) in self._data}

    def save_lookups(self, api, results):
        for title, value in results.items():
            self._data[(api, title)] = value


def ground(
    candidates: list[Candidate],
    *,
    language: str | None,
    subject: str,
    domain_wikis: Iterable[tuple[str, str, str]] = (),
    cache: LookupCache | None = None,
    timeout_s: float = 8.0,
) -> dict[str, str]:
    """Return ``source -> zh-TW`` for every candidate, grounded where possible."""
    cache = cache or MemoryCache()
    fetch = _Fetcher(timeout_s)
    found: dict[str, str] = {}

    # A name spoken in Latin letters stays exactly as spoken. The model turned
    # the YouTuber Dream into 夢, and a wiki turned "George" into the page
    # title GeorgeNotFound — neither is what anyone said.
    verbatim = {c.source for c in candidates if c.kind == "person" and not is_cjk(c.source)}
    pending = [c for c in candidates if c.source not in verbatim]

    wikis: list[tuple[str, str, list[Candidate]]] = []
    subject_l = subject.lower()
    for keyword, api, zh_api in domain_wikis:
        if keyword.lower() in subject_l:
            wikis.append((api, zh_api, pending))
    base = (language or "en").split("-")[0]
    named = [c for c in pending if c.kind in WIKIPEDIA_KINDS]
    if base in _WIKIPEDIA:
        wikis.append((_WIKIPEDIA[base], _ZH_WIKIPEDIA, named))
    if base != "en":
        wikis.append((_WIKIPEDIA["en"], _ZH_WIKIPEDIA, named))

    for api, zh_api, pool in wikis:
        todo = [c for c in pool if c.source not in found]
        if not todo:
            continue
        titles = sorted({c.lookup or c.source for c in todo})
        try:
            resolved = _resolve(api, zh_api, titles, cache, fetch)
        except _Offline as exc:
            log.info("terminology lookup skipped (%s); using the model's renderings", exc)
            break
        for c in todo:
            hit = resolved.get(c.lookup or c.source)
            if hit:
                found[c.source] = hit

    out: dict[str, str] = {}
    for c in candidates:
        target = c.source if c.source in verbatim else found.get(c.source) or c.guess
        if target:
            out[c.source] = target
    if found:
        log.info("grounded %d/%d terms in a wiki", len(found), len(candidates))
    return out


# --------------------------------------------------------------------------


class _Offline(RuntimeError):
    pass


class _Fetcher:
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    def __call__(self, api: str, **params: str) -> dict:
        params = {**params, "format": "json", "formatversion": "2"}
        url = f"{api}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - any failure means "no lookup"
            raise _Offline(f"{urllib.parse.urlparse(api).netloc}: {exc}") from exc


def _resolve(
    api: str, zh_api: str, titles: list[str], cache: LookupCache, fetch: _Fetcher
) -> dict[str, str]:
    key = f"{api}#r{TERMS_VERSION}"
    known = cache.get_lookups(key, titles)
    missing = [t for t in titles if t not in known]

    fresh: dict[str, str | None] = {t: None for t in missing}
    chinese: dict[str, str] = {}
    for i in range(0, len(missing), _BATCH):
        chunk = missing[i : i + _BATCH]
        data = fetch(
            api,
            action="query",
            titles="|".join(chunk),
            redirects="1",
            prop="langlinks|pageprops",
            lllang="zh",
            lllimit="max",
            ppprop="disambiguation",
        )
        chinese.update(_chinese_titles(data, chunk))

    if chinese:
        variants = _taiwan_variants(zh_api, sorted(set(chinese.values())), fetch)
        for title, zh in chinese.items():
            fresh[title] = _clean_title(variants.get(zh, zh))

    cache.save_lookups(key, fresh)
    merged = {**known, **fresh}
    return {t: v for t, v in merged.items() if v}


def _chinese_titles(data: dict, asked: list[str]) -> dict[str, str]:
    query = data.get("query") or {}
    normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
    redirects = {r["from"]: r for r in query.get("redirects", [])}
    pages = {p.get("title"): p for p in query.get("pages", [])}

    out: dict[str, str] = {}
    for title in asked:
        name = normalized.get(title, title)
        redirect = redirects.get(name)
        if redirect:
            if redirect.get("tofragment"):
                continue  # "crit" -> "Melee attack#Critical hit": not its own thing
            if not _same_thing(name, redirect["to"]):
                continue  # 猫汰つな -> "ぶいすぽっ! Virtual eSports Project"
            name = redirect["to"]
        page = pages.get(name)
        if not page or page.get("missing") or ":" in name:
            continue
        if "disambiguation" in (page.get("pageprops") or {}):
            continue
        links = page.get("langlinks") or []
        if not links:
            continue
        zh = links[0].get("title") or ""
        if zh and ":" not in zh and "消歧義" not in zh and "消歧义" not in zh:
            out[title] = zh
    return out


def _taiwan_variants(zh_api: str, titles: list[str], fetch: _Fetcher) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(titles), _BATCH):
        chunk = titles[i : i + _BATCH]
        try:
            data = fetch(
                zh_api,
                action="query",
                titles="|".join(chunk),
                prop="info",
                inprop="varianttitles",
            )
        except _Offline as exc:
            log.debug("variant lookup failed: %s", exc)
            continue
        for page in (data.get("query") or {}).get("pages", []):
            variants = page.get("varianttitles") or {}
            tw = variants.get("zh-tw") or variants.get("zh-hant")
            if tw:
                out[page["title"]] = tw
    return out


_NON_WORD = re.compile(r"[\W_]+")


def _same_thing(asked: str, target: str) -> bool:
    """A redirect that keeps the name (Nether (dimension) -> Nether) rather
    than replacing it with something broader."""
    a, b = _NON_WORD.sub("", asked.lower()), _NON_WORD.sub("", target.lower())
    return bool(a and b) and (a in b or b in a)


def _clean_title(title: str) -> str:
    """鐵魔像 stays; 橘雛乃 (虛擬YouTuber) loses its disambiguator."""
    return _PARENTHETICAL.sub("", title).strip()
