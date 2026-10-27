"""Grounding terminology in wiki page titles, with the network faked out.

The fake answers are shaped like real MediaWiki responses captured from
minecraft.wiki and zh.minecraft.wiki, including the redirect-into-a-section and
disambiguation cases that must be rejected.
"""

from __future__ import annotations

from youtube_dualsub import terms
from youtube_dualsub.terms import Candidate, MemoryCache, ground

EN_API = "https://minecraft.wiki/api.php"
ZH_API = "https://zh.minecraft.wiki/api.php"
WIKIS = (("minecraft", EN_API, ZH_API),)


def page(title, zh=None, disambiguation=False):
    p = {"title": title, "pageid": 1}
    if zh:
        p["langlinks"] = [{"lang": "zh", "title": zh}]
    if disambiguation:
        p["pageprops"] = {"disambiguation": ""}
    return p


EN_PAGES = {
    "Creeper": page("Creeper", "苦力怕"),
    "Nether": page("Nether", "下界"),
    "Iron Golem": page("Iron Golem", "铁傀儡"),
    "Redstone": page("Redstone", "红石（消歧义）"),
    "Melee attack": page("Melee attack", "近战攻击"),
    "Lapis": page("Lapis", "青金石", disambiguation=True),
    "Tutorial:Speedrun": page("Tutorial:Speedrun", "Tutorial:速通"),
}
REDIRECTS = {
    "Nether (dimension)": {"from": "Nether (dimension)", "to": "Nether"},
    "Crit": {"from": "Crit", "to": "Melee attack", "tofragment": "Critical hit"},
    "Speedrun": {"from": "Speedrun", "to": "Tutorial:Speedrun"},
}
TAIWAN = {"苦力怕": "苦力怕", "下界": "地獄", "铁傀儡": "鐵魔像", "近战攻击": "近戰攻擊"}


class FakeFetcher:
    calls: list[tuple[str, dict]] = []

    def __init__(self, timeout_s):
        pass

    def __call__(self, api, **params):
        FakeFetcher.calls.append((api, params))
        titles = params["titles"].split("|")
        if api == ZH_API:
            return {"query": {"pages": [
                {"title": t, "varianttitles": {"zh-tw": TAIWAN.get(t, t)}} for t in titles
            ]}}
        redirects = [REDIRECTS[t] for t in titles if t in REDIRECTS]
        names = [REDIRECTS[t]["to"] if t in REDIRECTS else t for t in titles]
        pages = [EN_PAGES.get(n) or {"title": n, "missing": True} for n in names]
        return {"query": {"redirects": redirects, "pages": pages}}


def run(monkeypatch, candidates, **kwargs):
    FakeFetcher.calls = []
    monkeypatch.setattr(terms, "_Fetcher", FakeFetcher)
    kwargs.setdefault("language", "en")
    kwargs.setdefault("subject", "Minecraft")
    kwargs.setdefault("domain_wikis", WIKIS)
    return ground(candidates, **kwargs)


def test_the_taiwanese_title_replaces_the_models_guess(monkeypatch):
    out = run(monkeypatch, [
        Candidate("creeper", "Creeper", "jargon", "苦力機"),
        Candidate("the Nether", "Nether (dimension)", "place", "下界"),
        Candidate("iron golem", "Iron Golem", "jargon", "鐵傀儡"),
    ])
    assert out == {"creeper": "苦力怕", "the Nether": "地獄", "iron golem": "鐵魔像"}


def test_pages_that_are_not_the_term_itself_are_rejected(monkeypatch):
    """Section redirects, disambiguation pages and other namespaces: keep the guess."""
    out = run(monkeypatch, [
        Candidate("crit", "Crit", "jargon", "爆擊"),
        Candidate("lapis", "Lapis", "jargon", "青金石"),
        Candidate("redstone", "Redstone", "jargon", "紅石"),
        Candidate("speedrun", "Speedrun", "jargon", "速通"),
    ])
    assert out == {"crit": "爆擊", "lapis": "青金石", "redstone": "紅石", "speedrun": "速通"}


def test_a_term_no_wiki_knows_keeps_the_models_rendering(monkeypatch):
    out = run(monkeypatch, [Candidate("gapple", "Enchanted Gapple", "jargon", "金蘋果")])
    assert out == {"gapple": "金蘋果"}


def test_the_domain_wiki_is_only_used_for_its_own_subject(monkeypatch):
    run(monkeypatch, [Candidate("creeper", "Creeper", "jargon", "苦力怕")], subject="Cooking")
    assert not any(api == EN_API for api, _ in FakeFetcher.calls)


def test_ordinary_nouns_are_never_sent_to_wikipedia(monkeypatch):
    """Wikipedia's "Chest" is anatomy."""
    run(monkeypatch, [Candidate("chest", "Chest", "jargon", "儲物箱")], subject="Cooking")
    assert FakeFetcher.calls == []


def test_results_are_cached_including_misses(monkeypatch):
    cache = MemoryCache()
    candidates = [Candidate("creeper", "Creeper", "jargon", ""), Candidate("x", "Nope", "jargon", "x")]
    run(monkeypatch, candidates, cache=cache)
    first = len(FakeFetcher.calls)

    out = run(monkeypatch, candidates, cache=cache)

    assert first > 0 and FakeFetcher.calls == []
    assert out["creeper"] == "苦力怕"


def test_being_offline_falls_back_to_the_models_renderings(monkeypatch):
    class Offline(FakeFetcher):
        def __call__(self, api, **params):
            raise terms._Offline("no network")

    monkeypatch.setattr(terms, "_Fetcher", Offline)
    out = ground(
        [Candidate("creeper", "Creeper", "jargon", "苦力怕")],
        language="en", subject="Minecraft", domain_wikis=WIKIS,
    )
    assert out == {"creeper": "苦力怕"}


def test_disambiguators_are_stripped_from_titles():
    assert terms._clean_title("橘雛乃 (虛擬YouTuber)") == "橘雛乃"
    assert terms._clean_title("鐵魔像") == "鐵魔像"


def test_a_redirect_to_a_differently_named_page_is_rejected(monkeypatch):
    """ja.wikipedia sends 猫汰つな to her agency's page; she is not the agency."""
    EN_PAGES["Agency"] = page("Agency", "經紀公司")
    REDIRECTS["Tsuna"] = {"from": "Tsuna", "to": "Agency"}
    try:
        out = run(monkeypatch, [Candidate("ツナ", "Tsuna", "person", "猫汰つな")])
    finally:
        del EN_PAGES["Agency"], REDIRECTS["Tsuna"]
    assert out == {"ツナ": "猫汰つな"}


def test_the_spoken_form_is_never_looked_up_in_place_of_the_canonical_name(monkeypatch):
    """ツナ is a nickname — and also the Japanese for tuna."""
    run(monkeypatch, [Candidate("Creeper", "Creeper Mob", "jargon", "苦力怕")])
    asked = {t for _, params in FakeFetcher.calls for t in params["titles"].split("|")}
    assert "Creeper" not in asked


def test_a_name_spoken_in_latin_letters_is_kept_as_spoken(monkeypatch):
    """The model rendered the YouTuber Dream as 夢; a wiki renamed George."""
    out = run(monkeypatch, [
        Candidate("Dream", "Dream (YouTuber)", "person", "夢"),
        Candidate("George", "GeorgeNotFound", "person", "喬治"),
        Candidate("ツナ", "猫汰つな", "person", "猫汰つな"),
    ])
    assert out["Dream"] == "Dream"
    assert out["George"] == "George"
    assert out["ツナ"] == "猫汰つな", "a Japanese name still takes its rendering"
