"""Terminology control: the layer that stops 苦力怕 becoming 爬行者."""

from __future__ import annotations

import pytest

from youtube_dualsub import glossary as glossary_mod
from youtube_dualsub.config import Settings, _merge
from youtube_dualsub.glossary import apply_glossary, load_glossary


@pytest.fixture
def curated(tmp_path, monkeypatch):
    (tmp_path / "domain.yaml").write_text(
        """
terms:
  - source: creeper
    target: 苦力怕
    aliases: [爬行者]
  - source: nether
    target: 地獄
    aliases: [下界]
  - source: netherite
    target: 獄髓
    aliases: [下界合金]
""",
        encoding="utf-8",
    )
    (tmp_path / "user.yaml").write_text(
        "terms:\n  - source: creeper\n    target: 綠色炸彈客\n", encoding="utf-8"
    )
    monkeypatch.setattr(glossary_mod, "GLOSSARY_DIR", tmp_path)
    return tmp_path


def settings_with(*files: str) -> Settings:
    return _merge(Settings(), {"postprocess": {"glossary_files": list(files)}})


def test_aliases_are_rewritten_to_the_canonical_term(curated):
    g = load_glossary(settings_with("domain.yaml"))
    assert apply_glossary("那隻爬行者炸了我的房子", g) == "那隻苦力怕炸了我的房子"


def test_untranslated_english_is_rewritten_too(curated):
    g = load_glossary(settings_with("domain.yaml"))
    assert apply_glossary("他被 creeper 炸死了", g) == "他被 苦力怕 炸死了"


def test_longer_terms_win_over_their_prefixes(curated):
    """下界合金 must become 獄髓, not 地獄合金."""
    g = load_glossary(settings_with("domain.yaml"))
    assert apply_glossary("他拿到下界合金劍了", g) == "他拿到獄髓劍了"


def test_user_file_overrides_the_curated_one(curated):
    g = load_glossary(settings_with("domain.yaml", "user.yaml"))
    assert g.prompt_terms["creeper"] == "綠色炸彈客"
    assert apply_glossary("creeper", g) == "綠色炸彈客"


def test_curated_terms_outrank_the_models_guesses(curated):
    """Decision Q28: auto-extracted terms are the weakest layer, by design."""
    g = load_glossary(settings_with("domain.yaml"), auto_terms={"creeper": "爬行者"})
    assert g.prompt_terms["creeper"] == "苦力怕"


def test_auto_terms_are_used_when_nothing_curated_covers_them(curated):
    g = load_glossary(settings_with("domain.yaml"), auto_terms={"Sapnap": "薩普納普"})
    assert g.prompt_terms["Sapnap"] == "薩普納普"


def test_a_missing_glossary_file_is_not_an_error(curated):
    assert load_glossary(settings_with("nope.yaml")).prompt_terms == {}


def test_the_shipped_minecraft_glossary_loads():
    g = load_glossary(Settings())
    assert g.prompt_terms["creeper"] == "苦力怕"
    assert apply_glossary("末影珍珠", g) == "終界珍珠"
