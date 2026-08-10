"""Checkpointing — specifically, that the expensive and cheap halves of the
work invalidate independently."""

from __future__ import annotations

import pytest

from youtube_dualsub.models import Sentence, Stage, Translation, TranslationStatus, VideoContext
from youtube_dualsub.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite3")


def sentences() -> list[Sentence]:
    return [
        Sentence(index=0, start=0.0, end=1.0, text="hello"),
        Sentence(index=1, start=1.0, end=2.0, text="world"),
    ]


def test_sentences_round_trip(store):
    store.save_sentences("vid", "fp-a", sentences())
    loaded = store.load_sentences("vid", "fp-a")
    assert [s.text for s in loaded] == ["hello", "world"]
    assert loaded[1].start == 1.0


def test_a_different_asr_fingerprint_is_a_different_cache_entry(store):
    store.save_sentences("vid", "fp-a", sentences())
    assert store.load_sentences("vid", "fp-b") == []


def test_swapping_the_llm_does_not_invalidate_the_transcript(store):
    """The whole point of the split: re-running translation is 10 minutes,
    re-running Whisper is 3 more on top."""
    store.save_sentences("vid", "asr-1", sentences())
    store.save_translations("vid", "asr-1", "gemma", [Translation(index=0, text="你好")])

    assert store.load_translations("vid", "asr-1", "qwen") == []
    assert len(store.load_sentences("vid", "asr-1")) == 2


def test_translations_are_upserted_per_batch(store):
    store.ensure_job("vid")
    store.save_translations("vid", "asr-1", "fp", [Translation(index=0, text="first")])
    store.save_translations("vid", "asr-1", "fp", [Translation(index=1, text="second")])
    store.save_translations("vid", "asr-1", "fp", [Translation(index=0, text="corrected")])

    loaded = store.load_translations("vid", "asr-1", "fp")
    assert [t.text for t in loaded] == ["corrected", "second"]


def test_fallback_status_survives_the_round_trip(store):
    store.save_translations(
        "vid", "a", "b",
        [Translation(index=0, text="raw english", status=TranslationStatus.SOURCE_FALLBACK)],
    )
    assert store.load_translations("vid", "a", "b")[0].status is TranslationStatus.SOURCE_FALLBACK


def test_clear_translations_keeps_the_transcript(store):
    store.save_sentences("vid", "asr-1", sentences())
    store.save_translations("vid", "asr-1", "fp", [Translation(index=0, text="x")])

    store.clear_translations("vid")

    assert store.load_translations("vid", "asr-1", "fp") == []
    assert len(store.load_sentences("vid", "asr-1")) == 2


def test_context_is_keyed_and_round_trips(store):
    store.ensure_job("vid")
    context = VideoContext(summary="A manhunt.", terms={"gapple": "金蘋果"})
    store.save_context("vid", "key-1", context)

    assert store.load_context("vid", "key-2") is None
    loaded = store.load_context("vid", "key-1")
    assert loaded.summary == "A manhunt."
    assert loaded.terms == {"gapple": "金蘋果"}


def test_job_stage_is_stored_as_its_value(store):
    store.ensure_job("vid", stage=Stage.TRANSLATE, message="working")
    assert store.get_job("vid")["stage"] == "translate"


def test_unknown_job_fields_are_rejected(store):
    store.ensure_job("vid")
    with pytest.raises(ValueError, match="unknown job field"):
        store.update_job("vid", nonsense=1)
