"""What invalidates which half of the cache.

The whole point of the two-fingerprint design is that the expensive stage
(transcription) and the cheap stage (translation) retire independently. These
tests pin that down, because getting it wrong is invisible: the pipeline
happily serves stale rows and looks like it ignored your fix.
"""

from __future__ import annotations

from youtube_dualsub.config import Settings, _merge


def with_(**overrides) -> Settings:
    return _merge(Settings(), overrides)


class TestAsrFingerprint:
    def test_the_whisper_model_matters(self):
        assert with_(asr={"model": "large-v3-turbo"}).asr_fingerprint != Settings().asr_fingerprint

    def test_vocal_isolation_matters(self):
        assert with_(vocals={"enabled": False}).asr_fingerprint != Settings().asr_fingerprint

    def test_the_caption_source_matters(self):
        """Human captions and Whisper produce different transcripts entirely."""
        assert (
            with_(audio={"prefer_manual_captions": False}).asr_fingerprint
            != Settings().asr_fingerprint
        )

    def test_sentence_building_thresholds_matter(self):
        """These decide what a stored 'sentence' is, so they belong in the key."""
        for override in (
            {"sentences": {"max_sentence_chars": 100}},
            {"sentences": {"pause_break_s": 0.5}},
            {"sentences": {"max_sentence_s": 8.0}},
            {"sentences": {"min_fragment_chars": 5}},
        ):
            assert with_(**override).asr_fingerprint != Settings().asr_fingerprint, override

    def test_the_llm_does_not_matter(self):
        assert (
            with_(translate={"model": "qwen3:14b"}).asr_fingerprint
            == Settings().asr_fingerprint
        )


class TestTranslationFingerprint:
    def test_the_llm_matters(self):
        assert (
            with_(translate={"model": "something-else"}).translation_fingerprint
            != Settings().translation_fingerprint
        )

    def test_the_prompt_version_matters(self):
        assert (
            with_(translate={"prompt_version": "v2"}).translation_fingerprint
            != Settings().translation_fingerprint
        )

    def test_the_style_matters(self):
        assert (
            with_(translate={"style": "literal"}).translation_fingerprint
            != Settings().translation_fingerprint
        )

    def test_whisper_settings_do_not_matter(self):
        """Swapping the LLM must not cost another transcription run."""
        assert (
            with_(asr={"model": "large-v3-turbo"}).translation_fingerprint
            == Settings().translation_fingerprint
        )
