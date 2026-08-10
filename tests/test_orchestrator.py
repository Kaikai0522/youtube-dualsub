"""End-to-end pipeline behaviour with the GPU and the network faked out.

Covers the properties that only appear once the stages are chained: caching,
resumption, streaming, and the manual-caption shortcut.
"""

from __future__ import annotations

import pytest

from youtube_dualsub.config import Settings, _merge
from youtube_dualsub.models import Cue, Fragment, Stage, Word
from youtube_dualsub.pipeline import orchestrator as orch
from youtube_dualsub.sources.ytdlp_source import AudioAsset, VideoInfo
from youtube_dualsub.store import Store

VIDEO = "IqcS1d3eXYc"


def info() -> VideoInfo:
    return VideoInfo(
        video_id=VIDEO, title="Minecraft Speedrunner Swap VS 3 Hunters",
        uploader="Dream", duration_s=3123.0, has_manual_subs=False, manual_sub_langs=(),
    )


def fragments() -> list[Fragment]:
    spec = [
        (0.0, 1.0, " Wait he's got gapples."),
        (1.2, 2.4, " We're cooked."),
        (3.0, 4.6, " Go for the portal!"),
    ]
    return [
        Fragment(s, e, t, [Word(start=s, end=e, text=t)]) for s, e, t in spec
    ]


class FakeClient:
    """Translates by numbering lines, so alignment is always satisfiable."""

    instances: list["FakeClient"] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.calls = 0
        self.unloaded = False
        FakeClient.instances.append(self)

    def ensure_available(self):
        return None

    def complete_json(self, prompt, *, system=None, max_tokens=None):
        self.calls += 1
        count = sum(1 for line in prompt.split("Lines:")[-1].splitlines() if line.strip())
        return {str(i + 1): f"譯文{i + 1}" for i in range(count)}

    def unload(self):
        self.unloaded = True


@pytest.fixture
def settings() -> Settings:
    return _merge(
        Settings(),
        {
            "vocals": {"enabled": False},
            "context": {"enabled": False},
            "translate": {"batch_size": 2, "min_batch_size": 2},
        },
    )


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite3")


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    calls = {"audio": 0, "asr": 0, "captions": 0}
    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"not really audio")

    def get_audio(video_id, settings, progress=None):
        calls["audio"] += 1
        return AudioAsset(path=audio_file, info=info(), format_spec="test")

    def transcribe(path, settings, duration_s=None, progress=None):
        calls["asr"] += 1
        return fragments()

    def get_manual_captions(video_id, settings):
        calls["captions"] += 1
        return None

    monkeypatch.setattr(orch.audio_stage, "probe", lambda video_id, settings: info())
    monkeypatch.setattr(orch.audio_stage, "get_audio", get_audio)
    monkeypatch.setattr(orch.audio_stage, "get_manual_captions", get_manual_captions)
    monkeypatch.setattr(orch.asr_stage, "transcribe", transcribe)
    monkeypatch.setattr(orch, "OllamaClient", FakeClient)
    FakeClient.instances = []
    return calls


def test_a_full_run_produces_bilingual_cues(settings, store, fakes):
    result = orch.Orchestrator(settings, store).run(VIDEO)

    assert result.title == "Minecraft Speedrunner Swap VS 3 Hunters"
    assert result.cues
    assert all(isinstance(c, Cue) for c in result.cues)
    assert all(c.source for c in result.cues)
    assert all(c.target for c in result.cues)
    assert store.get_job(VIDEO)["stage"] == Stage.DONE.value


def test_batches_stream_out_before_the_job_finishes(settings, store, fakes):
    """Decision Q16: the first batch reaches the viewer while later ones run."""
    emissions: list[int] = []

    orch.Orchestrator(settings, store).run(
        VIDEO, on_cues=lambda cues, lo, hi: emissions.append(len(cues))
    )

    assert len(emissions) >= 2, "cues should arrive incrementally, not once at the end"


def test_a_second_run_reuses_the_transcript_and_the_translation(settings, store, fakes):
    orch.Orchestrator(settings, store).run(VIDEO)
    first_calls = FakeClient.instances[0].calls

    orch.Orchestrator(settings, store).run(VIDEO)

    assert fakes["asr"] == 1, "Whisper must not run again"
    assert fakes["audio"] == 1, "the audio must not be downloaded again"
    assert sum(c.calls for c in FakeClient.instances) == first_calls


def test_retranslating_keeps_the_transcript(settings, store, fakes):
    """Swapping models costs 10 minutes, not 13."""
    orch.Orchestrator(settings, store).run(VIDEO)

    orch.Orchestrator(settings, store).run(VIDEO, force_retranslate=True)

    assert fakes["asr"] == 1
    assert sum(c.calls for c in FakeClient.instances) > FakeClient.instances[0].calls


def test_changing_the_llm_invalidates_only_the_translation(settings, store, fakes):
    orch.Orchestrator(settings, store).run(VIDEO)

    other = _merge(settings, {"translate": {"model": "qwen3:14b"}})
    result = orch.Orchestrator(other, store).run(VIDEO)

    assert fakes["asr"] == 1
    assert result.cues


def test_human_captions_skip_audio_and_whisper_entirely(settings, store, monkeypatch, fakes):
    """Decision Q2: a human transcript beats ours, is aligned, and is free."""
    monkeypatch.setattr(
        orch.audio_stage,
        "get_manual_captions",
        lambda video_id, s: [(0.0, 1.5, "Wait he's got gapples."), (1.6, 3.0, "We're cooked.")],
    )

    result = orch.Orchestrator(settings, store).run(VIDEO)

    assert result.used_manual_captions is True
    assert fakes["asr"] == 0
    assert fakes["audio"] == 0
    assert len(result.cues) >= 1


def test_cancelling_pauses_and_keeps_what_was_finished(settings, store, fakes):
    seen = {"batches": 0}

    def cancel_after_first_batch() -> bool:
        return seen["batches"] >= 1

    def count(cues, lo, hi):
        seen["batches"] += 1

    with pytest.raises(orch.Cancelled):
        orch.Orchestrator(settings, store).run(
            VIDEO, on_cues=count, should_cancel=cancel_after_first_batch
        )

    assert store.get_job(VIDEO)["stage"] == Stage.PAUSED.value
    assert store.load_sentences(VIDEO, settings.asr_fingerprint), "transcript must survive"
    assert store.load_translations(
        VIDEO, settings.asr_fingerprint, settings.translation_fingerprint
    ), "finished batches must survive"


def test_resuming_only_translates_what_is_missing(settings, store, fakes):
    seen = {"batches": 0}

    def count(cues, lo, hi):
        seen["batches"] += 1

    with pytest.raises(orch.Cancelled):
        orch.Orchestrator(settings, store).run(
            VIDEO, on_cues=count, should_cancel=lambda: seen["batches"] >= 1
        )

    done_before = len(
        store.load_translations(VIDEO, settings.asr_fingerprint, settings.translation_fingerprint)
    )
    result = orch.Orchestrator(settings, store).run(VIDEO)

    assert 0 < done_before < len(result.translations)
    assert fakes["asr"] == 1


def test_an_unreachable_llm_degrades_to_english_only(settings, store, fakes, monkeypatch):
    """Decision Q19: English subtitles still beat none."""
    from youtube_dualsub.llm import LlmError

    class Unreachable(FakeClient):
        def ensure_available(self):
            raise LlmError("Ollama is not running")

    monkeypatch.setattr(orch, "OllamaClient", Unreachable)

    result = orch.Orchestrator(settings, store).run(VIDEO)

    assert result.cues
    assert all(c.source for c in result.cues)
    assert all(c.target == "" for c in result.cues)
