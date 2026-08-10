"""The local HTTP surface the extension talks to."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from youtube_dualsub import main  # noqa: E402
from youtube_dualsub.store import Store  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "Store", lambda: Store(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(main, "LOCAL_CONFIG", tmp_path / "config.local.json")
    monkeypatch.setattr(main, "_installed_models", lambda: ["gemma4:12b", "qwen3:14b"])
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_settings_expose_the_three_knobs_and_the_model_list(client):
    from youtube_dualsub.config import Settings

    body = client.get("/api/settings").json()
    assert body["translate_model"] == Settings().translate.model
    assert body["opencc_enabled"] is True
    assert body["style"]["zh_on_top"] is True
    assert "qwen3:14b" in body["models"]


def test_patching_settings_persists_and_takes_effect(client, tmp_path):
    body = client.post("/api/settings", json={"opencc_enabled": False}).json()
    assert body["opencc_enabled"] is False

    written = json.loads((tmp_path / "config.local.json").read_text("utf-8"))
    assert written["postprocess"]["opencc_enabled"] is False
    assert client.get("/api/settings").json()["opencc_enabled"] is False


def test_the_summary_pass_can_be_switched_off(client, tmp_path):
    assert client.get("/api/settings").json()["context_enabled"] is True

    body = client.post("/api/settings", json={"context_enabled": False}).json()
    assert body["context_enabled"] is False

    written = json.loads((tmp_path / "config.local.json").read_text("utf-8"))
    assert written["context"]["enabled"] is False


def test_switching_the_summary_retires_translations_but_not_the_transcript(client):
    """context.enabled is part of the translation fingerprint only."""
    from youtube_dualsub.config import Settings, _merge

    on = Settings()
    off = _merge(on, {"context": {"enabled": False}})
    assert on.translation_fingerprint != off.translation_fingerprint
    assert on.asr_fingerprint == off.asr_fingerprint


def test_patching_one_field_leaves_the_others_alone(client, tmp_path):
    client.post("/api/settings", json={"translate_model": "qwen3:14b"})
    client.post("/api/settings", json={"opencc_enabled": False})

    written = json.loads((tmp_path / "config.local.json").read_text("utf-8"))
    assert written["translate"]["model"] == "qwen3:14b"
    assert written["postprocess"]["opencc_enabled"] is False


def test_a_malformed_video_id_is_rejected(client):
    assert client.post("/api/jobs", json={"video_id": "nope"}).status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/IqcS1d3eXYc").status_code == 404


def test_export_before_anything_is_transcribed_is_404(client):
    assert client.get("/api/export/IqcS1d3eXYc.srt").status_code == 404


def test_export_rejects_unknown_formats(client):
    assert client.get("/api/export/IqcS1d3eXYc.vtt").status_code == 400
