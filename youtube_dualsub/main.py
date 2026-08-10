"""Local HTTP + WebSocket API for the Chrome extension.

Binds to 127.0.0.1 only. The extension posts a video id, then holds a
WebSocket open for progress and for subtitle batches as they finish — the
viewer starts watching after the first batch rather than after the last
(decision Q16).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .config import LOCAL_CONFIG, Settings, ensure_dirs, load_settings
from .export import write_ass, write_srt
from .jobs import JobManager, initial_events, stage_label
from .models import VideoContext
from .pipeline import shape
from .pipeline.orchestrator import protected_terms
from .store import Store

log = logging.getLogger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_dirs()
    settings_ = load_settings(config_path=LOCAL_CONFIG)
    store_ = Store()
    _state["settings"] = settings_
    _state["store"] = store_
    _state["manager"] = JobManager(settings_, store_)
    log.info(
        "youtube_dualsub ready on http://%s:%d",
        settings_.server.host,
        settings_.server.port,
    )
    yield
    _state.clear()


app = FastAPI(title="youtube_dualsub", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # The content script runs on the YouTube origin; the popup on the extension's.
    allow_origin_regex=r"^(https://(www|m)\.youtube\.com|chrome-extension://.*)$",
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def allow_private_network(request, call_next):
    """Chrome's Private Network Access check.

    youtube.com is a public origin reaching into 127.0.0.1, which Chrome gates
    behind a preflight that wants this header explicitly. Without it the
    extension's very first fetch fails with an opaque CORS error.
    """
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network"):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def settings() -> Settings:
    return _state["settings"]


def store() -> Store:
    return _state["store"]


def manager() -> JobManager:
    return _state["manager"]


# ---------------------------------------------------------------- models ---


class StartRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11, pattern=r"^[A-Za-z0-9_-]{11}$")
    retranslate: bool = False


class SettingsPatch(BaseModel):
    """What the popup can change. Everything else stays in config.local.json."""

    translate_model: str | None = None
    opencc_enabled: bool | None = None
    font_size_zh: int | None = Field(default=None, ge=10, le=96)
    #: Turning the whole-video summary off is the one setting here that trades
    #: quality for speed rather than taste, so the popup warns about it.
    context_enabled: bool | None = None


# ------------------------------------------------------------------ api ---


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": app.version}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    s = settings()
    return {
        "translate_model": s.translate.model,
        "opencc_enabled": s.postprocess.opencc_enabled,
        "context_enabled": s.context.enabled,
        "vocals_enabled": s.vocals.enabled,
        "style": {
            "zh_on_top": s.style.zh_on_top,
            "font_size_zh": s.style.font_size_zh,
            "font_size_en": s.style.font_size_en,
        },
        "models": _installed_models(),
    }


@app.post("/api/settings")
def patch_settings(patch: SettingsPatch) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if LOCAL_CONFIG.is_file():
        try:
            overrides = json.loads(LOCAL_CONFIG.read_text("utf-8"))
        except json.JSONDecodeError:
            log.warning("config.local.json is invalid; replacing it")

    if patch.translate_model is not None:
        overrides.setdefault("translate", {})["model"] = patch.translate_model
    if patch.opencc_enabled is not None:
        overrides.setdefault("postprocess", {})["opencc_enabled"] = patch.opencc_enabled
    if patch.font_size_zh is not None:
        overrides.setdefault("style", {})["font_size_zh"] = patch.font_size_zh
    if patch.context_enabled is not None:
        overrides.setdefault("context", {})["enabled"] = patch.context_enabled

    LOCAL_CONFIG.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), "utf-8")
    fresh = load_settings(config_path=LOCAL_CONFIG)
    _state["settings"] = fresh
    manager().settings = fresh
    return get_settings()


@app.post("/api/jobs")
async def start_job(request: StartRequest) -> dict[str, Any]:
    handle = manager().start(request.video_id, force_retranslate=request.retranslate)
    return {"video_id": handle.video_id, "running": handle.running}


@app.get("/api/jobs/{video_id}")
def job_state(video_id: str) -> dict[str, Any]:
    row = store().get_job(video_id)
    if not row:
        raise HTTPException(404, f"no job for {video_id}")
    handle = manager().get(video_id)
    return {
        **row,
        "stage_label": stage_label(row["stage"]),
        "running": bool(handle and handle.running),
    }


@app.delete("/api/jobs/{video_id}")
def pause_job(video_id: str) -> dict[str, Any]:
    manager().pause(video_id)
    return {"video_id": video_id, "paused": True}


@app.get("/api/export/{video_id}.{fmt}", response_class=PlainTextResponse)
def export(video_id: str, fmt: str) -> PlainTextResponse:
    if fmt not in ("srt", "ass"):
        raise HTTPException(400, "format must be srt or ass")

    s = settings()
    sentences = store().load_sentences(video_id, s.asr_fingerprint)
    if not sentences:
        raise HTTPException(404, f"nothing transcribed for {video_id} yet")
    translations = store().load_translations(
        video_id, s.asr_fingerprint, s.translation_fingerprint
    )
    context = store().load_context(
        video_id, f"{s.asr_fingerprint}|{s.context.model or s.translate.model}"
    )
    cues = shape.build_cues(
        sentences, translations, s,
        protect=protected_terms(s, context or VideoContext()),
    )

    from .config import EXPORT_DIR

    path = EXPORT_DIR / f"{video_id}.{fmt}"
    writer = write_srt if fmt == "srt" else write_ass
    writer(cues, path, s)
    return PlainTextResponse(
        path.read_text("utf-8"),
        headers={"Content-Disposition": f'attachment; filename="{video_id}.{fmt}"'},
        media_type="text/plain; charset=utf-8",
    )


# ------------------------------------------------------------ websocket ---


@app.websocket("/ws/jobs/{video_id}")
async def job_socket(websocket: WebSocket, video_id: str) -> None:
    await websocket.accept()
    mgr = manager()
    handle = mgr.get(video_id) or mgr.start(video_id)

    if not handle.meta:
        row = store().get_job(video_id) or {}
        handle.meta = {
            "title": row.get("title") or "",
            "duration_s": row.get("duration_s") or 0.0,
        }

    queue = await mgr.subscribe(handle)
    try:
        for event in initial_events(handle):
            await websocket.send_json(event)
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in ("done", "error"):
                # Keep the socket open: the client decides when to leave, and
                # leaving is what pauses any follow-up work.
                continue
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket for %s failed", video_id)
    finally:
        await mgr.unsubscribe(handle, queue)


# ---------------------------------------------------------------- helpers --


def _installed_models() -> list[str]:
    try:
        import ollama

        return sorted(
            m.get("model") or m.get("name") for m in ollama.Client().list()["models"]
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("could not list Ollama models: %s", exc)
        return []


def run() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    s = load_settings()
    uvicorn.run(app, host=s.server.host, port=s.server.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run()
