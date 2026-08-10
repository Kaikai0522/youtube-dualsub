"""Whole-video context: a map-reduce summary plus a candidate glossary.

This stage is the entire reason for choosing a pre-processing architecture over
live translation. Google Translate does not know that this video is a Minecraft
manhunt, so it renders "pods" as 豆莢 and "gapples" as a mystery. A summary and
a term list, pinned into every translation batch, are what buy consistency
across a 52-minute video — the thing a per-sentence translator structurally
cannot do.

A 3-hour video will not fit in one prompt, so summaries are produced per chunk
and then reduced into one. Terms extracted here are *candidates only*: the
curated YAML glossaries outrank them at substitution time, because a model that
confidently proposes 爬行者 for "creeper" must not be allowed to win.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..config import Settings
from ..llm import LlmError, OllamaClient
from ..models import Sentence, VideoContext

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

_MAP_SYSTEM = (
    "You are a media analyst. You read transcript excerpts and report facts about them. "
    "You never invent details that are not present in the text."
)

_MAP_PROMPT = """Read this transcript excerpt and answer in JSON.

Return exactly this shape:
{{
  "summary": "<two sentences describing what happens in this excerpt>",
  "terms": [{{"source": "<proper noun or piece of jargon>", "target": "<its best Traditional Chinese rendering>"}}]
}}

Rules for "terms":
- Only include names, places, game/technical jargon, and recurring in-jokes.
- Skip ordinary vocabulary that any dictionary handles.
- At most 12 entries.
- If you are unsure of the established Chinese name, leave "target" as an empty string
  rather than guessing.

Transcript excerpt:
{excerpt}
"""

_REDUCE_SYSTEM = _MAP_SYSTEM

_REDUCE_PROMPT = """Below are summaries of consecutive parts of one video, plus terminology
collected from each part.

Produce a single JSON object:
{{
  "summary": "<three to five sentences: what this video is, who is in it, what happens>",
  "terms": [{{"source": "...", "target": "..."}}]
}}

For "terms": merge duplicates, keep the {max_terms} most important entries, and drop any
entry whose "target" is empty or that you are not confident about.

Parts:
{parts}
"""


def build_context(
    sentences: list[Sentence],
    settings: Settings,
    *,
    client: OllamaClient | None = None,
    progress: ProgressCb | None = None,
) -> VideoContext:
    cfg = settings.context
    if not cfg.enabled or not sentences:
        return VideoContext()

    owns_client = client is None
    if client is None:
        client = OllamaClient(
            cfg.model or settings.translate.model,
            num_ctx=settings.translate.num_ctx,
            temperature=0.2,
            timeout_s=settings.translate.request_timeout_s,
            think=settings.translate.think,
        )

    try:
        chunks = _chunk(sentences, cfg.chunk_sentences, cfg.max_chunks)
        parts: list[dict] = []
        for i, chunk in enumerate(chunks):
            if progress:
                progress(i / len(chunks), f"Summarising part {i + 1}/{len(chunks)}")
            parts.append(_map_chunk(client, chunk))

        if progress:
            progress(0.9, "Merging summaries")
        context = _reduce(client, parts, cfg.max_auto_terms) if len(parts) > 1 else _single(parts)
    except LlmError as exc:
        # Context is an enhancement, not a requirement: losing it costs
        # terminology consistency, not the subtitles themselves (Q19).
        log.warning("context building failed, translating without it: %s", exc)
        context = VideoContext()
    finally:
        if owns_client:
            client.unload()

    if progress:
        progress(1.0, f"Context ready ({len(context.terms)} terms)")
    return context


def _chunk(sentences: list[Sentence], size: int, max_chunks: int) -> list[list[Sentence]]:
    """Split into at most ``max_chunks`` pieces, widening the pieces if needed."""
    size = max(size, -(-len(sentences) // max_chunks))
    return [sentences[i : i + size] for i in range(0, len(sentences), size)]


def _map_chunk(client: OllamaClient, chunk: list[Sentence]) -> dict:
    excerpt = "\n".join(s.text for s in chunk)
    try:
        data = client.complete_json(
            _MAP_PROMPT.format(excerpt=excerpt), system=_MAP_SYSTEM
        )
    except LlmError as exc:
        log.warning("chunk summary failed: %s", exc)
        return {"summary": "", "terms": []}
    return {
        "summary": str(data.get("summary") or "").strip(),
        "terms": _clean_terms(data.get("terms")),
    }


def _single(parts: list[dict]) -> VideoContext:
    part = parts[0] if parts else {"summary": "", "terms": []}
    return VideoContext(
        summary=part["summary"],
        terms={t["source"]: t["target"] for t in part["terms"]},
    )


def _reduce(client: OllamaClient, parts: list[dict], max_terms: int) -> VideoContext:
    rendered = "\n\n".join(
        f"Part {i + 1} summary: {p['summary']}\n"
        f"Part {i + 1} terms: "
        + ", ".join(f"{t['source']} = {t['target']}" for t in p["terms"])
        for i, p in enumerate(parts)
    )
    try:
        data = client.complete_json(
            _REDUCE_PROMPT.format(parts=rendered, max_terms=max_terms),
            system=_REDUCE_SYSTEM,
        )
    except LlmError as exc:
        log.warning("summary reduction failed, concatenating instead: %s", exc)
        merged: dict[str, str] = {}
        for p in parts:
            for t in p["terms"]:
                merged.setdefault(t["source"], t["target"])
        return VideoContext(
            summary=" ".join(p["summary"] for p in parts if p["summary"])[:1500],
            terms=dict(list(merged.items())[:max_terms]),
        )

    terms = _clean_terms(data.get("terms"))[:max_terms]
    return VideoContext(
        summary=str(data.get("summary") or "").strip(),
        terms={t["source"]: t["target"] for t in terms},
    )


def _clean_terms(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        key = source.lower()
        if not source or not target or key in seen:
            continue
        seen.add(key)
        out.append({"source": source, "target": target})
    return out
