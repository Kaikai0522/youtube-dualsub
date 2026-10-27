"""Whole-video context: a summary, the subject, and a terminology list.

This stage is the entire reason for choosing a pre-processing architecture over
live translation. A per-line translator does not know that this video is a
Minecraft manhunt, so it renders "pods" as 豆莢 and "gapples" as a mystery; nor
that ツナ is a person's name rather than tuna. A summary and a term list, pinned
into every translation batch, are what buy consistency across a whole video.

Terminology is produced here without any hand-written glossary:

1. The model reads the transcript and the video's own metadata (title,
   description, tags — which is where a clip channel lists every member's
   name) and proposes the names and jargon it finds, each with a canonical
   name to look up and its own best rendering.
2. Proposals that do not actually occur in the transcript or metadata are
   dropped: the model is asked to find terms, not to brainstorm them.
3. ``terms.ground`` replaces the model's renderings with the zh-TW titles of
   matching wiki pages wherever one exists.

A transcript that fits comfortably in the context window is read in one
request. Longer ones are summarised in chunks and merged, with term proposals
pooled rather than re-summarised, so a term mentioned once in part 7 survives.
"""

from __future__ import annotations

import logging
from typing import Callable

from .. import terms as terms_mod
from ..config import Settings
from ..lang import is_cjk, language_name
from ..llm import LlmError, OllamaClient
from ..models import Sentence, VideoContext

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

#: Streamed answer-token ceiling (see llm._stream_bounded). A full answer with
#: 80 terms is about 3000 tokens; past this the model is looping.
_ANSWER_BUDGET = 5000


_SYSTEM = (
    "You are a media analyst preparing a subtitle translator. You read transcripts and "
    "report facts about them. You never invent details that are not present in the text."
)

_TERMS_SPEC = """"terms": a list of the proper nouns and specialist vocabulary in the transcript:
  people and nicknames, groups, places, titles of works, and the jargon of the subject
  (game items, creatures, mechanics, slang particular to the community). Skip ordinary
  vocabulary any dictionary handles. Each entry:
    "source": exactly as it is written in the transcript
    "lookup": the canonical full name a wiki would title its page with, in English for
              game and technical terms (e.g. "gapple" -> "Golden Apple"), and in the
              original script for people ("ツナ" -> "猫汰つな" if the metadata says so).
              Name the thing itself, not one edition of it, unless the speaker means one.
    "kind":   one of person, group, place, work, organization, product, jargon, other
    "target": its established name in Traditional Chinese as used in Taiwan. For a
              person with no established Chinese name, repeat the original name."""

_SINGLE_PROMPT = """Video metadata:
{metadata}

Transcript ({language}):
{transcript}

Answer with one JSON object:
{{
  "summary": "<three to five sentences: what this video is, who is in it, what happens>",
  "subject": "<the game, show or topic in two or three words, e.g. Minecraft>",
  "terms": [{{"source": "...", "lookup": "...", "kind": "...", "target": "..."}}]
}}

{terms_spec}
At most {max_terms} entries, most important first."""

_MAP_PROMPT = """Video metadata:
{metadata}

Transcript excerpt ({language}):
{transcript}

Answer with one JSON object:
{{
  "summary": "<two sentences describing what happens in this excerpt>",
  "terms": [{{"source": "...", "lookup": "...", "kind": "...", "target": "..."}}]
}}

{terms_spec}
At most 20 entries."""

_REDUCE_PROMPT = """Video metadata:
{metadata}

Below are summaries of consecutive parts of this video.

{parts}

Answer with one JSON object:
{{
  "summary": "<three to five sentences: what this video is, who is in it, what happens>",
  "subject": "<the game, show or topic in two or three words, e.g. Minecraft>"
}}"""


def build_context(
    sentences: list[Sentence],
    settings: Settings,
    *,
    metadata: str = "",
    language: str | None = None,
    client: OllamaClient | None = None,
    lookup_cache: terms_mod.LookupCache | None = None,
    progress: ProgressCb | None = None,
) -> VideoContext:
    cfg = settings.context
    if not cfg.enabled or not sentences:
        return VideoContext()

    owns_client = client is None
    if client is None:
        client = OllamaClient(
            cfg.model or settings.translate.model,
            num_ctx=cfg.num_ctx,
            temperature=0.2,
            timeout_s=settings.translate.request_timeout_s,
            think=settings.translate.think,
        )

    try:
        transcript = "\n".join(s.text for s in sentences)
        if _estimated_tokens(transcript) <= _single_pass_budget(settings):
            if progress:
                progress(0.1, "Reading the whole transcript")
            summary, subject, candidates = _single(
                client, transcript, metadata, language, cfg.max_auto_terms
            )
        else:
            summary, subject, candidates = _map_reduce(
                client, sentences, metadata, language, settings, progress
            )
    except LlmError as exc:
        # Context is an enhancement, not a requirement: losing it costs
        # terminology consistency, not the subtitles themselves (Q19).
        log.warning("context building failed, translating without it: %s", exc)
        return VideoContext()
    finally:
        if owns_client:
            client.unload()

    haystack = f"{metadata}\n{transcript}".lower()
    candidates = [c for c in candidates if c.source.lower() in haystack][: cfg.max_auto_terms]

    if progress:
        progress(0.9, f"Looking up {len(candidates)} terms")
    if cfg.wiki_lookup and candidates:
        glossary = terms_mod.ground(
            candidates,
            language=language,
            subject=subject,
            domain_wikis=cfg.domain_wikis,
            cache=lookup_cache,
            timeout_s=cfg.wiki_timeout_s,
        )
    else:
        glossary = {c.source: c.guess for c in candidates if c.guess}

    if progress:
        progress(1.0, f"Context ready ({len(glossary)} terms)")
    return VideoContext(summary=summary, terms=glossary, domain=subject)


def metadata_text(title: str, uploader: str, description: str, tags: tuple[str, ...]) -> str:
    """The parts of a video's metadata worth showing the model, bounded."""
    lines = [f"Title: {title}"]
    if uploader:
        lines.append(f"Channel: {uploader}")
    if description.strip():
        lines.append("Description:\n" + description.strip()[:1500])
    if tags:
        lines.append("Tags: " + ", ".join(tags[:30]))
    return "\n".join(lines)


# --------------------------------------------------------------------------


def _estimated_tokens(text: str) -> int:
    # About four characters per token in English, about one in Japanese.
    return len(text) if is_cjk(text[:2000]) else len(text) // 4


def _single_pass_budget(settings: Settings) -> int:
    # Leave room for the metadata, the instructions and a long JSON answer.
    return min(settings.context.single_pass_chars, settings.context.num_ctx - 4000)


def _single(client, transcript, metadata, language, max_terms):
    data = client.complete_json(
        _SINGLE_PROMPT.format(
            metadata=metadata or "(none)",
            language=language_name(language),
            transcript=transcript,
            terms_spec=_TERMS_SPEC,
            max_terms=max_terms,
        ),
        system=_SYSTEM,
        max_tokens=_ANSWER_BUDGET,
    )
    if not isinstance(data, dict):
        raise LlmError("context answer was not a JSON object")
    return (
        str(data.get("summary") or "").strip(),
        str(data.get("subject") or "").strip(),
        _candidates(data.get("terms")),
    )


def _map_reduce(client, sentences, metadata, language, settings, progress):
    cfg = settings.context
    chunks = _chunk(sentences, cfg.chunk_sentences, cfg.max_chunks)
    summaries: list[str] = []
    pooled: dict[str, terms_mod.Candidate] = {}
    for i, chunk in enumerate(chunks):
        if progress:
            progress(0.8 * i / len(chunks), f"Summarising part {i + 1}/{len(chunks)}")
        try:
            data = client.complete_json(
                _MAP_PROMPT.format(
                    metadata=metadata or "(none)",
                    language=language_name(language),
                    transcript="\n".join(s.text for s in chunk),
                    terms_spec=_TERMS_SPEC,
                ),
                system=_SYSTEM,
                max_tokens=_ANSWER_BUDGET,
            )
        except LlmError as exc:
            log.warning("chunk summary failed: %s", exc)
            continue
        if not isinstance(data, dict):
            continue
        summaries.append(str(data.get("summary") or "").strip())
        for c in _candidates(data.get("terms")):
            pooled.setdefault(c.source.lower(), c)

    if progress:
        progress(0.85, "Merging summaries")
    parts = "\n".join(f"Part {i + 1}: {s}" for i, s in enumerate(summaries) if s)
    try:
        data = client.complete_json(
            _REDUCE_PROMPT.format(metadata=metadata or "(none)", parts=parts),
            system=_SYSTEM,
            max_tokens=_ANSWER_BUDGET,
        )
        summary = str(data.get("summary") or "").strip()
        subject = str(data.get("subject") or "").strip()
    except (LlmError, AttributeError) as exc:
        log.warning("summary reduction failed, concatenating instead: %s", exc)
        summary, subject = " ".join(summaries)[:1500], ""
    return summary, subject, list(pooled.values())


def _chunk(sentences: list[Sentence], size: int, max_chunks: int) -> list[list[Sentence]]:
    """Split into at most ``max_chunks`` pieces, widening the pieces if needed."""
    size = max(size, -(-len(sentences) // max_chunks))
    return [sentences[i : i + size] for i in range(0, len(sentences), size)]


def _candidates(raw: object) -> list[terms_mod.Candidate]:
    if not isinstance(raw, list):
        return []
    out: list[terms_mod.Candidate] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        key = source.lower()
        if not source or key in seen or len(source) > 60:
            continue
        seen.add(key)
        out.append(
            terms_mod.Candidate(
                source=source,
                lookup=str(item.get("lookup") or source).strip(),
                kind=str(item.get("kind") or "other").strip().lower(),
                guess=str(item.get("target") or "").strip(),
            )
        )
    return out
