"""Batch translation into Traditional Chinese.

The dangerous failure mode of this stage is not crashing — it is *lying*. A
model told to translate "naturally" will happily invent detail the speaker
never uttered, and the user cannot catch it, because not understanding the
English is why they are here. So the prompt carries three hard constraints
(decision Q27): carry the tone, add nothing, and when unsure prefer a literal
rendering over an invention. The English line stays on screen underneath as the
user's own check.

The other failure mode is misalignment. If the model returns 29 lines for a
batch of 30, naive zipping shifts every remaining subtitle in the video. So the
model answers in an index-keyed JSON object, the result is verified against the
indices that went in, and a mismatch triggers: retry, then halve the batch,
then translate one sentence at a time, then finally give up on those specific
sentences and show the English (decision Q19). Misalignment can cost a batch;
it can never cost the video.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Sequence

from ..config import Settings
from ..llm import LlmError, OllamaClient
from ..models import Sentence, Translation, TranslationStatus, VideoContext

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]
BatchCb = Callable[[list[Translation]], None]

_SYSTEM = """You are a professional subtitle translator working into {target}.

You translate spoken dialogue for viewers who do not understand the source language.
You obey three rules without exception:

1. CARRY THE TONE. Keep shouting, swearing, interjections and slang. If the speaker
   yells "GO GO GO", the subtitle yells too. Do not sanitise and do not formalise.
2. ADD NOTHING. Never introduce a fact, name, number or explanation that is not in
   the source line. No bracketed notes, no clarifications, no filling in gaps.
3. WHEN UNSURE, STAY LITERAL. A plain rendering is always better than a confident
   invention.

Other requirements:
- Write {target} only. Never use Simplified Chinese characters or mainland-Chinese
  vocabulary (影片 not 视频, 軟體 not 软件, 品質 not 质量, 網路 not 网络).
- One input line produces exactly one output line. Never merge, split or reorder.
- Keep lines short enough to read at speaking pace; drop filler that carries no
  meaning ("uh", "like") rather than padding the line.
- If a line is unintelligible, output your best guess of the words, not a description
  of the problem."""

_LITERAL_ADDENDUM = """
- This job is set to LITERAL mode: stay as close to the source wording as the target
  language allows, even where it reads stiffly."""

_USER = """{context_block}Translate every numbered line below into {target}.

Answer with a JSON object whose keys are the line numbers as strings and whose values
are the translations. Include every number, and nothing else:

{{"1": "...", "2": "..."}}

Lines:
{lines}"""


def translate_sentences(
    sentences: Sequence[Sentence],
    context: VideoContext,
    settings: Settings,
    *,
    glossary: dict[str, str] | None = None,
    client: OllamaClient | None = None,
    on_batch: BatchCb | None = None,
    progress: ProgressCb | None = None,
) -> list[Translation]:
    """Translate ``sentences``, emitting each finished batch through ``on_batch``.

    Batches are emitted as they complete so the viewer can start watching after
    the first one instead of waiting for the whole video (decision Q16).
    """
    if not sentences:
        return []

    cfg = settings.translate
    owns_client = client is None
    if client is None:
        client = OllamaClient(
            cfg.model,
            num_ctx=cfg.num_ctx,
            temperature=cfg.temperature,
            timeout_s=cfg.request_timeout_s,
            think=cfg.think,
        )

    system = _SYSTEM.format(target=cfg.target_language)
    if cfg.style == "literal":
        system += _LITERAL_ADDENDUM
    context_block = _context_block(context, glossary or {})

    results: list[Translation] = []
    #: Sliding window of (source, target) pairs, so pronouns and running jokes
    #: stay coherent across batch boundaries.
    history: list[tuple[str, str]] = []

    batches = list(_batched(sentences, cfg.batch_size))
    try:
        for i, batch in enumerate(batches):
            if progress:
                progress(i / len(batches), f"Translating batch {i + 1}/{len(batches)}")

            translated = _translate_batch(
                client, batch, system, context_block, history, settings
            )
            results.extend(translated)

            by_index = {t.index: t for t in translated}
            for sentence in batch:
                target = by_index.get(sentence.index)
                if target and target.status is TranslationStatus.OK:
                    history.append((sentence.text, target.text))
            del history[: max(0, len(history) - cfg.context_pairs)]

            if on_batch:
                on_batch(translated)
    finally:
        if owns_client:
            client.unload()

    if progress:
        failed = sum(1 for t in results if t.status is TranslationStatus.SOURCE_FALLBACK)
        note = f", {failed} left in English" if failed else ""
        progress(1.0, f"Translated {len(results)} lines{note}")
    return results


# --------------------------------------------------------------------------


def _translate_batch(
    client: OllamaClient,
    batch: Sequence[Sentence],
    system: str,
    context_block: str,
    history: list[tuple[str, str]],
    settings: Settings,
) -> list[Translation]:
    cfg = settings.translate
    prompt = _USER.format(
        context_block=context_block + _history_block(history),
        target=cfg.target_language,
        lines="\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(batch)),
    )

    max_tokens = len(batch) * cfg.max_tokens_per_line + cfg.max_tokens_overhead
    for attempt in range(cfg.max_retries):
        try:
            raw = client.complete_json(prompt, system=system, max_tokens=max_tokens)
            texts = _align(raw, len(batch))
        except (LlmError, _Misaligned) as exc:
            log.warning(
                "batch of %d failed on attempt %d/%d: %s",
                len(batch),
                attempt + 1,
                cfg.max_retries,
                exc,
            )
            continue
        return [
            Translation(index=s.index, text=text, status=TranslationStatus.OK)
            for s, text in zip(batch, texts, strict=True)
        ]

    # Halve and recurse: a batch that will not align at 30 lines usually aligns
    # at 15, and the smaller the batch the smaller the blast radius.
    if len(batch) > cfg.min_batch_size:
        mid = len(batch) // 2
        return _translate_batch(
            client, batch[:mid], system, context_block, history, settings
        ) + _translate_batch(
            client, batch[mid:], system, context_block, history, settings
        )
    if len(batch) > 1:
        out: list[Translation] = []
        for sentence in batch:
            out.extend(
                _translate_batch(client, [sentence], system, context_block, history, settings)
            )
        return out

    sentence = batch[0]
    log.error("giving up on sentence %d; showing the source line", sentence.index)
    return [
        Translation(
            index=sentence.index,
            text=sentence.text,
            status=TranslationStatus.SOURCE_FALLBACK,
        )
    ]


class _Misaligned(RuntimeError):
    pass


def _align(raw: object, expected: int) -> list[str]:
    """Turn the model's answer into exactly ``expected`` non-empty lines."""
    if isinstance(raw, list):
        raw = {str(i + 1): v for i, v in enumerate(raw)}
    if not isinstance(raw, dict):
        raise _Misaligned(f"expected a JSON object, got {type(raw).__name__}")

    normalized: dict[int, str] = {}
    for key, value in raw.items():
        try:
            index = int(str(key).strip().rstrip("."))
        except ValueError:
            continue
        if isinstance(value, str) and value.strip():
            normalized[index] = value.strip()

    missing = [i for i in range(1, expected + 1) if i not in normalized]
    if missing:
        raise _Misaligned(
            f"missing line(s) {missing[:5]}{'...' if len(missing) > 5 else ''} of {expected}"
        )
    return [normalized[i] for i in range(1, expected + 1)]


def _context_block(context: VideoContext, glossary: dict[str, str]) -> str:
    parts: list[str] = []
    if context.summary:
        parts.append(f"What this video is about:\n{context.summary}")

    # Curated glossary entries are listed last so that, if the model reads the
    # list top-down, the authoritative renderings are the freshest in mind.
    terms = {**context.terms, **glossary}
    if terms:
        rendered = "\n".join(f"- {src} -> {dst}" for src, dst in list(terms.items())[:80])
        parts.append(
            "Established terminology. Use these renderings exactly:\n" + rendered
        )
    return "\n\n".join(parts) + "\n\n" if parts else ""


def _history_block(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    rendered = "\n".join(f"{src}\n  -> {dst}" for src, dst in history)
    return f"The immediately preceding lines, for continuity:\n{rendered}\n\n"


def _batched(items: Sequence[Sentence], size: int) -> Iterable[Sequence[Sentence]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
