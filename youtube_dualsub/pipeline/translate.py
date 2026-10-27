"""Batch translation into Traditional Chinese.

The dangerous failure mode of this stage is not crashing — it is *lying*. A
model told to translate "naturally" will happily invent detail the speaker
never uttered, and the user cannot catch it, because not understanding the
source language is why they are here. So the prompt carries three hard
constraints (decision Q27): carry the tone, add nothing, and when unsure prefer
a literal rendering over an invention. The source line stays on screen
underneath as the user's own check.

A line is a subtitle cue, not necessarily a sentence. Human captions are cut
where a person chose to cut them, often mid-sentence, and a Japanese sentence
puts its verb last — the half on screen now may only make sense once the next
line arrives. So every batch also shows the model the lines that *follow* it,
marked as context, as well as the pairs that preceded it.

The other failure mode is misalignment. If the model returns 29 lines for a
batch of 30, naive zipping shifts every remaining subtitle in the video. So the
model answers in an index-keyed JSON object, the result is verified against the
indices that went in, and a mismatch triggers: retry, then halve the batch,
then translate one sentence at a time, then finally give up on those specific
sentences and show the original line (decision Q19). Misalignment can cost a batch;
it can never cost the video.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence


from ..config import Settings
from ..lang import is_cjk, language_name
from ..llm import LlmError, OllamaClient
from ..models import Sentence, Translation, TranslationStatus, VideoContext

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]
BatchCb = Callable[[list[Translation]], None]

_SYSTEM = """You are a professional subtitle translator working from {source} into {target}.

You translate spoken dialogue for viewers who do not understand {source}.
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
- A line is one subtitle and may be only part of a sentence that continues on the
  next line. Translate each line as that part, so the lines read naturally in order.
  Never move words from one numbered line into another.
- Use the established terminology exactly. A person's name that is not listed stays
  as it is written in the source.
- Keep lines short enough to read at speaking pace; drop filler that carries no
  meaning ("uh", "like", "えっと") rather than padding the line.
- If a line is unintelligible, output your best guess of the words, not a description
  of the problem.{addendum}"""

_JAPANESE_ADDENDUM = """
- Japanese: carry casual or polite speech through word choice, not explanation.
  先輩 is 前輩. Laughter written as w, ww or 草 becomes 笑 or （笑）. Do not add
  subjects or pronouns the Japanese leaves out unless Chinese grammar needs one."""

_LITERAL_ADDENDUM = """
- This job is set to LITERAL mode: stay as close to the source wording as the target
  language allows, even where it reads stiffly."""

_USER = """{context_block}{lookahead_block}Translate every numbered line below into {target}.

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
    language: str | None = None,
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

    system = _system_prompt(settings, language)
    terms = {**context.terms, **(glossary or {})}

    # (position of the batch's first line in `sentences`, the batch)
    batches = [
        (i, sentences[i : i + cfg.batch_size]) for i in range(0, len(sentences), cfg.batch_size)
    ]
    lanes = _lanes(batches, cfg.parallel)

    results: list[Translation] = []
    lock = threading.Lock()
    stop = threading.Event()
    finished = [0]

    def run_lane(lane: list[tuple[int, Sequence[Sentence]]]) -> None:
        #: Sliding window of (source, target) pairs, so pronouns and running
        #: jokes stay coherent across batch boundaries within the lane.
        history: list[tuple[str, str]] = []
        for start, batch in lane:
            if stop.is_set():
                return
            end = start + len(batch)
            upcoming = [s.text for s in sentences[end : end + cfg.lookahead_lines]]
            context_block = _context_block(context.summary, _relevant(terms, batch))
            translated = _translate_batch(
                client, batch, system, context_block, history, settings,
                upcoming=upcoming,
            )

            by_index = {t.index: t for t in translated}
            for sentence in batch:
                target = by_index.get(sentence.index)
                if target and target.status is TranslationStatus.OK:
                    history.append((sentence.text, target.text))
            del history[: max(0, len(history) - cfg.context_pairs)]

            # Callbacks write to SQLite and rebuild cues from shared state, so
            # lanes take turns delivering; only the model calls overlap.
            with lock:
                results.extend(translated)
                finished[0] += 1
                if progress:
                    progress(
                        finished[0] / len(batches),
                        f"Translated batch {finished[0]}/{len(batches)}",
                    )
                if on_batch:
                    on_batch(translated)

    try:
        if len(lanes) == 1:
            run_lane(lanes[0])
        else:
            with ThreadPoolExecutor(len(lanes), thread_name_prefix="translate") as pool:
                futures = [pool.submit(run_lane, lane) for lane in lanes]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    # A cancelled job (raised from on_batch) stops every lane
                    # at its next batch; finished batches are already saved.
                    stop.set()
                    raise
    finally:
        if owns_client:
            client.unload()

    results.sort(key=lambda t: t.index)
    if progress:
        failed = sum(1 for t in results if t.status is TranslationStatus.SOURCE_FALLBACK)
        note = f", {failed} left untranslated" if failed else ""
        progress(1.0, f"Translated {len(results)} lines{note}")
    return results


def _lanes(batches: list, parallel: int) -> list[list]:
    """Split the batches into contiguous runs, one per concurrent request.

    Contiguous rather than interleaved, so each lane still carries its own
    running history; only the lane boundaries start cold. The first lane covers
    the start of the video, and even at a quarter of the throughput it stays
    far ahead of playback.
    """
    count = max(1, min(parallel, len(batches)))
    size = -(-len(batches) // count)
    return [batches[i : i + size] for i in range(0, len(batches), size)]


# --------------------------------------------------------------------------


def _translate_batch(
    client: OllamaClient,
    batch: Sequence[Sentence],
    system: str,
    context_block: str,
    history: list[tuple[str, str]],
    settings: Settings,
    *,
    upcoming: Sequence[str] = (),
) -> list[Translation]:
    cfg = settings.translate
    # The model reads the conversation in the order it happened: what came
    # before (history), then what comes after (look-ahead), then the lines.
    prompt = _USER.format(
        context_block=context_block + _history_block(history),
        lookahead_block=_lookahead_block(upcoming),
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
        first, second = batch[:mid], batch[mid:]
        return _translate_batch(
            client, first, system, context_block, history, settings,
            upcoming=[s.text for s in second][: cfg.lookahead_lines] or upcoming,
        ) + _translate_batch(
            client, second, system, context_block, history, settings, upcoming=upcoming
        )
    if len(batch) > 1:
        out: list[Translation] = []
        for i, sentence in enumerate(batch):
            rest = [s.text for s in batch[i + 1 :]] + list(upcoming)
            out.extend(
                _translate_batch(
                    client, [sentence], system, context_block, history, settings,
                    upcoming=rest[: cfg.lookahead_lines],
                )
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


def _system_prompt(settings: Settings, language: str | None) -> str:
    cfg = settings.translate
    addendum = _JAPANESE_ADDENDUM if (language or "").startswith("ja") else ""
    if cfg.style == "literal":
        addendum += _LITERAL_ADDENDUM
    return _SYSTEM.format(
        source=language_name(language), target=cfg.target_language, addendum=addendum
    )


def _relevant(terms: dict[str, str], batch: Sequence[Sentence]) -> dict[str, str]:
    """Only the terms that occur in this batch.

    Sending all eighty terms with every batch cost prefill time on every
    request and diluted the ones that mattered; a model follows a list of four
    far more reliably than a list of eighty.
    """
    text = "\n".join(s.text for s in batch)
    lowered = text.lower()
    out: dict[str, str] = {}
    for source, target in terms.items():
        if not source:
            continue
        if is_cjk(source):
            hit = source in text
        else:
            hit = re.search(rf"(?<![A-Za-z]){re.escape(source.lower())}", lowered) is not None
        if hit:
            out[source] = target
    return out


def _context_block(summary: str, terms: dict[str, str]) -> str:
    parts: list[str] = []
    if summary:
        parts.append(f"What this video is about:\n{summary}")
    if terms:
        rendered = "\n".join(f"- {src} -> {dst}" for src, dst in terms.items())
        parts.append(
            "Established terminology. Use these renderings exactly:\n" + rendered
        )
    return "\n\n".join(parts) + "\n\n" if parts else ""


def _lookahead_block(upcoming: Sequence[str]) -> str:
    if not upcoming:
        return ""
    rendered = "\n".join(upcoming)
    return (
        "The lines that come right after, for context only. Do not translate them:\n"
        f"{rendered}\n\n"
    )


def _history_block(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    rendered = "\n".join(f"{src}\n  -> {dst}" for src, dst in history)
    return f"The immediately preceding lines, for continuity:\n{rendered}\n\n"



