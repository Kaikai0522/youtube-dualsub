"""A very small Ollama wrapper.

It exists to hold three pieces of knowledge that would otherwise be duplicated
between the summarisation and translation stages:

* reasoning models wrap their scratchpad in ``<think>`` — that must never reach
  a subtitle, so it is stripped centrally;
* JSON mode still occasionally arrives fenced in Markdown, so extraction is
  tolerant;
* the model must be evictable, because translation runs last and should hand
  its ~9 GB back rather than sit resident.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import ollama

log = logging.getLogger(__name__)


class LlmError(RuntimeError):
    pass


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class OllamaClient:
    def __init__(
        self,
        model: str,
        *,
        num_ctx: int = 8192,
        temperature: float = 0.3,
        timeout_s: float = 180.0,
        think: bool = False,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self._think_supported = True
        self._client = ollama.Client(timeout=timeout_s)

    def _chat_kwargs(self, messages, json_mode: bool, options: dict, stream: bool) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": "json" if json_mode else None,
            "options": options,
            "stream": stream,
        }
        # Subtitle translation has nothing to reason about, and reasoning is
        # ruinous here: gemma4:12b spent up to 4000 thinking tokens on a
        # three-word line, filling the context before it wrote any answer — the
        # cause of the empty responses that looked like every other bug first.
        if not self.think and self._think_supported:
            kwargs["think"] = False
        return kwargs

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature if temperature is None else temperature,
        }

        try:
            if max_tokens:
                content = self._stream_bounded(messages, json_mode, options, max_tokens)
            else:
                kwargs = self._chat_kwargs(messages, json_mode, options, stream=False)
                kwargs.pop("stream")
                response = self._client.chat(**kwargs)
                content = response["message"].get("content") or ""
        except LlmError:
            raise
        except TypeError as exc:
            # Older Ollama builds reject `think`; retry once without it.
            if "think" not in str(exc) or not self._think_supported:
                raise LlmError(f"Ollama call failed ({self.model}): {exc}") from exc
            log.info("this Ollama build does not accept `think`; continuing without it")
            self._think_supported = False
            return self.complete(
                prompt, system=system, json_mode=json_mode,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a degradable failure
            raise LlmError(f"Ollama call failed ({self.model}): {exc}") from exc

        return _THINK.sub("", content).strip()

    def _stream_bounded(
        self,
        messages: list[dict[str, str]],
        json_mode: bool,
        options: dict[str, Any],
        max_tokens: int,
    ) -> str:
        """Generate with a client-side budget instead of ``num_predict``.

        Some batches send the model into a loop where it generates until the
        context window is full and returns nothing — a minute and a half of GPU
        time for no answer, several times per batch as the retry ladder
        descends. The obvious guard, ``num_predict``, is worse than the disease:
        setting it made *every* request behave that way, including ones that
        succeeded moments earlier without it.

        So the budget is enforced here rather than asked of the model. Sampling
        is untouched; we simply stop reading once the answer is clearly longer
        than the task can justify, and let the caller retry with fewer lines.
        """
        chunks: list[str] = []
        count = 0
        thinking = 0
        for part in self._client.chat(
            **self._chat_kwargs(messages, json_mode, options, stream=True)
        ):
            message = part["message"]
            # Reasoning arrives in its own `thinking` field, not inline in the
            # content, and must not be charged against the answer's budget —
            # doing so cut models off before they had written anything at all.
            if message.get("thinking"):
                thinking += 1
                continue
            piece = message.get("content") or ""
            if not piece:
                continue
            chunks.append(piece)
            count += 1
            if count > max_tokens:
                raise LlmError(
                    f"{self.model} produced more than {max_tokens} answer tokens "
                    f"without finishing ({thinking} spent reasoning). The batch is "
                    f"too large for it; retrying smaller."
                )
        return "".join(chunks)

    def complete_json(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> Any:
        raw = self.complete(prompt, system=system, json_mode=True, max_tokens=max_tokens)
        return parse_json(raw)

    def unload(self) -> None:
        """Ask Ollama to evict the model so the VRAM comes back."""
        try:
            self._client.generate(model=self.model, prompt="", keep_alive=0)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not unload %s: %s", self.model, exc)

    def ensure_available(self) -> None:
        try:
            names = {m.get("model") or m.get("name") for m in self._client.list()["models"]}
        except Exception as exc:  # noqa: BLE001
            raise LlmError(
                f"Cannot reach Ollama: {exc}. Is the Ollama service running?"
            ) from exc
        if self.model not in names:
            raise LlmError(
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )


def parse_json(raw: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace/bracket pair.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LlmError(f"Model did not return valid JSON: {raw[:300]!r}")
