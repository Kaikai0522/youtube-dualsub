"""OllamaClient behaviour, especially the runaway-generation guard.

The failure this guards against was measured, not imagined: gemma4:12b handed a
30-line batch generated exactly `num_ctx - prompt` tokens and returned an empty
string ~100 seconds later, over and over, with nothing in the exception path to
explain it.
"""

from __future__ import annotations

import pytest

from youtube_dualsub.llm import LlmError, OllamaClient, parse_json


class FakeOllama:
    """Serves a whole response, or streams one token per chunk."""

    def __init__(self, response=None, stream_tokens=None):
        self.response = response
        self.stream_tokens = stream_tokens
        self.calls: list[dict] = []
        self.consumed = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            return self.response
        return self._emit()

    def _emit(self):
        for token in self.stream_tokens:
            self.consumed += 1
            if isinstance(token, dict):
                yield {"message": token}
            else:
                yield {"message": {"content": token}}


def thinking(n: int) -> list[dict]:
    """n reasoning chunks, which carry no content."""
    return [{"thinking": "...", "content": ""} for _ in range(n)]


def client_with(response=None, stream_tokens=None) -> tuple[OllamaClient, FakeOllama]:
    client = OllamaClient("test-model", num_ctx=8192, temperature=0.3)
    fake = FakeOllama(response, stream_tokens)
    client._client = fake
    return client, fake


def test_a_budget_never_becomes_num_predict():
    """num_predict was measured to break generation outright — it must not reappear."""
    client, fake = client_with(stream_tokens=["ok"])
    client.complete("prompt", max_tokens=1000)
    assert "num_predict" not in fake.calls[0]["options"]
    assert fake.calls[0]["stream"] is True


def test_without_a_budget_the_call_is_not_streamed():
    client, fake = client_with({"message": {"content": "hi"}})
    assert client.complete("prompt") == "hi"
    assert not fake.calls[0].get("stream")


def test_streamed_chunks_are_joined():
    client, _ = client_with(stream_tokens=["他", "完", "蛋", "了"])
    assert client.complete("prompt", max_tokens=10) == "他完蛋了"


def test_a_runaway_is_cut_off_at_the_budget():
    """The failure that cost ~100 GPU-seconds per attempt, now bounded."""
    client, fake = client_with(stream_tokens=["x"] * 5000)
    with pytest.raises(LlmError, match="without\\s+finishing"):
        client.complete("prompt", max_tokens=50)
    assert fake.consumed <= 60, "generation must stop near the budget, not run to the end"


def test_reasoning_is_not_charged_against_the_answer_budget():
    """The bug that made every model look broken.

    Reasoning arrives in its own `thinking` field. Counting those chunks cut
    models off before they had written a single character of the answer —
    gemma4:12b spends thousands of them on a three-word line.
    """
    client, _ = client_with(stream_tokens=thinking(3000) + ["他", "完", "蛋", "了"])
    assert client.complete("prompt", max_tokens=50) == "他完蛋了"


def test_reasoning_is_disabled_by_default():
    client, fake = client_with(stream_tokens=["ok"])
    client.complete("prompt", max_tokens=10)
    assert fake.calls[0]["think"] is False


def test_reasoning_can_be_left_on():
    client = OllamaClient("test-model", think=True)
    fake = FakeOllama(stream_tokens=["ok"])
    client._client = fake
    client.complete("prompt", max_tokens=10)
    assert "think" not in fake.calls[0]


def test_an_ollama_build_without_think_support_is_handled():
    class Picky:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if "think" in kwargs:
                raise TypeError("chat() got an unexpected keyword argument 'think'")
            return {"message": {"content": "fine"}}

    client = OllamaClient("test-model")
    client._client = Picky()
    assert client.complete("prompt") == "fine"
    assert client._think_supported is False


def test_reasoning_scratchpad_never_reaches_a_subtitle():
    client, _ = client_with(
        {"message": {"content": "<think>hmm, tricky</think>\n他完蛋了"}}
    )
    assert client.complete("prompt") == "他完蛋了"


def test_system_message_is_sent_as_its_own_turn():
    client, fake = client_with({"message": {"content": "ok"}})
    client.complete("prompt", system="be terse")
    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "prompt"},
    ]


def test_transport_failures_become_llm_errors():
    client = OllamaClient("test-model")

    class Boom:
        def chat(self, **kwargs):
            raise ConnectionError("refused")

    client._client = Boom()
    with pytest.raises(LlmError, match="Ollama call failed"):
        client.complete("prompt")


class TestParseJson:
    def test_fenced_output_is_accepted(self):
        """Dropping format=json must stay survivable, so fences must parse."""
        assert parse_json('```json\n{"1": "你好"}\n```') == {"1": "你好"}

    def test_trailing_commentary_is_ignored(self):
        assert parse_json('{"1": "x"} Hope that helps!') == {"1": "x"}

    def test_unparseable_output_raises(self):
        with pytest.raises(LlmError, match="did not return valid JSON"):
            parse_json("I cannot help with that.")
