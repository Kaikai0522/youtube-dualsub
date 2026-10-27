"""Batch alignment and its fallbacks.

Misalignment is the failure that silently shifts every remaining subtitle in a
52-minute video, so it gets tested harder than the happy path.
"""

from __future__ import annotations

import pytest

from youtube_dualsub.config import Settings, _merge
from youtube_dualsub.llm import parse_json
from youtube_dualsub.models import Sentence, TranslationStatus, VideoContext
from youtube_dualsub.pipeline.translate import _Misaligned, _align, translate_sentences


class TestAlign:
    def test_a_well_formed_answer(self):
        assert _align({"1": "一", "2": "二"}, 2) == ["一", "二"]

    def test_out_of_order_keys_are_reordered(self):
        assert _align({"2": "二", "1": "一"}, 2) == ["一", "二"]

    def test_a_bare_array_is_accepted(self):
        assert _align(["一", "二"], 2) == ["一", "二"]

    def test_a_missing_line_is_caught(self):
        with pytest.raises(_Misaligned, match="missing"):
            _align({"1": "一", "3": "三"}, 3)

    def test_an_empty_value_counts_as_missing(self):
        with pytest.raises(_Misaligned):
            _align({"1": "一", "2": "   "}, 2)

    def test_extra_lines_are_ignored_rather_than_shifting_everything(self):
        assert _align({"1": "一", "2": "二", "3": "spurious"}, 2) == ["一", "二"]

    def test_a_non_object_answer_is_caught(self):
        with pytest.raises(_Misaligned):
            _align("just a string", 1)


class TestParseJson:
    def test_plain_json(self):
        assert parse_json('{"1": "x"}') == {"1": "x"}

    def test_markdown_fenced_json(self):
        assert parse_json('```json\n{"1": "x"}\n```') == {"1": "x"}

    def test_json_with_a_preamble(self):
        assert parse_json('Sure! Here you go:\n{"1": "x"}') == {"1": "x"}


class FakeClient:
    """Stands in for Ollama. Records prompts and replays scripted answers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.max_tokens: list[int | None] = []
        self.unloaded = False

    def complete_json(self, prompt, *, system=None, max_tokens=None):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        self.max_tokens.append(max_tokens)
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return answer

    def unload(self):
        self.unloaded = True


def sentences(n: int) -> list[Sentence]:
    return [Sentence(index=i, start=i, end=i + 1, text=f"line {i}") for i in range(n)]


def small_batches(**overrides) -> Settings:
    return _merge(
        Settings(),
        {"translate": {
            "batch_size": 2, "min_batch_size": 2, "max_retries": 1, "parallel": 1, **overrides,
        }},
    )


def test_batches_are_emitted_as_they_finish():
    """Decision Q16: the viewer starts watching after batch one, not after all 30."""
    client = FakeClient([{"1": "一", "2": "二"}, {"1": "三", "2": "四"}])
    seen: list[list[str]] = []

    translate_sentences(
        sentences(4), VideoContext(), small_batches(),
        client=client, on_batch=lambda b: seen.append([t.text for t in b]),
    )

    assert seen == [["一", "二"], ["三", "四"]]


def test_a_pair_that_only_aligns_one_at_a_time_still_gets_translated():
    """Rung three of the ladder: give up on batching before giving up on meaning."""
    client = FakeClient([{"1": "只有一行"}, {"1": "第一"}, {"1": "第二"}])
    result = translate_sentences(sentences(2), VideoContext(), small_batches(), client=client)

    assert [t.text for t in result] == ["第一", "第二"]
    assert all(t.status is TranslationStatus.OK for t in result)


def test_halving_recovers_a_batch_the_model_could_not_align():
    client = FakeClient([
        {"1": "一"},                      # a batch of 4 comes back with one line
        {"1": "一", "2": "二"},           # ...but each half aligns
        {"1": "三", "2": "四"},
    ])
    settings = _merge(
        Settings(),
        {"translate": {"batch_size": 4, "min_batch_size": 2, "max_retries": 1}},
    )
    result = translate_sentences(sentences(4), VideoContext(), settings, client=client)

    assert [t.text for t in result] == ["一", "二", "三", "四"]


def test_when_nothing_aligns_the_english_is_shown_rather_than_a_guess():
    """Decision Q19: the bottom rung is always 'show the source', never 'invent'."""
    client = FakeClient([])  # every answer is an empty object -> never aligns
    result = translate_sentences(sentences(2), VideoContext(), small_batches(), client=client)

    assert [t.status for t in result] == [TranslationStatus.SOURCE_FALLBACK] * 2
    assert [t.text for t in result] == ["line 0", "line 1"]


def test_one_bad_batch_does_not_poison_the_rest():
    client = FakeClient([
        {"1": "一", "2": "二"},          # batch 1 fine
        {"1": "broken"},                  # batch 2 misaligned...
        {"1": "三"}, {"1": "四"},         # ...and recovers one line at a time
    ])
    result = translate_sentences(sentences(4), VideoContext(), small_batches(), client=client)

    assert [t.text for t in result] == ["一", "二", "三", "四"]
    assert all(t.status is TranslationStatus.OK for t in result)


def test_context_and_glossary_reach_the_prompt():
    client = FakeClient([{"1": "一", "2": "二"}])
    lines = [
        Sentence(index=0, start=0, end=1, text="He ate a gapple."),
        Sentence(index=1, start=1, end=2, text="A Creeper blew up."),
    ]
    translate_sentences(
        lines,
        VideoContext(summary="A Minecraft manhunt.", terms={"gapple": "金蘋果"}),
        small_batches(),
        glossary={"creeper": "苦力怕"},
        client=client,
    )

    prompt = client.prompts[0]
    assert "A Minecraft manhunt." in prompt
    assert "gapple -> 金蘋果" in prompt
    assert "creeper -> 苦力怕" in prompt, "matching is case-insensitive"


def test_only_the_terms_a_batch_uses_are_sent():
    """Eighty terms with every batch cost prefill time and dilute the four that matter."""
    client = FakeClient([{"1": "一", "2": "二"}])
    translate_sentences(
        sentences(2),
        VideoContext(terms={"line": "行", "nether": "地獄", "ツナ": "ツナ"}),
        small_batches(),
        client=client,
    )

    prompt = client.prompts[0]
    assert "line -> 行" in prompt
    assert "nether" not in prompt
    assert "ツナ" not in prompt


def test_japanese_terms_match_without_word_boundaries():
    client = FakeClient([{"1": "一"}])
    translate_sentences(
        [Sentence(index=0, start=0, end=1, text="ツナ見て、ほら")],
        VideoContext(terms={"ツナ": "ツナ"}),
        small_batches(),
        language="ja",
        client=client,
    )
    assert "ツナ -> ツナ" in client.prompts[0]


def test_the_lines_that_follow_a_batch_are_shown_but_not_translated():
    """A caption line that stops mid-sentence needs the rest of the sentence."""
    client = FakeClient([{"1": "一", "2": "二"}, {"1": "三", "2": "四"}])
    translate_sentences(sentences(4), VideoContext(), small_batches(), client=client)

    before, lines = client.prompts[0].split("Lines:")
    assert "Do not translate them" in before
    assert "line 2" in before and "line 3" in before
    assert "line 2" not in lines


def test_the_source_language_shapes_the_system_prompt():
    client = FakeClient([{"1": "一", "2": "二"}])
    translate_sentences(
        sentences(2), VideoContext(), small_batches(), language="ja", client=client
    )

    system = client.systems[0]
    assert "from Japanese" in system
    assert "先輩" in system



def test_previous_lines_are_carried_forward_for_continuity():
    client = FakeClient([{"1": "一", "2": "二"}, {"1": "三", "2": "四"}])
    translate_sentences(sentences(4), VideoContext(), small_batches(), client=client)

    assert "line 0" not in client.prompts[0].split("Lines:")[0]
    continuity = client.prompts[1].split("Lines:")[0]
    assert "line 1" in continuity and "二" in continuity


def test_the_hard_constraints_are_always_in_the_system_prompt():
    """Decision Q27: the model must never be free to invent detail."""
    client = FakeClient([{"1": "一", "2": "二"}])
    translate_sentences(sentences(2), VideoContext(), small_batches(), client=client)

    system = client.systems[0]
    assert "ADD NOTHING" in system
    assert "CARRY THE TONE" in system
    assert "WHEN UNSURE, STAY LITERAL" in system
    assert "Simplified" in system


class TestParallelLanes:
    def test_lanes_are_contiguous_runs_of_batches(self):
        from youtube_dualsub.pipeline.translate import _lanes

        assert _lanes(list(range(10)), 4) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert _lanes(list(range(2)), 4) == [[0], [1]]
        assert _lanes([0], 4) == [[0]]

    def test_concurrent_lanes_still_translate_every_line_in_order(self):
        client = FakeClient([{"1": "甲", "2": "乙"}] * 4)
        emitted: list[int] = []

        result = translate_sentences(
            sentences(8), VideoContext(), small_batches(parallel=4),
            client=client, on_batch=lambda b: emitted.extend(t.index for t in b),
        )

        assert [t.index for t in result] == list(range(8))
        assert sorted(emitted) == list(range(8))
        assert all(t.status is TranslationStatus.OK for t in result)

    def test_a_cancellation_in_one_lane_stops_the_others(self):
        class Stop(Exception):
            pass

        client = FakeClient([{"1": "甲", "2": "乙"}] * 16)
        calls = {"n": 0}

        def on_batch(batch):
            calls["n"] += 1
            raise Stop()

        with pytest.raises(Stop):
            translate_sentences(
                sentences(32), VideoContext(), small_batches(parallel=2),
                client=client, on_batch=on_batch,
            )
        assert len(client.prompts) < 16, "lanes stopped instead of finishing the video"
