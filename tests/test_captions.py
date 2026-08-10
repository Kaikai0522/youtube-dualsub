"""Human-uploaded caption ingestion.

YouTube does not serve captions as one cue per line. It serves a rolling
highlight: the same line re-sent every time the highlighted word advances.
Measured on a real 6-minute video, 85% of consecutive cues were exact repeats
of their predecessor and the median cue lasted 0.18s.
"""

from __future__ import annotations

from youtube_dualsub.sources.ytdlp_source import _collapse_rolling, _parse_json3, _parse_vtt


def test_rolling_repeats_become_one_line():
    cues = [
        (0.11, 0.13, "Hey guys!"),
        (0.13, 0.41, "Hey guys!"),
        (0.41, 0.67, "Hey guys!"),
        (0.68, 0.81, "Next line."),
        (0.81, 1.27, "Next line."),
    ]
    assert _collapse_rolling(cues) == [
        (0.11, 0.67, "Hey guys!"),
        (0.68, 1.27, "Next line."),
    ]


def test_a_line_genuinely_said_twice_stays_two_lines():
    """Merging must not swallow real repetition — people repeat themselves."""
    cues = [(0.0, 1.0, "Go!"), (30.0, 31.0, "Go!")]
    assert _collapse_rolling(cues) == cues


def test_repeats_separated_by_a_short_pause_are_kept_apart():
    cues = [(0.0, 1.0, "Go!"), (2.0, 3.0, "Go!")]
    assert len(_collapse_rolling(cues)) == 2


def test_the_merged_line_spans_the_whole_utterance():
    cues = [(5.0, 5.1, "x"), (5.1, 5.2, "x"), (5.2, 7.5, "x")]
    assert _collapse_rolling(cues) == [(5.0, 7.5, "x")]


def test_empty_input():
    assert _collapse_rolling([]) == []


class TestParsers:
    def test_json3(self):
        payload = (
            '{"events": ['
            '{"tStartMs": 100, "dDurationMs": 900, "segs": [{"utf8": "Hey "}, {"utf8": "guys!"}]},'
            '{"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "\\n"}]},'
            '{"tStartMs": 2000, "dDurationMs": 500, "segs": [{"utf8": "Next."}]}]}'
        )
        assert _parse_json3(payload) == [(0.1, 1.0, "Hey guys!"), (2.0, 2.5, "Next.")]

    def test_json3_that_is_not_json(self):
        assert _parse_json3("<html>nope</html>") == []

    def test_vtt(self):
        payload = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.500\n<c>Hey</c> guys!\n\n"
            "00:00:03.000 --> 00:00:04.000\nNext.\n"
        )
        assert _parse_vtt(payload) == [(1.0, 2.5, "Hey guys!"), (3.0, 4.0, "Next.")]

    def test_vtt_with_hours(self):
        payload = "WEBVTT\n\n01:02:03.500 --> 01:02:04.000\nLate.\n"
        assert _parse_vtt(payload)[0][0] == 3723.5
