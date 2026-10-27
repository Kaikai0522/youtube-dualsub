"""Filling Whisper's gaps with YouTube auto-captions — cases from VlaTbwhSSqk."""

from __future__ import annotations

from youtube_dualsub.models import Fragment, Word
from youtube_dualsub.pipeline.gapfill import fill_gaps
from youtube_dualsub.sources.ytdlp_source import _parse_json3_pieces


def heard(*spec: tuple[float, float, str]) -> list[Fragment]:
    words = [Word(start=s, end=e, text=t) for s, e, t in spec]
    return [Fragment(start=words[0].start, end=words[-1].end, text="".join(w.text for w in words), words=words)]


def test_a_shout_whisper_never_heard_is_added():
    """178.56 s: nothing from Whisper until さくさま at 181.56 s."""
    whisper = heard((181.56, 181.85, "さくさまー、"))
    lines = [(178.56, 181.28, [(178.56, "こいつやばいんだけど。")])]

    added = fill_gaps(whisper, lines)

    assert [f.text for f in added] == ["こいつやばいんだけど。"]
    assert added[0].start == 178.56 and added[0].end == 181.28


def test_a_line_whisper_wrote_another_way_is_not_duplicated():
    """YouTube's LINE低く is Whisper's ライン引く: the text differs, but Whisper's
    words fill the time, so nothing is missing."""
    whisper = heard((70.3, 70.9, "友達の"), (70.9, 71.6, "ライン"), (71.6, 72.2, "引く"),
                    (72.3, 73.4, "いやもう友達かなって"))
    lines = [(70.32, 73.48, [(70.32, "LINE低く。")])]

    assert fill_gaps(whisper, lines) == []


def test_a_line_whisper_already_has_is_not_added():
    whisper = heard((53.42, 53.98, "そう、だから"), (53.98, 54.94, "喋ったことない。"))
    lines = [(53.3, 55.0, [(53.3, "そう、だから喋ったことない。")])]

    assert fill_gaps(whisper, lines) == []


def test_sound_tags_are_not_speech():
    lines = [(140.83, 143.0, [(140.83, "[叫び声]")]), (171.9, 174.3, [(171.9, "うわ、うわ、[笑い]")])]

    added = fill_gaps(heard((100.0, 100.5, "はい")), lines)

    assert [f.text for f in added] == ["うわ、うわ、"]


def test_a_rolling_window_ends_where_the_next_line_starts():
    """json3 keeps a line up while the next rolls in; the speech is shorter."""
    lines = [(176.64, 180.56, [(176.64, "こいつやば。")]), (178.56, 181.28, [(178.56, "こいつやばいんだけど。")])]

    added = fill_gaps(heard((100.0, 100.5, "はい")), lines)

    assert [(f.start, f.end) for f in added] == [(176.64, 178.56), (178.56, 181.28)]


def test_a_sound_tag_rolling_in_does_not_cut_a_line_short():
    """98.48 s: a [笑い] event 0.43 s later made the line look 0.43 s long, too
    short to overlap Whisper's copy of it at 99.27 s — so it was duplicated."""
    whisper = heard((99.27, 100.4, "ちょっと間があったから、"), (100.45, 102.27, "友達じゃないかと思ったわ。"))
    lines = [(98.479, 100.926, [(98.479, "びっくりした。ちょっと間があったから友達じゃないかと思ったわ。")]),
             (98.906, 100.926, [(98.906, "[笑い]")])]

    assert fill_gaps(whisper, lines) == []


def test_json3_pieces_keep_their_offsets():
    payload = """{"events": [
      {"tStartMs": 0, "dDurationMs": 99999, "id": 1, "wpWinPosId": 1},
      {"tStartMs": 26039, "dDurationMs": 6711, "wWinId": 1, "segs": [
        {"utf8": "うん。", "acAsrConf": 0, "isSpeakerChange": 1},
        {"utf8": "人はね、初めて", "tOffsetMs": 601},
        {"utf8": "VC", "tOffsetMs": 1961}]},
      {"tStartMs": 30000, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\\n"}]}
    ]}"""

    lines = _parse_json3_pieces(payload)

    assert lines == [(26.039, 32.75, [(26.039, "うん。"), (26.64, "人はね、初めて"), (28.0, "VC")])]
    assert [w.start for w in fill_gaps(heard((0.0, 0.5, "え")), lines)[0].words] == [26.039, 26.64, 28.0]


# --- retiming: YouTube's text, Whisper's timing (v-V9NpCvPDk, 7:13-7:20) ----

from youtube_dualsub.pipeline.gapfill import aligned_span, retime, retime_windows  # noqa: E402


def words(*spec: tuple[float, float, str]) -> list[Word]:
    return [Word(start=s, end=e, text=t) for s, e, t in spec]


#: What Whisper heard over 432-442 s with the VAD off.
BARE_714 = words(
    (433.2, 433.5, "な"), (433.5, 433.8, "に"), (433.8, 434.1, "それ!"),
    (434.2, 435.8, "あはははははは"),
    (436.4, 436.8, "な"), (436.8, 437.1, "に"), (437.1, 437.4, "それ。"),
    (438.3, 438.8, "やばい、"), (438.8, 439.3, "つたあきが"), (439.3, 439.7, "ついて"),
    (439.7, 440.2, "きてる!"),
    (440.3, 441.4, "やばいやばい。"),
)


def test_a_line_youtube_closed_early_runs_until_it_was_actually_said():
    """YouTube windowed it 433.68-436.68 s; it was spoken until 440.2 s."""
    line = Fragment(start=433.68, end=436.68, text="何それ?何それ?やばい。ついてきてる。",
                    words=[Word(start=433.68, end=436.68, text="何それ?何それ?やばい。ついてきてる。")])

    assert retime([line], [BARE_714]) == []

    assert 433.0 <= line.start <= 433.8
    assert line.end == 440.2  # not the next line's やばいやばい
    assert (line.words[0].start, line.words[-1].end) == (line.start, line.end)


def test_laughter_decoded_as_words_keeps_youtube_timing():
    """Whisper over laughter writes something else; nothing must move."""
    heard = words((10.0, 11.5, "ありがとうございました"))

    assert aligned_span("ちょっと待って", heard) is None


def test_windows_reach_past_youtube_end():
    line = Fragment(start=1.0, end=4.0, text="x", words=[])

    assert retime_windows([line]) == [(0.0, 9.0)]


# --- words Whisper already has stay Whisper's (d1MzjzSTwvc) -----------------


def test_a_first_word_whisper_heard_seconds_earlier_is_not_repeated():
    """これ was said at 12.72 s; YouTube stamped it with its line's start, 19.40 s,
    and the line opened with it at 0:20 after seven silent seconds."""
    whisper = heard((11.94, 12.40, "思い出した!"), (12.72, 12.80, "これ!"), (21.68, 22.00, "しゃがめ。"))
    lines = [(9.72, 13.16, [(9.72, "ここ"), (11.12, "端"), (11.88, "思い出し")]),
             (19.40, 24.72, [(19.40, "これ"), (20.40, "ちょっと"), (21.04, "やばい"), (21.72, "しゃがめ")])]

    added = fill_gaps(whisper, lines)

    assert [(f.text, f.start, f.end) for f in added] == [("ちょっとやばい", 20.40, 21.72)]


def test_a_first_word_is_kept_when_whisper_did_not_hear_it():
    whisper = heard((84.05, 84.53, "な、"), (89.04, 89.34, "なにこれ?"))
    lines = [(82.12, 86.88, [(82.12, "早く"), (84.48, "な")]),
             (87.28, 91.92, [(87.28, "投げ"), (88.28, "え"), (88.8, "死ん"), (89.04, "だ"), (89.2, "これ")])]

    added = fill_gaps(whisper, lines)

    assert added[-1].text == "投げえ死んだ"  # これ is Whisper's
    assert (added[-1].start, added[-1].end) == (87.28, 89.2)



# --- forced alignment where decoding found nothing (VlaTbwhSSqk) ------------

from youtube_dualsub.pipeline.gapfill import forced_span, retime_forced  # noqa: E402


def test_a_shout_is_moved_onto_its_forced_alignment():
    """YouTube windowed it 178.56-180.56 s; decoding that stretch looped on はぁっ.
    Whisper's main pass, elsewhere, heard こいつ at 178.3 s and けど until 179.4 s."""
    line = Fragment(start=178.56, end=180.56, text="こいつやばいんだけど。",
                    words=[Word(start=178.56, end=180.56, text="こいつやばいんだけど。")])
    aligned = words((177.06, 178.50, "こ"), (178.50, 178.70, "いつ"), (178.70, 178.88, "や"),
                    (178.88, 179.14, "ばい"), (179.14, 179.36, "んだ"), (179.36, 179.68, "けど"),
                    (179.68, 179.80, "。"))

    assert retime_forced([line], [aligned]) == 1

    assert (round(line.start, 2), round(line.end, 2)) == (178.30, 179.56)
    assert (line.words[0].start, line.words[-1].end) == (line.start, line.end)


def test_silence_absorbed_by_the_edge_words_is_trimmed():
    """The first kana held 0.5 s of lead-in, the closing 。 five seconds of quiet."""
    aligned = words((232.18, 232.68, "オ"), (232.68, 232.88, "ッ"), (232.88, 233.72, "ケ"),
                    (233.72, 234.12, "ー"), (234.12, 239.16, "。"))

    start, end = forced_span(233.68, aligned)

    assert round(start, 2) == 232.68  # no more than 1 s before YouTube's start
    assert round(end, 2) == 233.92


def test_an_alignment_piled_onto_the_window_end_keeps_youtube_timing():
    """全員だよな was said at 224.9 s; DTW put four of five letters at 232.62 s."""
    aligned = words((224.42, 232.62, "全"), (232.62, 232.62, "員"), (232.62, 232.62, "だ"),
                    (232.62, 232.62, "よ"), (232.62, 232.62, "な"), (232.62, 232.62, "。"))

    assert forced_span(225.92, aligned) is None
