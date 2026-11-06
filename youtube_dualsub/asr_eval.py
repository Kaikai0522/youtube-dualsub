"""Score the transcript against human captions: the ASR acceptance harness.

    uv run --extra dev dualsub-eval                       # whole set, current settings
    uv run --extra dev dualsub-eval --lang ja --label beam1 --set asr.beam_size=1
    uv run --extra dev dualsub-eval --compare data/eval/results/<old>.json

The videos are listed in ``eval/asr_set.json``, with how each was chosen. The
human captions are downloaded once and frozen under ``data/eval/refs``, so a
caption edit on YouTube cannot move the baseline. The transcript is produced
by the real pipeline with human captions turned off, and cached under the ASR
fingerprint like any run, so re-scoring unchanged settings is free.

What is measured, per language (pooled over its videos):

* **error rate** — CER for Japanese, WER for English, over the whole video's
  text with punctuation, spacing and bracketed notes removed. Human captions
  are edited (fillers dropped, kanji chosen), so the absolute number overstates
  Whisper's errors; compare runs, not against published figures.
* **recall** — human lines with at least half their letters found in the
  transcript within a few seconds of them. A missed line.
* **precision** — transcript lines with at least half their letters found in
  the human captions nearby. A low value means invented text.
* **timing** — for human lines whose first four letters open a transcript line
  nearby, how far apart the two starts are. AGENTS.md: matching a line's first
  words anywhere is biased early (いや, これ recur), so it is the nearest start.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

from .config import DATA_DIR, ROOT, ensure_dirs, load_settings

log = logging.getLogger(__name__)

SET_PATH = ROOT / "eval" / "asr_set.json"
EVAL_DIR = DATA_DIR / "eval"

#: (start, end, text)
Line = tuple[float, float, str]

#: A transcript line may sit this far from the human line it matches. Human
#: caption times are display times and drift a second or two.
_WINDOW_S = 3.0
#: A line counts as found when this share of its letters is matched.
_FOUND = 0.5
#: Starts are compared only when a transcript line opens this close by.
_TIMING_REACH_S = 5.0
_KEY_LETTERS = 4

_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】|<[^>]*>|＜[^＞]*＞")
#: "ぼたん：" or "Phil:" opening a caption line names the speaker.
_SPEAKER = re.compile(r"^\s*[^\s:：\d]{1,12}[:：]\s*")
_NOT_LETTER = re.compile(r"[\W_]+", re.UNICODE)
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_DRAWN_OUT = re.compile(r"([a-z])\1{2,}")


# --------------------------------------------------------------------------
# normalisation and metrics (pure; tests/test_asr_eval.py)
# --------------------------------------------------------------------------


def clean(text: str) -> str:
    """Width-folded, lower case, with notes, speaker labels and invisible marks removed."""
    text = unicodedata.normalize("NFKC", text).replace("​", "").replace("﻿", "")
    text = _SPEAKER.sub("", _BRACKETS.sub(" ", text))
    return text.lower()


def letters(text: str) -> str:
    return _NOT_LETTER.sub("", clean(text))


def words(text: str) -> list[str]:
    # Captioners spell a drawn-out word "nooo", "howww", "longgg" (on
    # SVihhbs64aU, 33 of the first 25 commonest errors); Whisper writes "no".
    text = _DRAWN_OUT.sub(r"\1", clean(text).replace("’", "'"))
    return _WORD.findall(text)


def error_rate(reference: Sequence, hypothesis: Sequence) -> tuple[int, int]:
    """``(edits, reference length)`` — Levenshtein over characters or words."""
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(reference, hypothesis), len(reference)


def _matched(needle: str, haystack: str) -> int:
    if not needle or not haystack:
        return 0
    return sum(
        b.size for b in SequenceMatcher(None, needle, haystack, autojunk=False).get_matching_blocks()
    )


def _nearby(lines: Sequence[Line], start: float, end: float, reach: float) -> str:
    return "".join(letters(t) for s, e, t in lines if s < end + reach and e > start - reach)


def found_lines(lines: Sequence[Line], other: Sequence[Line]) -> tuple[int, int]:
    """``(found, counted)``: lines of ``lines`` whose letters ``other`` mostly has nearby."""
    found = counted = 0
    for start, end, text in lines:
        wanted = letters(text)
        if len(wanted) < 2:
            continue
        counted += 1
        if _matched(wanted, _nearby(other, start, end, _WINDOW_S)) >= _FOUND * len(wanted):
            found += 1
    return found, counted


def start_offsets(reference: Sequence[Line], hypothesis: Sequence[Line]) -> list[float]:
    """Transcript start minus human start, for lines whose opening both agree on."""
    opened = [(s, letters(t)) for s, _e, t in hypothesis]
    out: list[float] = []
    for start, _end, text in reference:
        key = letters(text)[:_KEY_LETTERS]
        if len(key) < _KEY_LETTERS:
            continue
        near = [
            s for s, lt in opened
            if abs(s - start) <= _TIMING_REACH_S and lt.startswith(key)
        ]
        if near:
            out.append(min(near, key=lambda s: abs(s - start)) - start)
    return out


@dataclass
class Score:
    edits: int = 0
    length: int = 0
    recall: tuple[int, int] = (0, 0)
    precision: tuple[int, int] = (0, 0)
    offsets: list[float] = field(default_factory=list)
    audio_s: float = 0.0

    def add(self, other: "Score") -> None:
        self.edits += other.edits
        self.length += other.length
        self.recall = (self.recall[0] + other.recall[0], self.recall[1] + other.recall[1])
        self.precision = (
            self.precision[0] + other.precision[0],
            self.precision[1] + other.precision[1],
        )
        self.offsets += other.offsets
        self.audio_s += other.audio_s

    def summary(self) -> dict:
        offs = [abs(o) for o in self.offsets]
        return {
            "error_rate": self.edits / self.length if self.length else None,
            "recall": self.recall[0] / self.recall[1] if self.recall[1] else None,
            "precision": self.precision[0] / self.precision[1] if self.precision[1] else None,
            "timing_n": len(offs),
            "timing_median_s": statistics.median(offs) if offs else None,
            "timing_over_1s": sum(o > 1.0 for o in offs) / len(offs) if offs else None,
            "timing_bias_s": statistics.median(self.offsets) if offs else None,
            "audio_min": self.audio_s / 60,
        }


def score(reference: Sequence[Line], hypothesis: Sequence[Line], language: str) -> Score:
    ref_text = " ".join(t for _s, _e, t in reference)
    hyp_text = " ".join(t for _s, _e, t in hypothesis)
    if language == "ja":
        edits, length = error_rate(letters(ref_text), letters(hyp_text))
    else:
        edits, length = error_rate(words(ref_text), words(hyp_text))
    return Score(
        edits=edits,
        length=length,
        recall=found_lines(reference, hypothesis),
        precision=found_lines(hypothesis, reference),
        offsets=start_offsets(reference, hypothesis),
    )


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def load_set(path: Path = SET_PATH) -> list[dict]:
    return json.loads(path.read_text("utf-8"))["videos"]


def reference(video_id: str, language: str) -> dict:
    """The human captions, downloaded once and frozen under data/eval/refs."""
    path = EVAL_DIR / "refs" / f"{video_id}.{language}.json"
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    from .sources import ytdlp_source

    info = ytdlp_source.probe(video_id)
    found = ytdlp_source.fetch_manual_subtitles(info, language)
    if not found or found[0] != language:
        raise RuntimeError(f"{video_id} has no human {language} captions any more")
    data = {
        "id": video_id,
        "language": language,
        "title": info.title,
        "duration_s": info.duration_s,
        "fetched": time.strftime("%Y-%m-%d"),
        "cues": [list(c) for c in found[1]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data


def transcribe(video_id: str, overrides: dict) -> list[Line]:
    from .pipeline.orchestrator import Orchestrator
    from .store import Store

    settings = load_settings(
        _deep_merge(overrides, {"audio": {"prefer_manual_captions": False}})
    )
    started = time.time()
    result = Orchestrator(settings, Store()).transcript(video_id)
    log.info("%s: %d lines in %.0f s", video_id, len(result.sentences), time.time() - started)
    return [(s.start, s.end, s.text) for s in result.sentences]


def run(
    videos: Iterable[dict], overrides: dict
) -> tuple[dict[str, dict], dict[str, Score]]:
    per_video: dict[str, dict] = {}
    pooled: dict[str, Score] = {}
    for video in videos:
        vid, language = video["id"], video["lang"]
        ref = reference(vid, language)
        cues = [tuple(c) for c in ref["cues"]]
        hyp = transcribe(vid, overrides)
        s = score(cues, hyp, language)
        s.audio_s = ref["duration_s"]
        per_video[vid] = {"lang": language, **s.summary()}
        pooled.setdefault(language, Score()).add(s)
    return per_video, pooled


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_set(pairs: Sequence[str]) -> dict:
    """``["asr.beam_size=1", "asr.language=\"ja\""]`` -> nested overrides."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


def _fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "%":
        return f"{value * 100:5.1f}%"
    if kind == "s":
        return f"{value:.2f}s"
    return str(value)


_COLUMNS = (
    ("error_rate", "CER/WER", "%"),
    ("recall", "recall", "%"),
    ("precision", "precision", "%"),
    ("timing_median_s", "|Δstart|", "s"),
    ("timing_over_1s", ">1s", "%"),
    ("timing_bias_s", "bias", "s"),
    ("timing_n", "n", ""),
)


def print_table(rows: dict[str, dict], baseline: dict[str, dict] | None = None, out=sys.stdout) -> None:
    header = f"{'':14s}" + "".join(f"{title:>11s}" for _k, title, _f in _COLUMNS)
    print(header, file=out)
    for name, row in rows.items():
        cells = "".join(f"{_fmt(row.get(k), f):>11s}" for k, _t, f in _COLUMNS)
        print(f"{name:14s}{cells}", file=out)
        old = (baseline or {}).get(name)
        if old:
            delta = []
            for k, _t, f in _COLUMNS:
                a, b = row.get(k), old.get(k)
                if a is None or b is None or k == "timing_n":
                    delta.append(f"{'':>11s}")
                elif f == "%":
                    delta.append(f"{(a - b) * 100:+10.1f} ")
                else:
                    delta.append(f"{a - b:+10.2f} ")
            print(f"{'  vs baseline':14s}" + "".join(delta), file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dualsub-eval", description=__doc__.split("\n")[0])
    p.add_argument("--lang", choices=("ja", "en"), help="only this language's videos")
    p.add_argument("--only", help="comma-separated video ids from the set")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help='setting override, e.g. asr.beam_size=1 (repeatable)')
    p.add_argument("--label", default="", help="name for the saved result file")
    p.add_argument("--compare", type=Path, help="an earlier result file to diff against")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ensure_dirs()

    videos = load_set()
    if args.lang:
        videos = [v for v in videos if v["lang"] == args.lang]
    if args.only:
        wanted = set(args.only.split(","))
        videos = [v for v in videos if v["id"] in wanted]
    overrides = parse_set(args.set)

    started = time.time()
    per_video, pooled = run(videos, overrides)
    rows = {**per_video, **{f"ALL {k}": v.summary() for k, v in pooled.items()}}

    baseline = None
    if args.compare:
        baseline = json.loads(args.compare.read_text("utf-8"))["rows"]
    print()
    print_table(rows, baseline)

    settings = load_settings(_deep_merge(overrides, {"audio": {"prefer_manual_captions": False}}))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = EVAL_DIR / "results" / f"{stamp}{'-' + args.label if args.label else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": args.label,
                "overrides": overrides,
                "asr_fingerprint": settings.asr_fingerprint,
                "elapsed_s": time.time() - started,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
