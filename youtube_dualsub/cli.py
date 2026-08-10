"""Command line entry point — the acceptance harness for the whole pipeline.

Deliberately usable without the browser extension: translation quality is the
entire value of this project, and it can be judged from a .srt in a text editor
far faster than from an overlay. Get this producing good subtitles first.

    uv run dualsub IqcS1d3eXYc --export srt
    uv run dualsub IqcS1d3eXYc --no-vocals --retranslate     # A/B the Demucs switch
    uv run dualsub IqcS1d3eXYc --model qwen3:14b --retranslate
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time

from .config import Settings, ensure_dirs, load_settings
from .models import Stage, TranslationStatus
from .pipeline.orchestrator import Cancelled, JobResult, Orchestrator, export_result
from .pipeline.postprocess import simplified_ratio
from .store import Store

_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def parse_video_id(value: str) -> str:
    match = _ID.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    raise argparse.ArgumentTypeError(f"could not find a YouTube video id in {value!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dualsub",
        description="Generate bilingual (source + Traditional Chinese) subtitles locally.",
    )
    p.add_argument("video", type=parse_video_id, help="YouTube video id or URL")
    p.add_argument(
        "--export", default="srt",
        help="comma-separated formats to write: srt, ass, or 'none' (default: srt)",
    )
    p.add_argument("--model", help="override the translation model, e.g. qwen3:14b")
    p.add_argument("--asr-model", help="override the Whisper model, e.g. large-v3-turbo")
    p.add_argument("--language", help="source language code, or 'auto' to detect")
    p.add_argument("--no-vocals", action="store_true", help="skip Demucs vocal isolation")
    p.add_argument("--vocals", action="store_true", help="force Demucs vocal isolation on")
    p.add_argument("--no-opencc", action="store_true", help="skip Simplified->Traditional cleanup")
    p.add_argument("--no-context", action="store_true", help="skip the whole-video summary pass")
    p.add_argument("--no-manual-captions", action="store_true",
                   help="always transcribe, even when human captions exist")
    p.add_argument("--retranslate", action="store_true",
                   help="discard cached translations but keep the transcript")
    p.add_argument("--reset", action="store_true", help="discard everything for this video")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if args.model:
        out.setdefault("translate", {})["model"] = args.model
    if args.asr_model:
        out.setdefault("asr", {})["model"] = args.asr_model
    if args.language:
        out.setdefault("asr", {})["language"] = None if args.language == "auto" else args.language
    if args.no_vocals:
        out.setdefault("vocals", {})["enabled"] = False
    if args.vocals:
        out.setdefault("vocals", {})["enabled"] = True
    if args.no_opencc:
        out.setdefault("postprocess", {})["opencc_enabled"] = False
    if args.no_context:
        out.setdefault("context", {})["enabled"] = False
    if args.no_manual_captions:
        out.setdefault("audio", {})["prefer_manual_captions"] = False
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # yt-dlp and httpx are chatty at INFO and drown out the progress line.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ensure_dirs()
    settings = load_settings(_overrides(args))
    store = Store()

    if args.reset:
        store.clear_video(args.video)
        print(f"Cleared all cached data for {args.video}.")

    printer = _ProgressPrinter()
    try:
        result = Orchestrator(settings, store).run(
            args.video,
            on_progress=printer,
            force_retranslate=args.retranslate,
        )
    except Cancelled:
        print("\nPaused.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        printer.finish()
        print(f"\nFailed: {exc}", file=sys.stderr)
        return 1
    printer.finish()

    formats = [f.strip() for f in args.export.split(",") if f.strip() and f.strip() != "none"]
    written = export_result(result, settings, formats) if formats else []

    _report(result, settings, written)
    return 0


class _ProgressPrinter:
    """One rewriting line on stderr; a 52-minute job should not scroll the terminal."""

    def __init__(self) -> None:
        self._last_len = 0
        self._stage: Stage | None = None
        self._started = time.time()

    def __call__(self, progress) -> None:
        if progress.stage is not self._stage:
            if self._stage is not None:
                self._write("")
                print(file=sys.stderr)
            self._stage = progress.stage
        pct = f" {progress.fraction * 100:3.0f}%" if progress.fraction is not None else "     "
        elapsed = time.time() - self._started
        self._write(f"[{elapsed:5.0f}s] {progress.stage.value:<10}{pct}  {progress.message}")

    def _write(self, line: str) -> None:
        sys.stderr.write("\r" + line.ljust(self._last_len))
        sys.stderr.flush()
        self._last_len = max(len(line), 0)

    def finish(self) -> None:
        if self._stage is not None:
            print(file=sys.stderr)


def _report(result: JobResult, settings: Settings, written: list[str]) -> None:
    zh = [c.target for c in result.cues if c.target]
    fallbacks = sum(
        1 for t in result.translations if t.status is TranslationStatus.SOURCE_FALLBACK
    )
    ratio = simplified_ratio(zh)

    print()
    print(f"  {result.title}")
    print(f"  {result.duration_s / 60:.0f} min video processed in {result.elapsed_s / 60:.1f} min")
    speed = result.duration_s / result.elapsed_s if result.elapsed_s else 0
    print(f"  {speed:.1f}x faster than watching it")
    print()
    print(f"  transcript source   {'human-uploaded captions' if result.used_manual_captions else 'Whisper ' + settings.asr.model}")
    print(f"  vocal isolation     {'on' if settings.vocals.enabled else 'off'}")
    print(f"  translation model   {settings.translate.model}")
    print(f"  OpenCC {settings.postprocess.opencc_config:<12} {'on' if settings.postprocess.opencc_enabled else 'off'}")
    print()
    print(f"  sentences           {len(result.sentences)}")
    print(f"  subtitles           {len(result.cues)}")
    print(f"  glossary terms      {len(result.context.terms)} auto-extracted")
    if fallbacks:
        print(f"  left in English     {fallbacks}  (translation could not be aligned)")
    if zh:
        verdict = "clean" if ratio == 0 else f"{ratio * 100:.1f}% of lines - check these"
        print(f"  Simplified probe    {verdict}")
    if written:
        print()
        for path in written:
            print(f"  wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
