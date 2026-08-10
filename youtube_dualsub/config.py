"""Every tunable parameter in the project lives here.

Nothing else in the codebase should hard-code a threshold, a model name or a
path. Override anything by dropping a ``config.local.json`` next to
``pyproject.toml`` with the same nested shape, e.g.::

    {"translate": {"model": "qwen3:14b"}, "vocals": {"enabled": false}}
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DUALSUB_DATA_DIR") or (ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"          # downloaded audio, isolated vocals
DB_PATH = DATA_DIR / "dualsub.sqlite3"
EXPORT_DIR = DATA_DIR / "exports"
GLOSSARY_DIR = Path(__file__).resolve().parent / "glossaries"
LOCAL_CONFIG = ROOT / "config.local.json"

#: Bump when transcript *ingestion* changes in a way settings do not capture —
#: a caption parser fix, a change to hallucination filtering, and so on. It is
#: part of the ASR fingerprint, so bumping it retires every cached transcript.
#: v2: collapse YouTube's rolling caption repeats.
INGEST_VERSION = 2


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8756
    # Seconds a job may sit with no connected client before it pauses itself.
    # Closing the tab pauses the job (decision Q24); it does not run to completion.
    orphan_grace_s: float = 20.0


@dataclass(frozen=True)
class AudioConfig:
    """Acquisition. Only ``pipeline.audio.get_audio`` knows about any of this."""

    source: str = "ytdlp"  # "ytdlp" | "mse" (mse arrives in Phase 5)
    # Preferred yt-dlp format chain. The second entry is the SABR-resistant
    # fallback: a muxed HLS stream we strip the audio out of with ffmpeg.
    format_chain: tuple[str, ...] = (
        "bestaudio[ext=m4a]/bestaudio",
        "best[protocol^=m3u8]/best",
    )
    sample_rate: int = 16000  # what Whisper wants; resampled by ffmpeg
    cookies_from_browser: str | None = None  # e.g. "chrome" for age-gated videos
    # Human-uploaded captions beat anything we can transcribe locally and cost
    # zero GPU seconds. Auto-generated tracks are never used (decision Q2).
    prefer_manual_captions: bool = True
    max_duration_s: int = 3 * 60 * 60  # hard ceiling: 3 hours (decision Q15)


@dataclass(frozen=True)
class VocalsConfig:
    """Demucs vocal isolation — on by default, A/B-tested during acceptance (Q30)."""

    enabled: bool = True
    model: str = "htdemucs"
    device: str = "cuda"
    # Demucs peaks around 4 GB; the stage releases it before ASR loads.
    segment_s: int = 30
    # If demucs is not installed or crashes we degrade to raw audio (Q19).
    required: bool = False


@dataclass(frozen=True)
class AsrConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "en"  # None = autodetect (Q5: other languages later)
    beam_size: int = 5
    word_timestamps: bool = True
    # CRITICAL for noisy multi-speaker content: with this on, one hallucination
    # seeds the next window and the damage cascades through the whole video.
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_min_silence_ms: int = 400
    vad_speech_pad_ms: int = 200
    # Whisper falls back through these temperatures when a window looks degenerate.
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass(frozen=True)
class SentenceConfig:
    """Rebuilding ASR fragments into whole semantic sentences."""

    # A pause longer than this ends a sentence even without punctuation.
    pause_break_s: float = 0.7
    # Never glue more than this much audio into one sentence.
    max_sentence_s: float = 12.0
    max_sentence_chars: int = 240
    sentence_end_chars: str = ".!?"
    # A fragment shorter than this is always glued to its neighbour.
    min_fragment_chars: int = 3


@dataclass(frozen=True)
class ContextConfig:
    """Map-reduce whole-video summary + terminology extraction (Q22)."""

    enabled: bool = True
    model: str = ""  # empty = reuse TranslateConfig.model
    chunk_sentences: int = 120
    max_chunks: int = 24  # 3h video -> summaries stay bounded
    max_auto_terms: int = 60


@dataclass(frozen=True)
class TranslateConfig:
    # Measured on this project's acceptance video, 12 hard single lines plus
    # three batches of ten:
    #   qwen3:14b, reasoning off  12/12 and 3/3, ~2s per batch of ten
    #   gemma4:12b               11/12 and 1/3, 57-75s per batch
    # Both are reasoning models; the difference is almost entirely that Gemma
    # cannot be talked out of thinking for thousands of tokens per line.
    model: str = "qwen3:14b"
    # Subtitle translation has nothing to reason about. Leaving this on cost
    # gemma4 up to 4000 thinking tokens for a three-word line.
    think: bool = False
    batch_size: int = 10
    # The ladder must be able to reach one line: measured on this video, every
    # sentence translates correctly on its own, including the ones that make a
    # batch of ten loop forever. Single-line translation is the slow floor that
    # guarantees the job finishes.
    min_batch_size: int = 2
    max_retries: int = 2
    # Client-side generation budget (see llm._stream_bounded). Ten lines of this
    # material need ~250 tokens; anything past this many is the model looping,
    # not the model working.
    max_tokens_per_line: int = 60
    max_tokens_overhead: int = 150
    # Sliding window: how many previous sentence pairs to show as context (Q22).
    context_pairs: int = 6
    temperature: float = 0.3
    num_ctx: int = 8192
    # Bump when the prompt changes so cached translations are invalidated.
    prompt_version: str = "v1"
    target_language: str = "繁體中文（台灣）"
    style: str = "natural"  # "natural" (Q27) | "literal"
    request_timeout_s: float = 180.0


@dataclass(frozen=True)
class PostprocessConfig:
    opencc_enabled: bool = True  # A/B switch: does gemma4 actually emit 简体? (Q9)
    opencc_config: str = "s2twp"  # simplified -> Taiwan traditional, with phrases
    glossary_files: tuple[str, ...] = ("minecraft_zh_tw.yaml", "user.yaml")
    # Auto-extracted terms are per-video only; never carried across videos (Q28).
    use_auto_terms: bool = True


@dataclass(frozen=True)
class ShapeConfig:
    """Cue shaping. See the module docstring of pipeline/shape.py for the rules."""

    merge_below_s: float = 1.2      # anti-flicker is done by MERGING, not by holding
    # A short cue only absorbs its neighbour across a gap this small. Without
    # the ceiling, a run of stubs keeps absorbing forward across silences and
    # ends up showing text seconds before it is spoken.
    merge_max_gap_s: float = 0.6
    max_chars_zh: int = 20          # full-width chars per Chinese line
    max_chars_en: int = 84
    max_duration_s: float = 7.0
    min_duration_s: float = 1.0     # soft floor; never crosses the next cue (Q29)
    # Gap left between a cue and the next one when they would otherwise touch.
    cue_gap_s: float = 0.0


@dataclass(frozen=True)
class SubtitleStyleConfig:
    """Defaults for the browser overlay and the .ass export."""

    zh_on_top: bool = True   # decision Q11(b)
    font_size_zh: int = 30
    font_size_en: int = 22
    show_speaker_prefix: bool = True  # inert until diarization lands (Q26)


@dataclass(frozen=True)
class Settings:
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vocals: VocalsConfig = field(default_factory=VocalsConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    sentences: SentenceConfig = field(default_factory=SentenceConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    style: SubtitleStyleConfig = field(default_factory=SubtitleStyleConfig)

    # --- cache keys -----------------------------------------------------
    @property
    def asr_fingerprint(self) -> str:
        """Changing any of these invalidates the cached transcript.

        This covers everything that shapes the stored sentences, not just the
        ASR model — the sentence-building thresholds and the caption source
        belong here too. Leaving them out meant a fix to caption ingestion or a
        change to the sentence limits was silently ignored in favour of stale
        rows, which wasted a debugging session twice.
        """
        s = self.sentences
        return "|".join(
            [
                f"v{INGEST_VERSION}",
                self.asr.model,
                self.asr.compute_type,
                str(self.asr.language),
                f"vad{int(self.asr.vad_filter)}",
                f"vocals{int(self.vocals.enabled)}",
                f"manual{int(self.audio.prefer_manual_captions)}",
                f"sent{s.pause_break_s},{s.max_sentence_s},"
                f"{s.max_sentence_chars},{s.min_fragment_chars}",
            ]
        )

    @property
    def translation_fingerprint(self) -> str:
        """Changing any of these invalidates cached translations but not ASR."""
        return "|".join(
            [
                self.translate.model,
                self.translate.prompt_version,
                self.translate.style,
                self.translate.target_language,
                f"ctx{int(self.context.enabled)}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(instance: Any, overrides: dict[str, Any]) -> Any:
    """Rebuild a frozen dataclass with nested overrides applied."""
    kwargs: dict[str, Any] = {}
    for f in fields(instance):
        current = getattr(instance, f.name)
        if f.name not in overrides:
            kwargs[f.name] = current
        elif is_dataclass(current) and isinstance(overrides[f.name], dict):
            kwargs[f.name] = _merge(current, overrides[f.name])
        elif isinstance(current, tuple) and isinstance(overrides[f.name], list):
            kwargs[f.name] = tuple(overrides[f.name])
        else:
            kwargs[f.name] = overrides[f.name]
    return type(instance)(**kwargs)


def load_settings(
    overrides: dict[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> Settings:
    """Defaults, then ``config.local.json``, then explicit overrides.

    ``config_path`` is a parameter rather than a module lookup so that whoever
    writes the file and whoever re-reads it are provably talking about the same
    one — they were not, once.
    """
    path = config_path or LOCAL_CONFIG
    settings = Settings()
    if path.is_file():
        settings = _merge(settings, json.loads(path.read_text("utf-8")))
    if overrides:
        settings = _merge(settings, overrides)
    return settings


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
