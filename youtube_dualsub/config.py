"""Every tunable parameter in the project lives here.

Nothing else in the codebase should hard-code a threshold, a model name or a
path. Override anything by dropping a ``config.local.json`` next to
``pyproject.toml`` with the same nested shape, e.g.::

    {"translate": {"model": "qwen3:14b"}, "asr": {"language": "ja"}}
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
MODEL_DIR = DATA_DIR / "models"          # downloaded ONNX models (diarization)
GLOSSARY_DIR = Path(__file__).resolve().parent / "glossaries"
LOCAL_CONFIG = ROOT / "config.local.json"

#: Bump when transcript *ingestion* changes in a way settings do not capture —
#: a caption parser fix, a change to hallucination filtering, and so on. It is
#: part of the ASR fingerprint, so bumping it retires every cached transcript.
#: v2: collapse YouTube's rolling caption repeats.
#: v3: language-aware ingestion (Japanese), caption lines kept as cues,
#:     scored sentence segmentation, batched Whisper with a context prompt.
#: v4: timestamps snapped to speech, stranded kana re-attached, short
#:     repeated reactions kept, zero-length lines dropped.
#: v5: Japanese sentences break at Whisper's spaces and segment ends, speaker
#:     changes move to the nearest breakable gap, "V C" pieces stay one word.
#: v6: stranded prefixes re-attached across digits and Latin too, and an
#:     over-long first token cut back before re-attaching.
#: v7: YouTube gap-fill lines drop edge words Whisper already has.
#: v8: gap-fill lines Whisper's decode cannot match are force-aligned.
INGEST_VERSION = 8

#: Bump when the rules that accept or reject a wiki answer for a term change
#: (``terms.py``). It keys the lookup cache, the cached per-video context and
#: the translations built on it, so a fixed rule is not ignored in favour of
#: an answer cached under the old one.
TERMS_VERSION = 3


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
    # zero GPU seconds. Auto-generated tracks never replace Whisper (decision
    # Q2); they only fill its gaps (asr.youtube_gap_fill).
    prefer_manual_captions: bool = True
    max_duration_s: int = 3 * 60 * 60  # hard ceiling: 3 hours (decision Q15)


@dataclass(frozen=True)
class VocalsConfig:
    """Optional Demucs vocal isolation (``uv sync --extra vocals``).

    Off by default: it needs a 3 GB torch install, and the transcript's
    fingerprint used to claim "vocals on" on machines where Demucs was missing
    and the stage had quietly degraded to the raw mix.
    """

    enabled: bool = False
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
    #: None = use the language YouTube reports for the video, and let Whisper
    #: detect it when YouTube does not know. Forcing "en" on a Chinese video
    #: made Whisper *translate* it into English instead of transcribing it.
    language: str | None = None
    beam_size: int = 5
    word_timestamps: bool = True
    # CRITICAL for noisy multi-speaker content: with this on, one hallucination
    # seeds the next window and the damage cascades through the whole video.
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_min_silence_ms: int = 400
    vad_speech_pad_ms: int = 200
    # Silero's default is 0.5. At 0.35 the batched decoder is handed quieter
    # speech: missing caption lines on a two-speaker clip went 10 -> 7 of 48.
    vad_threshold: float = 0.35
    # Clamp word timestamps to the audio that has speech (asr.snap_to_speech).
    snap_to_speech: bool = True
    # Batched inference decodes many VAD chunks at once. Measured on a 196 s
    # Japanese multi-speaker clip: 22.5 s sequential, 4.1 s batched. Batched
    # mode only uses the first temperature, so a chunk that comes back looping
    # is re-decoded sequentially with the full fallback ladder below.
    batched: bool = True
    batch_size: int = 16
    # The names in the title, as the batched decoder's prompt. Lowered CER in
    # both languages and tripled Japanese punctuation (see pipeline/asr.py).
    context_prompt: bool = True
    # Add YouTube's auto-caption lines where Whisper wrote nothing — shouted
    # speech, mostly, which the VAD never hands to Whisper. pipeline/gapfill.py
    # has the measurements.
    youtube_gap_fill: bool = True
    # Keep those lines' text but take their timing from Whisper, decoded over
    # the line's stretch without the VAD; YouTube's windows are display times,
    # not speech times (gapfill.retime).
    youtube_gap_retime: bool = True
    # Whisper falls back through these temperatures when a window looks degenerate.
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass(frozen=True)
class DiarizationConfig:
    """Who is speaking: sherpa-onnx on the CPU (pipeline/diarize.py).

    Speakers are never shown. A change of voice only tells the sentence
    builder where one person's line ends and the next begins.
    """

    enabled: bool = True
    embedding_model: str = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    # CAM++ (zh/en) was measured against ERes2NetV2, TitaNet and WeSpeaker on
    # three Japanese clips: best agreement with reference speaker labels and
    # the fastest (5 s for a 7.5-minute clip). WeSpeaker merged everyone into
    # one voice.
    # Higher merges more voices into one speaker; lower splits one voice into
    # several. At 0.9, with near-silent clusters dropped, the two- and
    # three-person clips came out as two and three speakers.
    cluster_threshold: float = 0.9
    # Set when you know the count; 0 = decide from the threshold.
    num_speakers: int = 0
    # 8 threads measured no faster than 4.
    num_threads: int = 4


@dataclass(frozen=True)
class SentenceConfig:
    """Rebuilding Whisper's word stream into subtitle-sized sentences.

    Lengths are in *visible* characters (a kanji counts as two), so one set of
    numbers serves English and Japanese. See ``pipeline/sentences.py``.
    """

    # A sentence this long reads comfortably as one or two subtitle lines.
    target_chars: int = 84
    target_s: float = 6.0
    # Never glue more than this into one translation unit.
    max_chars: int = 200
    max_sentence_s: float = 12.0
    # A pause this long always ends a sentence, whatever the grammar says.
    hard_pause_s: float = 2.5
    # Human captions are already cut by a human; a line is only split when it
    # runs longer than this.
    caption_max_s: float = 10.0


@dataclass(frozen=True)
class ContextConfig:
    """Whole-video summary + automatic terminology (Q22).

    Terminology is never hand-written. The LLM proposes the names and jargon it
    finds; each one is then looked up in Wikipedia (and a domain wiki when the
    video's subject has one) for its Traditional Chinese title. Measured against
    the old hand-made Minecraft glossary: gemma4:12b alone matched 55/99 terms
    (下界 for Nether, 苦力機 for creeper); the wiki lookup matched 76/99 in
    1.8 s, and most of its "misses" were the old glossary being wrong.
    """

    enabled: bool = True
    model: str = ""  # empty = reuse TranslateConfig.model
    # A transcript this long or shorter is read in one request; longer ones are
    # summarised in chunks and merged.
    single_pass_chars: int = 24000
    chunk_sentences: int = 150
    max_chunks: int = 24  # 3h video -> summaries stay bounded
    max_auto_terms: int = 80
    num_ctx: int = 16384
    # Look terms up online. Off = rely on the model's own knowledge.
    wiki_lookup: bool = True
    wiki_timeout_s: float = 8.0
    # MediaWiki sites that know a subject better than Wikipedia does, as
    # (keyword in the detected subject, English wiki API, Chinese wiki API).
    # Any wiki whose pages link to a Chinese edition with zh-tw variant
    # conversion works; Wikipedia itself is always consulted.
    domain_wikis: tuple[tuple[str, str, str], ...] = (
        ("minecraft", "https://minecraft.wiki/api.php", "https://zh.minecraft.wiki/api.php"),
    )


@dataclass(frozen=True)
class TranslateConfig:
    # gemma4:12b translates Japanese and English better than qwen3:14b. It once
    # measured 57-75 s per batch, but that was its reasoning running unchecked;
    # with `think` off it answers in seconds.
    model: str = "gemma4:12b"
    # Subtitle translation has nothing to reason about. Leaving this on cost
    # gemma4 up to 4000 thinking tokens for a three-word line.
    think: bool = False
    # Batch size barely matters for speed (96 lines: 0.175 s/line at 8 and at
    # 32); generation is the bottleneck. Concurrent requests are what help:
    # Ollama decodes them together, 1.5x faster at 2 and 1.8x at 4, with no
    # extra VRAM. Needs OLLAMA_NUM_PARALLEL >= this on the server (start.ps1
    # sets it); otherwise the extra requests just queue. On v-V9NpCvPDk (736
    # lines, 2026-10-28): 71.6 s at 4, 64.3 s at 6, 62.5 s at 8, 60.0 s at 8
    # with num_ctx 4096, 55.8 s with an 8-bit KV cache (OLLAMA_KV_CACHE_TYPE,
    # also set by start.ps1). The GPU is close to saturated, so more will not help.
    batch_size: int = 16
    parallel: int = 8

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
    # Sliding window: how many previous pairs to show as context (Q22), and how
    # many upcoming source lines. Look-ahead is what lets a caption line that
    # stops mid-sentence be translated as the first half of that sentence.
    context_pairs: int = 6
    lookahead_lines: int = 3
    temperature: float = 0.3
    # Per request slot; Ollama reserves num_ctx x OLLAMA_NUM_PARALLEL. The
    # largest translation prompt measured was 982 tokens plus a ~1100-token
    # answer budget for 16 lines, so 4096 leaves room.
    num_ctx: int = 4096
    # Bump when the prompt changes so cached translations are invalidated.
    prompt_version: str = "v2"
    target_language: str = "繁體中文（台灣）"
    style: str = "natural"  # "natural" (Q27) | "literal"
    request_timeout_s: float = 180.0


@dataclass(frozen=True)
class PostprocessConfig:
    opencc_enabled: bool = True  # A/B switch: does gemma4 actually emit 简体? (Q9)
    opencc_config: str = "s2twp"  # simplified -> Taiwan traditional, with phrases
    # Optional overrides only. Terminology is extracted automatically per video;
    # this file exists for the rare name you want rendered your own way.
    glossary_files: tuple[str, ...] = ("user.yaml",)
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
    max_chars_en: int = 84          # source line in a spaced script
    max_chars_ja: int = 30          # source line in Japanese, full-width chars
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


@dataclass(frozen=True)
class Settings:
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vocals: VocalsConfig = field(default_factory=VocalsConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
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
        s, a = self.sentences, self.asr
        return "|".join(
            [
                f"v{INGEST_VERSION}",
                a.model,
                a.compute_type,
                str(a.language or "auto"),
                f"vad{int(a.vad_filter)}",
                f"batch{int(a.batched)}",
                f"prompt{int(a.context_prompt)}",
                f"snap{int(a.snap_to_speech)}",
                f"ytfill{int(a.youtube_gap_fill)}",
                f"retime{int(a.youtube_gap_retime)}",
                f"vadt{a.vad_threshold}",
                f"vocals{int(self.vocals.enabled)}",
                self._diarization_key(),
                f"manual{int(self.audio.prefer_manual_captions)}",
                f"sent{s.target_chars},{s.target_s},{s.max_chars},"
                f"{s.max_sentence_s},{s.hard_pause_s},{s.caption_max_s}",
            ]
        )

    def _diarization_key(self) -> str:
        d = self.diarization
        if not d.enabled:
            return "diar0"
        return f"diar{d.embedding_model},{d.cluster_threshold},{d.num_speakers}"

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
                f"wiki{int(self.context.wiki_lookup)}r{TERMS_VERSION}",
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
