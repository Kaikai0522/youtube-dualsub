"""Speech recognition with faster-whisper.

Settings here that are load-bearing for noisy multi-speaker content and should
not be "tidied up":

``condition_on_previous_text=False``
    With it on, Whisper feeds its own last output back in as a prompt. One
    hallucination during a music sting then seeds the next window, and the
    damage cascades for minutes. Off costs a little coherence and buys back
    the whole video.

``vad_filter=True``
    Not optional. Whisper invents speech in silence and under music — the
    classic "Thanks for watching!" (ご視聴ありがとうございました) over an outro.
    The VAD never shows it those regions in the first place.

``word_timestamps=True``
    Sentence rebuilding needs pause lengths, and cue shaping needs to split a
    long sentence at a real word boundary.

Batched inference plus repair
    ``BatchedInferencePipeline`` decodes many VAD chunks in parallel, five times
    faster on a real clip, but it only ever uses the first temperature. A chunk
    that falls into a loop ("もう1個探せよ" twenty-seven times) is therefore
    detected and re-decoded on its own with the sequential decoder's full
    temperature fallback.

The context prompt
    The names in the video's title go to the batched decoder as its
    ``initial_prompt``, after one punctuated sentence in the spoken language.
    Measured against human captions it lowered the error rate in both
    languages (English 2.0% -> 1.5%, Japanese 24.3% -> 21.8%) and nearly
    tripled the punctuation Whisper writes in Japanese, which sentence
    segmentation depends on. The prompt is fixed text, not Whisper's own
    output, so it cannot cascade the way ``condition_on_previous_text`` does.
    It is kept short and clean: a long list of a clip channel's hashtags
    (精華, 翻譯, 中文) sent the decoder into loops.
"""

from __future__ import annotations

import bisect
import gc

import logging
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .._cuda import ensure_cuda_dlls
from ..config import Settings
from ..lang import is_cjk, is_cjk_char

from ..models import Fragment, Word

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

_SAMPLE_RATE = 16000

#: Phrases Whisper emits when it has nothing to transcribe. Only ever dropped
#: when the model itself also reports a high no-speech probability, or away
#: from the end of the video where a real sign-off could not be.
_FILLER_HALLUCINATIONS = {
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching!",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "you",
    "bye",
    "♪",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "ご清聴ありがとうございました",
    "ご清聴ありがとうございました。",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録よろしくお願いします。",
    "おやすみなさい",
    "おやすみなさい。",
}
#: Subtitle credits Whisper learned from fan-subbed training data.
_CREDIT = re.compile(
    r"(字幕|subtitles?)\s*(は|by|作成|提供|制作)|ご覧の方によって|amara\.org", re.IGNORECASE
)


class AsrFailed(RuntimeError):
    pass


@dataclass(slots=True)
class Transcription:
    fragments: list[Fragment] = field(default_factory=list)
    language: str | None = None


def transcribe(
    audio_path: Path,
    settings: Settings,
    *,
    language: str | None = None,
    prompt: str | None = None,
    duration_s: float | None = None,
    progress: ProgressCb | None = None,
) -> Transcription:
    ensure_cuda_dlls()
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
    except ImportError as exc:  # pragma: no cover
        raise AsrFailed("faster-whisper is not installed. Run: uv sync") from exc

    cfg = settings.asr
    if progress:
        progress(None, f"Loading Whisper {cfg.model}")

    try:
        model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except Exception as exc:  # noqa: BLE001
        raise AsrFailed(_explain_load_failure(exc)) from exc

    prompt = prompt if cfg.context_prompt else None
    fragments: list[Fragment] = []
    try:
        audio = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
        total = duration_s or len(audio) / _SAMPLE_RATE
        common = dict(
            language=language,
            beam_size=cfg.beam_size,
            word_timestamps=cfg.word_timestamps,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            no_speech_threshold=cfg.no_speech_threshold,
            vad_filter=cfg.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": cfg.vad_min_silence_ms,
                "speech_pad_ms": cfg.vad_speech_pad_ms,
                "threshold": cfg.vad_threshold,
            },
        )

        # No prompt here: measured against human captions, `hotwords` made the
        # sequential decoder *worse* in both languages (English CER 1.0% ->
        # 4.3%, Japanese 18.7% -> 20.6%), while the same text as a batched
        # `initial_prompt` made it better. This path is also the repair path
        # for chunks that looped, where a prompt is the last thing wanted.
        def sequential(clip, **overrides):
            return model.transcribe(
                clip,
                condition_on_previous_text=cfg.condition_on_previous_text,
                temperature=list(cfg.temperature),
                **(common | overrides),
            )

        if cfg.batched:
            segments, info = BatchedInferencePipeline(model).transcribe(
                audio,
                batch_size=cfg.batch_size,
                without_timestamps=False,
                initial_prompt=prompt,
                temperature=cfg.temperature[0],
                **common,
            )
        else:
            segments, info = sequential(audio)

        detected = getattr(info, "language", None) or language
        if progress:
            progress(0.0, f"Transcribing ({detected})")

        recent: list[str] = []
        dropped = repaired = 0
        for seg in segments:
            batch = [seg]
            if cfg.batched and _looks_degenerate(seg, cfg.compression_ratio_threshold):
                batch = _redecode(sequential, audio, seg, language=detected)
                repaired += 1

            for piece in batch:
                fragment = _to_fragment(piece)
                if fragment is None:
                    continue
                if _is_hallucination(fragment, recent, detected, total):
                    log.debug("dropped %.2f-%.2f %r", fragment.start, fragment.end, fragment.text)
                    dropped += 1
                    continue
                recent.append(fragment.text.lower())
                del recent[:-3]
                fragments.append(fragment)

            if progress and total:
                progress(min(1.0, float(seg.end) / total), "Transcribing")

        if repaired:
            log.info("re-decoded %d looping chunk(s) with temperature fallback", repaired)
        if dropped:
            log.info("dropped %d hallucinated segment(s)", dropped)

        fragments.sort(key=lambda f: f.start)
        if cfg.snap_to_speech:
            fragments = snap_to_speech(fragments, _speech_regions(audio, cfg))
    finally:
        del model
        _release_cuda()

    if not fragments:
        raise AsrFailed(
            "Whisper produced no speech at all. The audio file may be silent or corrupt."
        )

    if progress:
        progress(1.0, f"Transcribed {len(fragments)} fragments")
    return Transcription(fragments=fragments, language=detected)


def decode_windows(
    audio_path: Path,
    settings: Settings,
    windows: list[tuple[float, float]],
    *,
    language: str | None = None,
) -> list[list[Word]]:
    """Whisper's words (absolute times) inside each window, decoded without the VAD.

    Only for *timing* the YouTube lines Whisper skipped (``gapfill.retime``):
    the VAD is off because it is what rejected these stretches in the first
    place, and the text decoded here is matched against YouTube's, never shown.
    """
    if not windows:
        return []
    ensure_cuda_dlls()
    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

    cfg = settings.asr
    # One clip per window, decoded in batches. On a 27-minute video with 165
    # added lines: one at a time 189 s, batched 88 s with the same result.
    # Merging overlapping windows into 30 s clips took 36 s, but Whisper
    # skipped more of each long noisy clip: 64 lines retimed instead of 84.
    regions = [(lo, min(hi, lo + _CLIP_S)) for lo, hi in windows]
    try:
        model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except Exception as exc:  # noqa: BLE001
        raise AsrFailed(_explain_load_failure(exc)) from exc
    words: list[list[Word]] = [[] for _ in regions]
    try:
        audio = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
        duration_s = len(audio) / _SAMPLE_RATE
        regions = [(lo, min(hi, duration_s)) for lo, hi in regions]
        for group in _within(regions, duration_s):
            segments, _info = BatchedInferencePipeline(model).transcribe(
                audio,
                language=language,
                batch_size=cfg.batch_size,
                beam_size=cfg.beam_size,
                word_timestamps=True,
                without_timestamps=False,
                temperature=0.0,
                clip_timestamps=[{"start": regions[k][0], "end": regions[k][1]} for k in group],
            )
            starts = [regions[k][0] * model.frames_per_second for k in group]
            for seg in segments:
                # A segment's `seek` is its clip's offset in frames. Neighbouring
                # windows overlap, so the time alone does not say which clip heard it.
                r = group[min(range(len(starts)), key=lambda k: abs(starts[k] - seg.seek))]
                words[r].extend(
                    Word(start=float(w.start), end=float(w.end), text=w.word)
                    for w in seg.words or []
                    if w.start is not None and w.end is not None
                )
    finally:
        del model
        _release_cuda()
    return words


def align_texts(
    audio_path: Path,
    settings: Settings,
    items: list[tuple[float, float, str]],
    *,
    language: str | None = None,
) -> list[list[Word]]:
    """Where each given text was said inside its ``(start, end)`` window.

    Forced alignment, not transcription: nothing is decoded. The text goes
    through Whisper's alignment heads (the same DTW its word timestamps use),
    so a window where decoding loops (ああ ×30) or hallucinates still places the
    words. The first and last word absorb any silence at the window's edges;
    ``gapfill.forced_span`` trims them. Words carry the token probability.
    """
    if not items:
        return []
    ensure_cuda_dlls()
    import numpy as np
    from faster_whisper import WhisperModel, decode_audio
    from faster_whisper.tokenizer import Tokenizer

    cfg = settings.asr
    try:
        model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except Exception as exc:  # noqa: BLE001
        raise AsrFailed(_explain_load_failure(exc)) from exc
    out: list[list[Word]] = []
    try:
        audio = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
        tokenizer = Tokenizer(
            model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language
        )
        frames = int(_CLIP_S * model.frames_per_second)
        for lo, hi, text in items:
            lo = max(0.0, lo)
            hi = min(hi, lo + _CLIP_S, len(audio) / _SAMPLE_RATE)
            tokens = tokenizer.encode(text)
            if hi <= lo or not tokens:
                out.append([])
                continue
            features = model.feature_extractor(
                audio[int(lo * _SAMPLE_RATE) : int(hi * _SAMPLE_RATE)]
            )
            num_frames = min(features.shape[-1], frames)
            features = np.pad(features[:, :frames], ((0, 0), (0, frames - num_frames)))
            alignment = model.find_alignment(
                tokenizer, [tokens], model.encode(features), num_frames
            )[0]
            out.append(
                [
                    Word(
                        start=lo + float(a["start"]),
                        end=lo + float(a["end"]),
                        text=a["word"],
                        probability=float(a["probability"]),
                    )
                    for a in alignment
                ]
            )
    finally:
        del model
        _release_cuda()
    return out


#: Whisper's input length; a longer clip is cut at 30 s.
_CLIP_S = 30.0


def _within(regions: list[tuple[float, float]], duration_s: float) -> list[list[int]]:
    """Indices of ``regions`` in groups whose clips add up to no more than the audio.

    faster-whisper logs "VAD filter removed <duration - total clip length>" and
    asserts that it is not negative, so overlapping windows that add up to more
    than the audio crash the call: 53 lines on the 586 s d1MzjzSTwvc.
    """
    groups: list[list[int]] = []
    total = duration_s
    for k, (lo, hi) in enumerate(regions):
        if total + (hi - lo) > duration_s:
            groups.append([])
            total = 0.0
        groups[-1].append(k)
        total += hi - lo
    return groups


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def _speech_regions(audio, cfg) -> list[tuple[float, float]]:
    """Where someone is audibly speaking, from Silero VAD at a fine grain.

    The VAD that feeds the decoder merges anything closer than 400 ms into one
    chunk; this pass keeps pauses down to 150 ms, which is what a word's
    boundary needs.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=cfg.vad_threshold,
        min_silence_duration_ms=150,
        speech_pad_ms=100,
    )
    return [
        (chunk["start"] / _SAMPLE_RATE, chunk["end"] / _SAMPLE_RATE)
        for chunk in get_speech_timestamps(audio, options)
    ]


def snap_to_speech(
    fragments: list[Fragment], regions: list[tuple[float, float]]
) -> list[Fragment]:
    """Pull each word's timestamps inside the audio that actually has speech.

    Whisper's first word after a pause absorbs the silence before it: on a
    real clip "シャル、スマブラしないと思ってたの" was stamped from 79.7 s while
    the speaker started at 84.8 s, so the subtitle appeared five seconds early.
    Clamping every word to the speech it overlaps (the approach of stable-ts's
    silence suppression) moves those starts to the moment someone speaks, and
    turns the absorbed silence back into a real pause for segmentation.

    A word that overlaps no speech at all is left alone: shrinking it to
    nothing would lose its text, and the VAD misses quiet speech too.
    """
    if not regions:
        return fragments
    starts = [s for s, _ in regions]

    def clamp(start: float, end: float) -> tuple[float, float]:
        i = max(0, bisect.bisect_right(starts, end) - 1)
        overlapping = []
        while i >= 0 and regions[i][1] > start:
            if regions[i][0] < end:
                overlapping.append(regions[i])
            i -= 1
        if not overlapping:
            return start, end
        new_start = max(start, min(s for s, _ in overlapping))
        new_end = min(end, max(e for _, e in overlapping))
        if new_end - new_start < _MIN_WORD_S:
            new_end = min(end, new_start + _MIN_WORD_S)
        return new_start, new_end

    for fragment in fragments:
        if fragment.words:
            for word in fragment.words:
                word.start, word.end = clamp(word.start, word.end)
            _unstrand(fragment.words)
            fragment.start = fragment.words[0].start
            fragment.end = max(w.end for w in fragment.words)
        else:
            fragment.start, fragment.end = clamp(fragment.start, fragment.end)
    return fragments


#: A pause this long inside a Japanese word cannot be real.

_STRANDED_GAP_S = 0.5
_STRANDED_TOKEN_S = 0.1
#: One or two kana take well under this; longer, at a fragment's start, the
#: token has swallowed the silence or the other voice before it.
_OVERLONG_TOKEN_S = 0.6
#: A held vowel (えーー) is legitimately long.
_HELD_VOWEL = "ー〜"


def _unstrand(words: list[Word]) -> None:
    """Move sub-word tokens Whisper left behind back next to their word.

    Whisper often stamps the first kana of a Japanese phrase at the tail of
    the *previous* speaker's audio: シ at 79.72 s, then ャル、スマブラ… from
    84.44 s. VAD snapping cannot help, because 79.72 s really is speech —
    someone else's. Two tokens of one word with no punctuation between them
    are spoken together, so a gap between them is a misplaced prefix: the
    run of tokens before the gap is moved to end where the word continues.

    The prefix need not sit next to CJK: で at 228.13 s, then 1ラウンド from
    235.04 s. And the gap can hide inside a token: なん at 597.95 s, then で
    stretched over 598.53-600.41 s, then すぐ逃げる — so an over-long token at
    the start of a fragment is first cut back to its tail.
    """
    for word, following in zip(words[:2], words[1:3]):
        text = word.text.strip()
        if (
            word.end - word.start > _OVERLONG_TOKEN_S
            and following.start - word.end <= _STRANDED_TOKEN_S
            and 0 < len(text) <= 2
            and all(is_cjk_char(c) and c not in _HELD_VOWEL for c in text)
        ):
            word.start = word.end - _STRANDED_TOKEN_S
    for i in range(len(words) - 1):
        left, right = words[i], words[i + 1]
        if right.start - left.end <= _STRANDED_GAP_S:
            continue
        before, after = left.text.strip(), right.text.strip()
        if not (before and after and before[-1].isalnum() and after[0].isalnum()):
            continue
        if not (is_cjk_char(before[-1]) or is_cjk_char(after[0])):
            continue
        # The stranded run: back to the last punctuation or the fragment start.
        first = i
        while first > 0 and (words[first - 1].text.strip()[-1:] or " ").isalnum():
            first -= 1
        cursor = right.start
        for word in reversed(words[first : i + 1]):
            duration = min(word.end - word.start, _STRANDED_TOKEN_S)
            word.end = cursor
            word.start = cursor - duration
            cursor = word.start


_MIN_WORD_S = 0.05


def fragments_from_cues(cues: list[tuple[float, float, str]]) -> list[Fragment]:
    """Adapt human-uploaded captions into the shape the rest of the pipeline expects."""
    return [
        Fragment(start=start, end=end, text=text)
        for start, end, text in cues
        if text.strip()
    ]


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

#: One punctuated sentence in the target style. Whisper imitates the prompt's
#: formatting, so this is what switches Japanese punctuation on.
_STYLE = {
    "ja": "えっと、今日はみんなでゲームをします。よろしくお願いします！",
    "en": "Okay, so today we're playing with everyone. Let's go!",
}
_BRACKETED = re.compile(r"[【\[「『]([^】\]」』]{1,24})[】\]」』]")
#: Words clip and translation channels bracket that are not anyone's name.
_NOT_A_NAME = re.compile(
    r"切り抜き|切抜|精華|翻譯|翻訳|中文|中字|字幕|烤肉|熟肉|clip|highlight|vtuber|sub|eng|jp",
    re.IGNORECASE,
)
_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_MAX_NAMES = 6


def build_prompt(title: str, description: str, tags: tuple[str, ...], language: str | None) -> str:
    """The names in a video's title, after a punctuated sentence in its language.

    Only the title's bracketed names are used — 【橘ひなの】【猫汰つな】 — and
    only up to six. Descriptions and tags are where channels pile hashtags,
    and a long prompt of those produced loops rather than better spelling.
    ``description`` and ``tags`` are accepted for callers that have them and
    for a future, better-measured use.
    """
    picked: list[str] = []
    seen: set[str] = set()
    for name in _BRACKETED.findall(title):
        name = name.strip()
        if not name or name.lower() in seen or _NOT_A_NAME.search(name):
            continue
        # In a Japanese video, Latin glued to Chinese characters ("Vspo精華")
        # is a Chinese channel's label, not a name anyone says.
        if language == "ja" and _LATIN.search(name) and _HAN.search(name) and not _KANA.search(name):
            continue
        seen.add(name.lower())
        picked.append(name)
        if len(picked) >= _MAX_NAMES:
            break

    style = _STYLE.get(language or "", "")
    if not picked:
        return style
    separator, stop = ("、", "。") if language == "ja" else (", ", ".")
    return f"{style} {separator.join(picked)}{stop}".strip()



# --------------------------------------------------------------------------
# repair and filtering
# --------------------------------------------------------------------------

#: Worth a second, slower decode: six repeats of one phrase.
_LOOP = re.compile(r"(.{2,20}?)\1{5,}", re.DOTALL)
#: Not worth showing at all. Real speech repeats — 待って待って待って — but
#: not nine times in one breath.
_RUNAWAY = re.compile(r"(.{2,20}?)\1{8,}", re.DOTALL)


def _compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def _looks_degenerate(seg, threshold: float) -> bool:
    text = (seg.text or "").strip()
    if len(text) < 12:
        return False
    return _compression_ratio(text) > threshold or bool(_LOOP.search(text.replace(" ", "")))


def _redecode(sequential, audio, seg, *, language: str | None) -> list:
    start = max(0, int(seg.start * _SAMPLE_RATE))
    end = min(len(audio), int(seg.end * _SAMPLE_RATE) + _SAMPLE_RATE // 4)
    try:
        pieces, _ = sequential(audio[start:end], **({"language": language} if language else {}))
        pieces = list(pieces)
    except Exception as exc:  # noqa: BLE001 - keep the batched result rather than nothing
        log.warning("re-decoding %.1f-%.1fs failed: %s", seg.start, seg.end, exc)
        return [seg]
    offset = start / _SAMPLE_RATE
    return [_Shifted(p, offset) for p in pieces]


class _Shifted:
    """A segment from a clipped re-decode, moved back onto the video's clock."""

    def __init__(self, seg, offset: float) -> None:
        self.start = seg.start + offset
        self.end = seg.end + offset
        self.text = seg.text
        self.no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
        self.avg_logprob = getattr(seg, "avg_logprob", 0.0)
        self.words = [_ShiftedWord(w, offset) for w in (seg.words or [])]


class _ShiftedWord:
    def __init__(self, word, offset: float) -> None:
        self.start = None if word.start is None else word.start + offset
        self.end = None if word.end is None else word.end + offset
        self.word = word.word
        self.probability = getattr(word, "probability", 1.0)


def _to_fragment(seg) -> Fragment | None:
    text = (seg.text or "").strip()
    if not text:
        return None
    return Fragment(
        start=float(seg.start),
        end=float(seg.end),
        text=text,
        words=[
            Word(
                start=float(w.start),
                end=float(w.end),
                text=w.word,
                probability=float(getattr(w, "probability", 1.0)),
            )
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ],
        no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
        avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
    )


_REPEAT = re.compile(r"\b(\w[\w']*)(?:\W+\1\b){3,}", re.IGNORECASE)
_NON_WORD = re.compile(r"[\W_]+")

_LATIN_ONLY = re.compile(r"^[\sA-Za-z'.,!?-]+$")


def _is_hallucination(
    fragment: Fragment, recent: list[str], language: str | None, duration_s: float
) -> bool:
    text = fragment.text.strip()
    normalized = text.lower()
    near_end = duration_s and fragment.start > duration_s - 30.0
    if normalized in _FILLER_HALLUCINATIONS and (fragment.no_speech_prob > 0.5 or not near_end):
        return True
    if _CREDIT.search(text):
        return True
    # Text with no time to have been spoken in: a loop Whisper spliced in as
    # zero-length segments (four copies of an earlier line at 319.71 s).
    spoken = len(_NON_WORD.sub("", normalized))
    if spoken >= 4 and fragment.end - fragment.start < 0.1:
        return True
    # The same longer line three times running is the model looping. Short
    # reactions repeat for real when several people talk: two people's "え？"
    # three times over is a conversation, and dropping it lost those lines.
    if spoken >= 8 and len(normalized) < 60 and recent.count(normalized) >= 2:
        return True
    if _REPEAT.search(text) or _RUNAWAY.search(text.replace(" ", "")):
        return True
    # A stray Latin phrase in a CJK-language video, decoded with low confidence,
    # is noise Whisper spelled in the wrong alphabet ("insurance calcula").
    if (
        language in ("ja", "zh", "ko")
        and _LATIN_ONLY.match(text)
        and not is_cjk(text)
        and fragment.avg_logprob < -0.6
    ):
        return True
    return False


def _explain_load_failure(exc: Exception) -> str:
    message = str(exc)
    if "cudnn" in message.lower() or "cublas" in message.lower():
        return (
            f"{message}\n\n"
            "This is the cuDNN/cuBLAS DLL problem. Run `.\\setup.ps1` again, or:\n"
            "  uv pip install 'nvidia-cudnn-cu12>=9.1,<10' 'nvidia-cublas-cu12>=12.4'"
        )
    return message


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
