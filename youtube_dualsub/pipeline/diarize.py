"""Who is speaking when: speaker diarization with sherpa-onnx.

Three stages, all ONNX on the CPU, so this never competes with Whisper or the
LLM for VRAM: pyannote's segmentation-3.0 finds speech turns (overlaps
included), a speaker-embedding model fingerprints each turn, and clustering
groups the fingerprints into speakers. Nobody is named and nothing is shown —
the result is "speaker 1, 2, 3", numbered in order of first appearance, and
its only job is to stop two people's words being glued into one subtitle.

Two things here look odd and are not:

``_preload_onnxruntime``
    The sherpa-onnx wheel does not ship ``onnxruntime.dll``. Windows then finds
    the copy in System32 (1.17, installed for Windows ML) before ours, and
    sherpa fails with "API version [28] is not available". Loading the pip
    package's DLL first makes every later lookup resolve to it.

Models come from Hugging Face, not GitHub releases
    The same files downloaded at ~35 KB/s from GitHub releases and at ~7 MB/s
    from the author's Hugging Face mirror.
"""

from __future__ import annotations

import bisect
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import MODEL_DIR, Settings
from ..models import Fragment, Sentence

log = logging.getLogger(__name__)

ProgressCb = Callable[[float | None, str], None]

_HF = "https://huggingface.co/csukuangfj"
_SEGMENTATION_URL = f"{_HF}/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
_EMBEDDING_URL = f"{_HF}/speaker-embedding-models/resolve/main/{{name}}"
_SAMPLE_RATE = 16000

# The official example slides pyannote's window by 10 % of its length. At
# 50 % it ran 4.5x faster on a real clip and agreed with reference labels
# more (0.67 vs 0.56), because the finer grid mostly invented extra speakers.
_WINDOW_SHIFT = 0.5
_MINOR_SPEAKER_S = 5.0
_MINOR_SPEAKER_SHARE = 0.03



class DiarizationUnavailable(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def diarize(audio_path: Path, settings: Settings, *, progress: ProgressCb | None = None) -> list[Turn]:
    """Speaker turns for the whole file, speakers numbered "1", "2", … by first appearance."""
    cfg = settings.diarization
    segmentation, embedding = ensure_models(settings, progress=progress)

    _preload_onnxruntime()
    import sherpa_onnx
    from faster_whisper import decode_audio

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation), window_shift_ratio=_WINDOW_SHIFT
            ),
            num_threads=cfg.num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=cfg.num_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=cfg.num_speakers or -1, threshold=cfg.cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise DiarizationUnavailable("sherpa-onnx rejected the diarization config")

    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    audio = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)

    reported = [-1]

    def callback(done: int, total: int) -> int:
        # sherpa calls back once per chunk — hundreds of times — and every
        # progress report is a database write; tenths are plenty.
        tenth = 10 * done // max(1, total)
        if progress and tenth > reported[0]:
            reported[0] = tenth
            progress(done / total, "Telling speakers apart")
        return 0

    result = engine.process(audio, callback=callback).sort_by_start_time()
    raw = _drop_minor_speakers([(r.start, r.end, r.speaker) for r in result])
    return _number_by_appearance(raw)


def ensure_models(settings: Settings, *, progress: ProgressCb | None = None) -> tuple[Path, Path]:
    """Download the two ONNX models on first use (~7 MB + ~30 MB)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = settings.diarization.embedding_model
    wanted = [
        (MODEL_DIR / "pyannote-segmentation-3-0.onnx", _SEGMENTATION_URL),
        (MODEL_DIR / name, _EMBEDDING_URL.format(name=name)),
    ]
    for path, url in wanted:
        if path.is_file() and path.stat().st_size > 0:
            continue
        if progress:
            progress(None, f"Downloading {path.name}")
        log.info("downloading %s", url)
        partial = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
        except Exception as exc:  # noqa: BLE001 - surfaced as a degradable failure
            partial.unlink(missing_ok=True)
            raise DiarizationUnavailable(f"could not download {path.name}: {exc}") from exc
        partial.replace(path)
    return wanted[0][0], wanted[1][0]


def label_fragments(fragments: list[Fragment], turns: list[Turn]) -> None:
    """Give every word the speaker whose turn covers most of it."""
    index = TurnIndex(turns)
    for fragment in fragments:
        for word in fragment.words:
            word.speaker = index.speaker_at(word.start, word.end)
        if not fragment.words:
            fragment.speaker = index.speaker_at(fragment.start, fragment.end)


def label_sentences(sentences: list[Sentence], turns: list[Turn]) -> None:
    """For caption lines, which carry no word timings: the dominant speaker.

    A human caption line sometimes holds two people ("クーラーの温度？ はい
    熱くなりますので"); it gets whoever talks longest in it.
    """
    index = TurnIndex(turns)
    for sentence in sentences:
        sentence.speaker = index.speaker_at(sentence.start, sentence.end)


class TurnIndex:
    """Turns sorted by start, searchable without scanning the whole video."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.longest = max((t.end - t.start for t in self.turns), default=0.0)

    def speaker_at(self, start: float, end: float, reach: float = 0.5) -> str | None:
        """Most overlap with [start, end]; else the nearest turn within ``reach``."""
        lo = bisect.bisect_left(self.starts, start - self.longest - reach)
        hi = bisect.bisect_right(self.starts, end + reach)
        overlap_by: dict[str, float] = {}
        nearest, nearest_gap = None, reach
        for turn in self.turns[lo:hi]:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0:
                overlap_by[turn.speaker] = overlap_by.get(turn.speaker, 0.0) + overlap
            elif -overlap < nearest_gap:
                nearest, nearest_gap = turn.speaker, -overlap
        if overlap_by:
            return max(overlap_by, key=overlap_by.get)
        return nearest


def _drop_minor_speakers(raw: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Forget clusters that barely speak.

    Clustering also finds "speakers" made of laughter, a game's voice lines or
    one odd breath: on a three-person clip, seven clusters, four of them under
    8 seconds. Their turns are dropped, and the words inside take the nearest
    real speaker instead of a colour of their own.
    """
    talk: dict[int, float] = {}
    for start, end, cluster in raw:
        talk[cluster] = talk.get(cluster, 0.0) + (end - start)
    floor = max(_MINOR_SPEAKER_S, _MINOR_SPEAKER_SHARE * sum(talk.values()))
    keep = {c for c, t in talk.items() if t >= floor} or set(talk)
    return [turn for turn in raw if turn[2] in keep]


def _number_by_appearance(raw: list[tuple[float, float, int]]) -> list[Turn]:
    names: dict[int, str] = {}
    turns: list[Turn] = []
    for start, end, cluster in sorted(raw):
        name = names.setdefault(cluster, str(len(names) + 1))
        turns.append(Turn(start=round(start, 3), end=round(end, 3), speaker=name))
    return turns


_preloaded = False


def _preload_onnxruntime() -> None:
    global _preloaded
    if _preloaded:
        return
    import ctypes
    import os
    import sys

    if sys.platform == "win32":
        import onnxruntime

        dll = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
        if dll.is_file():
            ctypes.WinDLL(os.fspath(dll))
    _preloaded = True


__all__ = [
    "DiarizationUnavailable",
    "Turn",
    "diarize",
    "ensure_models",
    "label_fragments",
    "label_sentences",
    "TurnIndex",
]

