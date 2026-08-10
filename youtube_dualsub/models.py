"""The vocabulary every pipeline stage speaks.

The chain of transformations is::

    Fragment   raw ASR output, often cut mid-sentence
      -> Sentence   one whole spoken thought, source language
      -> Translation  the zh-TW rendering of one Sentence, by index
      -> Cue        what actually gets drawn on screen, after shaping

Sentence.index is the join key that survives the whole pipeline: it is what
lets translations be cached separately from ASR and re-run on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    QUEUED = "queued"
    AUDIO = "audio"
    VOCALS = "vocals"
    ASR = "asr"
    SENTENCES = "sentences"
    CONTEXT = "context"
    TRANSLATE = "translate"
    SHAPE = "shape"
    DONE = "done"
    PAUSED = "paused"
    FAILED = "failed"


#: Human-readable ordering used for progress reporting.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.AUDIO,
    Stage.VOCALS,
    Stage.ASR,
    Stage.SENTENCES,
    Stage.CONTEXT,
    Stage.TRANSLATE,
    Stage.SHAPE,
)


class TranslationStatus(str, Enum):
    OK = "ok"
    #: The batch could not be aligned or the LLM never answered; the English
    #: source is shown instead of a guess (decision Q19).
    SOURCE_FALLBACK = "source_fallback"


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass(slots=True)
class Fragment:
    """A raw faster-whisper segment. Frequently half a sentence."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0


@dataclass(slots=True)
class Sentence:
    index: int
    start: float
    end: float
    text: str
    #: Reserved for diarization (decision Q26): the field and its rendering
    #: exist now so adding pyannote later touches one stage, not the pipeline.
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Translation:
    index: int
    text: str
    status: TranslationStatus = TranslationStatus.OK


@dataclass(slots=True)
class Cue:
    """A drawable subtitle. ``end`` is already clamped and never overlaps the next cue."""

    start: float
    end: float
    source: str
    target: str
    speaker: str | None = None
    translated: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class VideoContext:
    """Whole-video knowledge injected into every translation batch (decision Q22)."""

    summary: str = ""
    #: source term -> preferred zh-TW rendering, auto-extracted from this video only.
    terms: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Progress:
    stage: Stage
    #: 0.0-1.0 within the current stage, or None when indeterminate.
    fraction: float | None = None
    message: str = ""
    detail: dict[str, object] = field(default_factory=dict)
