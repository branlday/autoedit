"""Shared data structures for the auto-edit pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class MotionSample:
    """Motion score at a point in time (mean frame-diff intensity)."""
    time: float
    score: float


@dataclass
class AudioSpike:
    time: float
    zscore: float


@dataclass
class ClipSegment:
    """A single classified region of source footage - the atomic unit that
    either becomes part of the rough cut, or gets logged as a cut candidate.
    """
    source_path: str
    start: float
    end: float
    tag: str                # TALKING_HEAD | NARRATION | REACTION | DEAD_AIR | REPEATED_ACTION_TRIM | KEEP_WORK
    reason: str = ""         # human-readable explanation, goes in the audit log
    keep: bool = True
    text: Optional[str] = None  # transcript text if speech is present

    @property
    def duration(self) -> float:
        return self.end - self.start
