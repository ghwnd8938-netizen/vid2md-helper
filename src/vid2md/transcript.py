"""Transcript record: segments plus where they came from and how far to trust them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .segments import Segment

Method = Literal["captions-manual", "captions-auto", "subtitle-file", "whisper"]
# observed: text a person published for this media (uploaded captions, a supplied
#   subtitle file). inferred: a speech-recognition model's guess (YouTube auto
#   captions, Whisper). Keep the two apart; they deserve different trust.
Confidence = Literal["observed", "inferred"]


@dataclass(frozen=True)
class SourceInfo:
    kind: Literal["youtube", "file"]
    locator: str  # URL or absolute path
    title: str
    duration: float | None = None
    video_id: str | None = None
    sha256: str | None = None  # local files only


@dataclass(frozen=True)
class Transcript:
    source: SourceInfo
    method: Method
    confidence: Confidence
    provider: str  # producing tool, version and model, e.g. "faster-whisper 1.2.1 (large-v3-turbo)"
    language: str
    segments: tuple[Segment, ...]
    limitations: tuple[str, ...] = field(default=())
    hints: tuple[str, ...] = field(default=())  # glossary words picked for this recording (automatic choice)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.speaker and s.speaker not in seen:
                seen.append(s.speaker)
        return seen

    @property
    def covered_until(self) -> float:
        return self.segments[-1].end if self.segments else 0.0
