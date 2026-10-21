"""Timed text segments and subtitle-file parsing (WebVTT, SRT)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # as Whisper emits it, usually with a leading space


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float
    text: str
    speaker: str | None = None  # e.g. "화자 1" once diarization ran
    words: tuple[Word, ...] = ()  # filled when word timestamps were requested


_TIMING = re.compile(
    r"(?P<s>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*(?P<e>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)
_TAG = re.compile(r"<[^>]*>")


def parse_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _clean(line: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", line)).split())


def _cues(text: str) -> list[tuple[float, float, list[str]]]:
    """Split subtitle text into (start, end, lines) cues; works for VTT and SRT."""
    cues: list[tuple[float, float, list[str]]] = []
    current: tuple[float, float, list[str]] | None = None
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _TIMING.search(raw)
        if match:
            current = (parse_timestamp(match["s"]), parse_timestamp(match["e"]), [])
            cues.append(current)
        elif not raw.strip():
            current = None
        elif current is not None:
            line = _clean(raw)
            if line:
                current[2].append(line)
    return cues


def parse_subtitles(text: str) -> list[Segment]:
    """Parse WebVTT or SRT into segments.

    YouTube auto-caption VTT repeats the previous line at the top of each cue
    ("rolling" captions). Lines identical to the previous cue's lines are
    dropped so each spoken phrase appears once.
    """
    segments: list[Segment] = []
    previous: list[str] = []
    for start, end, lines in _cues(text):
        fresh = [line for line in lines if line not in previous]
        previous = lines
        if fresh and end > start:
            segments.append(Segment(start, end, " ".join(fresh)))
    return segments
