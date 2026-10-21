"""Subtitle and plain-text exports: SRT, WebVTT, TXT."""

from __future__ import annotations

from .segments import Segment


def _clock(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _line(segment: Segment) -> str:
    return f"[{segment.speaker}] {segment.text}" if segment.speaker else segment.text


def to_srt(segments: list[Segment] | tuple[Segment, ...]) -> str:
    blocks = [
        f"{i}\n{_clock(s.start, ',')} --> {_clock(max(s.end, s.start + 0.2), ',')}\n{_line(s)}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def _cue(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(segments: list[Segment] | tuple[Segment, ...]) -> str:
    blocks = [
        f"{_clock(s.start, '.')} --> {_clock(max(s.end, s.start + 0.2), '.')}\n"
        + (f"<v {_cue(s.speaker)}>{_cue(s.text)}" if s.speaker else _cue(s.text)) + "\n"
        for s in segments
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(segments: list[Segment] | tuple[Segment, ...]) -> str:
    """Plain text; a new paragraph whenever the speaker changes."""
    paragraphs: list[tuple[str | None, list[str]]] = []
    for s in segments:
        if paragraphs and paragraphs[-1][0] == s.speaker:
            paragraphs[-1][1].append(s.text)
        else:
            paragraphs.append((s.speaker, [s.text]))
    out = [(f"{who}: " if who else "") + " ".join(texts) for who, texts in paragraphs]
    return "\n\n".join(out) + "\n"
