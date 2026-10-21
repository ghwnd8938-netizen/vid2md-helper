"""Render a Transcript as Obsidian-friendly Markdown."""

from __future__ import annotations

import json
import re
from datetime import date

from .segments import Segment, format_timestamp
from .transcript import Transcript

PARAGRAPH_GAP = 2.0  # seconds of silence that start a new paragraph
PARAGRAPH_MAX_SECONDS = 60.0
PARAGRAPH_MAX_CHARS = 500
_SENTENCE_END = re.compile(r"[.?!。？！]$|(?:다|요|죠|까)[.?!]?$")


def paragraphs(segments: tuple[Segment, ...] | list[Segment]) -> list[list[Segment]]:
    """Group segments into readable paragraphs.

    A paragraph breaks on a pause, or once it is long and the current segment
    ends a sentence (hard break if it grows far past the limits).
    """
    groups: list[list[Segment]] = []
    for segment in segments:
        current = groups[-1] if groups else None
        if current is None:
            groups.append([segment])
            continue
        span = segment.end - current[0].start
        chars = sum(len(s.text) for s in current)
        long_enough = span >= PARAGRAPH_MAX_SECONDS or chars >= PARAGRAPH_MAX_CHARS
        gap = segment.start - current[-1].end
        if (
            segment.speaker != current[-1].speaker
            or gap >= PARAGRAPH_GAP
            or (long_enough and _SENTENCE_END.search(current[-1].text))
            or span >= PARAGRAPH_MAX_SECONDS * 2
        ):
            groups.append([segment])
        else:
            current.append(segment)
    return groups


def _time_link(transcript: Transcript, seconds: float) -> str:
    label = format_timestamp(seconds)
    if transcript.source.kind == "youtube" and transcript.source.video_id:
        return f"[{label}](https://www.youtube.com/watch?v={transcript.source.video_id}&t={int(seconds)}s)"
    return f"`{label}`"


def _yaml(value: object) -> str:
    # JSON strings are valid YAML double-quoted scalars, so this escapes safely.
    return json.dumps(value, ensure_ascii=False)


def render(transcript: Transcript, section_minutes: int = 5, today: date | None = None) -> str:
    src = transcript.source
    front = [
        "---",
        f"title: {_yaml(src.title)}",
        f"source: {_yaml(src.locator)}",
        f"method: {transcript.method}",
        f"confidence: {transcript.confidence}",
        f"provider: {_yaml(transcript.provider)}",
        f"language: {transcript.language}",
    ]
    if src.duration:
        front.append(f"duration: {_yaml(format_timestamp(src.duration))}")
    front.append(f"transcribed_until: {_yaml(format_timestamp(transcript.covered_until))}")
    if src.sha256:
        front.append(f"sha256: {src.sha256}")
    front.append(f"generated: {(today or date.today()).isoformat()}")
    if transcript.speakers:
        front.append(f"speakers: {len(transcript.speakers)}")
    if transcript.hints:
        front.append(f"hints: {_yaml(', '.join(transcript.hints))}")
    if transcript.limitations:
        front.append("limitations:")
        front.extend(f"  - {_yaml(item)}" for item in transcript.limitations)
    front.append("---")

    body = [f"# {src.title}", ""]
    section_seconds = max(section_minutes, 1) * 60
    current_section = -1
    for group in paragraphs(transcript.segments):
        start = group[0].start
        section = int(start // section_seconds)
        if section != current_section:
            current_section = section
            body.extend([f"## {format_timestamp(section * section_seconds)}", ""])
        text = " ".join(s.text for s in group)
        who = f"**{group[0].speaker}** " if group[0].speaker else ""
        body.extend([f"{_time_link(transcript, start)} {who}{text}", ""])

    if not transcript.segments:
        body.extend(["_전사된 내용이 없습니다._", ""])
    return "\n".join(front + [""] + body).rstrip() + "\n"
