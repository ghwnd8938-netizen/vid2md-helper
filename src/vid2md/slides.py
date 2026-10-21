"""화면 글자: the text on the slides of a lecture video.

Frames are taken every few seconds (keyframes first, which costs little); a frame where enough of a small grey
thumbnail changed since the last one kept counts as a new slide, and Tesseract reads it. Only a trusted worker runs
this: the external server never receives a video.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

log = logging.getLogger("vid2md.slides")

EVERY_SECONDS = 3.0  # at most one frame this often
# A new slide: at least CHANGE of the thumbnail's pixels moved by more than PIXEL (0-1). Measured on slides with
# the same background: 2.5-4.5% of pixels change between slides, none within one.
CHANGE = 0.02
PIXEL = 0.15
MAX_SLIDES = 300
THUMB = (72, 128)  # rows, columns
LANGS = "kor+eng"
PSM = "4"  # one column of text in mixed sizes: on test slides it read every line where the default (3) lost two slides

Ocr = Callable[["object"], str]  # a grey frame (2-D uint8 array) -> its text


def read_slides(video: Path, ocr: Ocr | None = None, every: float = EVERY_SECONDS) -> str:
    """The text of the video's slides, each slide's new lines once, in order ('' if there is nothing to read)."""
    if ocr is None:
        if not shutil.which("tesseract"):
            log.warning("tesseract가 없어 화면 글자를 읽지 않습니다")
            return ""
        ocr = tesseract
    seen: set[str] = set()
    blocks = []
    for gray in slide_frames(video, every):
        lines = [line.strip() for line in ocr(gray).splitlines()]
        new = [line for line in lines if len(line) >= 2 and line not in seen]
        seen.update(new)
        if new:
            blocks.append("\n".join(new))
    log.info("화면 글자: 슬라이드 %d장에서 %d줄", len(blocks), len(seen))
    return "\n\n".join(blocks)


def slide_frames(video: Path, every: float = EVERY_SECONDS) -> list:
    """Grey frames where the picture changed. Keyframes only, unless the video has too few of them to notice
    slide changes (then every frame is decoded, which is slower)."""
    frames, times, duration = _frames(video, every, keyframes_only=True)
    if duration and len(times) < duration / 30:
        frames, _, _ = _frames(video, every, keyframes_only=False)
    return frames


def _frames(video: Path, every: float, keyframes_only: bool):
    import av
    import numpy as np

    kept, times, last, due = [], [], None, 0.0
    with av.open(str(video)) as container:
        if not container.streams.video:
            return [], [], 0.0
        stream = container.streams.video[0]
        duration = float(container.duration or 0) / 1_000_000
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            t = float(frame.time or 0.0)
            times.append(t)
            if t < due:
                continue
            due = t + every
            gray = frame.to_ndarray(format="gray")
            thumb = _thumbnail(gray, np)
            if last is None or float((np.abs(thumb - last) > PIXEL).mean()) >= CHANGE:
                kept.append(gray)
                last = thumb
                if len(kept) >= MAX_SLIDES:
                    break
    return kept, times, duration


def _thumbnail(gray, np):
    rows, cols = THUMB
    h, w = gray.shape
    if h < rows or w < cols:
        return gray.astype("float32") / 255.0
    cut = gray[: h - h % rows, : w - w % cols].astype("float32") / 255.0
    return cut.reshape(rows, cut.shape[0] // rows, cols, cut.shape[1] // cols).mean(axis=(1, 3))


def tesseract(gray) -> str:
    """Tesseract on one grey frame (written as a PGM file, which it reads without extra libraries)."""
    h, w = gray.shape
    with tempfile.NamedTemporaryFile(suffix=".pgm") as image:
        image.write(f"P5\n{w} {h}\n255\n".encode() + gray.tobytes())
        image.flush()
        try:
            done = subprocess.run(["tesseract", image.name, "stdout", "-l", LANGS, "--psm", PSM],
                                  capture_output=True, timeout=120, check=False)
        except subprocess.TimeoutExpired:
            return ""
    return done.stdout.decode("utf-8", "replace")
