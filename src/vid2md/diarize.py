"""Speaker diarization (who spoke when) with pyannote.audio, and assigning speakers to text.

The pyannote model is gated on Hugging Face: accept its conditions once at
https://huggingface.co/pyannote/speaker-diarization-community-1 and pass a token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from .segments import Segment, Word

log = logging.getLogger("vid2md.diarize")

MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000


class DiarizationError(RuntimeError):
    """Diarization could not run; the message is meant for the user."""


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str  # raw pyannote label, e.g. "SPEAKER_00"


def diarize(path: Path, token: str | None, num_speakers: int | None = None, device: str = "auto") -> list[Turn]:
    if not token:
        raise DiarizationError("화자 구분에는 Hugging Face 토큰(HF_TOKEN)이 필요합니다.")
    try:
        import torch
        from faster_whisper.audio import decode_audio
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise DiarizationError("화자 구분 라이브러리가 없습니다: pip install 'vid2md[diarize]'") from error

    try:
        pipeline = Pipeline.from_pretrained(MODEL, token=token)
    except Exception as error:  # noqa: BLE001 - surfaced to the user with the likely fix
        raise DiarizationError(
            f"화자 구분 모델을 받지 못했습니다. https://huggingface.co/{MODEL} 에서 이용 조건에 동의했는지, "
            f"토큰이 맞는지 확인하세요. ({str(error).splitlines()[0][:120]})"
        ) from error
    if pipeline is None:
        raise DiarizationError(f"화자 구분 모델 접근이 거부됐습니다. https://huggingface.co/{MODEL} 에서 이용 조건에 동의하세요.")
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        pipeline.to(torch.device("cuda"))

    # Decode ourselves (same decoder as Whisper) and hand pyannote a waveform,
    # which avoids depending on its own audio backend.
    try:
        waveform = torch.from_numpy(decode_audio(str(path), sampling_rate=SAMPLE_RATE))[None]
        output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, num_speakers=num_speakers)
    except Exception as error:  # noqa: BLE001 - optional step; the caller keeps the transcript without speakers
        raise DiarizationError(f"화자 구분 중 오류가 났습니다 ({str(error).splitlines()[0][:120] if str(error) else type(error).__name__})") from error
    # pyannote 4 also offers a version without overlapping turns, meant for transcripts.
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    turns = [Turn(float(seg.start), float(seg.end), str(label)) for seg, _, label in annotation.itertracks(yield_label=True)]
    log.info("화자 구분: 화자 %d명, 구간 %d개", len({t.speaker for t in turns}), len(turns))
    return turns


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_at(start: float, end: float, turns: list[Turn]) -> str | None:
    """Speaker with the most overlap; for a gap between turns, the nearest turn."""
    best, best_overlap = None, 0.0
    for turn in turns:
        amount = _overlap(start, end, turn.start, turn.end)
        if amount > best_overlap:
            best, best_overlap = turn.speaker, amount
    if best is not None or not turns:
        return best
    middle = (start + end) / 2
    return min(turns, key=lambda t: min(abs(middle - t.start), abs(middle - t.end))).speaker


def assign_speakers(segments: list[Segment] | tuple[Segment, ...], turns: list[Turn]) -> list[Segment]:
    """Label each segment with a speaker, splitting segments whose words change speaker.

    Speakers are renamed "화자 1", "화자 2", ... in order of first appearance.
    """
    names: dict[str, str] = {}

    def name(raw: str | None) -> str | None:
        if raw is None:
            return None
        if raw not in names:
            names[raw] = f"화자 {len(names) + 1}"
        return names[raw]

    out: list[Segment] = []
    for segment in segments:
        if not segment.words:
            out.append(replace(segment, speaker=name(_speaker_at(segment.start, segment.end, turns))))
            continue
        runs: list[tuple[str | None, list[Word]]] = []
        for word in segment.words:
            speaker = name(_speaker_at(word.start, word.end, turns))
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(word)
            else:
                runs.append((speaker, [word]))
        for speaker, words in runs:
            text = "".join(w.text for w in words).strip()
            if text:
                out.append(Segment(words[0].start, words[-1].end, text, speaker=speaker, words=tuple(words)))
    return out
