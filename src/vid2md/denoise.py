"""Keep only the voice (remove background music and noise) with Demucs before recognition."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Callable

log = logging.getLogger("vid2md.denoise")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 300  # separate 5 minutes at a time so a long video does not exhaust memory
MODEL = "htdemucs"


class DenoiseError(RuntimeError):
    """Voice separation could not run; the message is meant for the user."""


def isolate_voice(path: Path, out_path: Path, device: str = "auto",
                  progress: Callable[[float, float], None] | None = None) -> Path:
    """Write a 16 kHz mono WAV with only the vocals of `path` and return it."""
    try:
        import numpy as np
        import torch
        import torchaudio
        from demucs.api import Separator
        from faster_whisper.audio import decode_audio
    except ImportError as error:
        raise DenoiseError("배경음 제거 라이브러리가 없습니다: pip install 'vid2md[denoise]'") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return _separate(path, out_path, device, progress, np, torch, torchaudio, Separator, decode_audio)
    except Exception as error:  # noqa: BLE001 - optional step; the caller falls back to the original audio
        raise DenoiseError(f"배경음 제거 중 오류가 났습니다 ({str(error).splitlines()[0][:120] if str(error) else type(error).__name__})") from error


def _separate(path, out_path, device, progress, np, torch, torchaudio, Separator, decode_audio) -> Path:
    separator = Separator(model=MODEL, device=device, progress=False)
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    total = len(audio) / SAMPLE_RATE
    step = CHUNK_SECONDS * SAMPLE_RATE
    log.info("배경음 제거: %s, %.0f초, %s", MODEL, total, device)

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        for start in range(0, len(audio), step):
            chunk = torch.from_numpy(audio[start:start + step])[None]
            _, stems = separator.separate_tensor(chunk, SAMPLE_RATE)
            vocals = stems["vocals"].mean(0)  # stereo -> mono, at the model's sample rate
            vocals = torchaudio.functional.resample(vocals, separator.samplerate, SAMPLE_RATE)
            pcm = (vocals.clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
            out.writeframes(pcm.tobytes())
            if progress:
                progress(min((start + step) / SAMPLE_RATE, total), total)
    return out_path
