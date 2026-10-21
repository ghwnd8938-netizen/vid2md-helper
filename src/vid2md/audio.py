"""Save the sound of a video or recording as a small MP3 (for listening or sharing next to the transcript)."""

from __future__ import annotations

from pathlib import Path

MP3_RATE = 44100
MP3_BITRATE = 64000  # mono speech; about 29 MB per hour


class AudioError(RuntimeError):
    """The file has no sound to save, or it could not be read."""


def extract_mp3(source: Path, target: Path, bitrate: int = MP3_BITRATE) -> Path:
    """Write the first audio track of ``source`` to ``target`` as mono MP3. Video is skipped, not decoded."""
    import av  # bundled with faster-whisper

    try:
        with av.open(str(source)) as inp:
            if not inp.streams.audio:
                raise AudioError("소리가 없는 파일입니다.")
            stream = inp.streams.audio[0]
            with av.open(str(target), "w", format="mp3") as out:
                enc = out.add_stream("libmp3lame", rate=MP3_RATE)
                enc.layout = "mono"
                enc.bit_rate = bitrate
                resampler = av.AudioResampler(format="s16p", layout="mono", rate=MP3_RATE)
                for frame in inp.decode(stream):
                    frame.pts = None
                    for piece in resampler.resample(frame):
                        out.mux(enc.encode(piece))
                for piece in resampler.resample(None):
                    out.mux(enc.encode(piece))
                out.mux(enc.encode(None))
    except AudioError:
        raise
    except Exception as error:  # noqa: BLE001 - any decode/encode failure
        raise AudioError(f"음성 파일을 만들지 못했습니다: {error}") from error
    return target
