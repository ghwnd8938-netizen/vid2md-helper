"""Source -> Transcript: captions when trustworthy ones exist, Whisper otherwise."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import denoise as denoise_mod
from . import diarize as diarize_mod
from .segments import Segment, format_timestamp, parse_subtitles
from .transcript import SourceInfo, Transcript
from .whisper import Progress, WhisperOptions, WhisperResult, transcribe_file
from .youtube import SourceError, YouTubeClient, choose_caption_track, is_url

log = logging.getLogger("vid2md")

SUBTITLE_SUFFIXES = {".srt", ".vtt"}
Transcriber = Callable[[Path, WhisperOptions, "Progress | None"], WhisperResult]

WHISPER_LIMITS = (
    "음성 인식 모델의 추정 결과입니다. 고유명사와 영어 용어는 틀릴 수 있습니다.",
    "화자를 구분하지 않습니다.",
)
AUTO_CAPTION_LIMITS = (
    "YouTube 자동 생성 자막(음성 인식)입니다. 고유명사와 영어 용어는 틀릴 수 있습니다.",
    "화자를 구분하지 않습니다.",
)
# Whisper's VAD drops silence, so a short tail without speech is normal.
UNCOVERED_TAIL_TOLERANCE = 60.0


@dataclass(frozen=True)
class Options:
    force_whisper: bool = False
    whisper: WhisperOptions = WhisperOptions()
    diarize: bool = False  # label speakers (needs hf_token); implies speech recognition
    num_speakers: int | None = None  # known number of speakers, if any
    denoise: bool = False  # remove background music/noise before recognition
    hf_token: str | None = None


@dataclass(frozen=True)
class Enhancers:
    """The heavy optional steps, replaceable in tests."""

    diarize: Callable[..., list] = diarize_mod.diarize
    isolate_voice: Callable[..., Path] = denoise_mod.isolate_voice


DIARIZED_LIMIT = "화자 구분은 목소리로 자동 추정한 것입니다. 목소리가 비슷하거나 겹치면 틀릴 수 있습니다."
DENOISED_LIMIT = "배경음을 제거한 목소리로 인식했습니다."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_coverage(transcript: Transcript) -> Transcript:
    """Say so when the transcript stops well before the media ends: unknown, not 'nothing said'."""
    duration = transcript.source.duration
    covered = transcript.covered_until
    if duration and duration - covered > UNCOVERED_TAIL_TOLERANCE:
        note = (
            f"{format_timestamp(covered)} 이후 {format_timestamp(duration)}까지는 전사된 내용이 없습니다 "
            "(무음일 수도, 인식 누락일 수도 있습니다)."
        )
        return replace(transcript, limitations=transcript.limitations + (note,))
    return transcript


def _whisper_transcript(
    audio: Path, source: SourceInfo, options: Options, transcriber: Transcriber, progress: Progress | None,
    enhancers: Enhancers = Enhancers(),
) -> Transcript:
    limits = list(WHISPER_LIMITS)
    with tempfile.TemporaryDirectory(prefix="vid2md-audio-") as work:
        # The optional steps never cost the transcript: if one cannot run, it is skipped with a note.
        if options.denoise:
            log.info("배경음을 제거하는 중입니다")
            try:
                audio = enhancers.isolate_voice(audio, Path(work) / "voice.wav", device=options.whisper.device)
                limits.append(DENOISED_LIMIT)
            except denoise_mod.DenoiseError as error:
                log.warning("배경음 제거를 건너뜁니다: %s", error)
                limits.append(f"배경음 제거를 건너뛰었습니다: {error}")
        whisper = replace(options.whisper, word_timestamps=options.whisper.word_timestamps or options.diarize)
        result = transcriber(audio, whisper, progress)
        segments = list(result.segments)
        if options.diarize:
            log.info("화자를 구분하는 중입니다")
            try:
                turns = enhancers.diarize(audio, options.hf_token, num_speakers=options.num_speakers,
                                          device=options.whisper.device)
                segments = diarize_mod.assign_speakers(segments, turns)
                limits = [item for item in limits if item != "화자를 구분하지 않습니다."] + [DIARIZED_LIMIT]
            except diarize_mod.DiarizationError as error:
                log.warning("화자 구분을 건너뜁니다: %s", error)
                limits.append(f"화자 구분을 건너뛰었습니다: {error}")
    if source.duration is None:
        source = replace(source, duration=result.duration)
    return Transcript(
        source=source,
        method="whisper",
        confidence="inferred",
        provider=result.provider,
        language=options.whisper.language,
        segments=tuple(segments),
        limitations=tuple(limits),
        hints=tuple(getattr(result, "hints", ()) or ()),
    )


def from_youtube(
    url: str,
    options: Options,
    client: YouTubeClient,
    transcriber: Transcriber,
    progress: Progress | None,
    enhancers: Enhancers = Enhancers(),
) -> Transcript:
    info = client.extract_info(url)
    source = SourceInfo(
        kind="youtube",
        locator=info.get("webpage_url") or url,
        title=info.get("title") or info.get("id") or "untitled",
        duration=info.get("duration"),
        video_id=info.get("id"),
    )
    lang = options.whisper.language
    # Captions carry no speaker information, so speaker labelling needs speech recognition.
    track = None if options.force_whisper or options.diarize else choose_caption_track(info, lang)
    if track is not None:
        segments: list[Segment] = client.fetch_captions(track)
        if segments:
            manual = track.kind == "manual"
            log.info("%s 자막(%s)을 사용합니다", "업로드된" if manual else "자동 생성", track.lang)
            return Transcript(
                source=source,
                method="captions-manual" if manual else "captions-auto",
                confidence="observed" if manual else "inferred",
                provider=f"yt-dlp {client.version()} ({track.kind} captions: {track.lang})",
                language=lang,
                segments=tuple(segments),
                limitations=() if manual else AUTO_CAPTION_LIMITS,
            )
        log.warning("자막 트랙(%s)이 비어 있어 음성 인식으로 전환합니다", track.lang)
    else:
        log.info("쓸 만한 %s 자막이 없어 오디오를 받아 음성 인식합니다", lang)

    with tempfile.TemporaryDirectory(prefix="vid2md-") as work:
        audio = client.download_audio(url, Path(work))
        return _whisper_transcript(audio, source, options, transcriber, progress, enhancers)


def from_file(path: Path, options: Options, transcriber: Transcriber, progress: Progress | None,
              enhancers: Enhancers = Enhancers()) -> Transcript:
    if not path.is_file():
        raise SourceError(f"파일을 찾을 수 없습니다: {path}")
    source = SourceInfo(kind="file", locator=str(path), title=path.stem, sha256=sha256_file(path))
    if path.suffix.lower() in SUBTITLE_SUFFIXES:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        return Transcript(
            source=source,
            method="subtitle-file",
            confidence="observed",
            provider=f"subtitle file ({path.suffix.lower().lstrip('.')})",
            language=options.whisper.language,
            segments=tuple(parse_subtitles(text)),
            limitations=("자막 파일의 내용을 그대로 옮겼습니다. 자막 자체의 정확도는 확인하지 않았습니다.",),
        )
    return _whisper_transcript(path, source, options, transcriber, progress, enhancers)


def build_transcript(
    source: str,
    options: Options = Options(),
    *,
    client: YouTubeClient | None = None,
    transcriber: Transcriber = transcribe_file,
    progress: Progress | None = None,
    enhancers: Enhancers = Enhancers(),
) -> Transcript:
    source = source.strip().strip('"')
    if is_url(source):
        transcript = from_youtube(source, options, client or YouTubeClient(), transcriber, progress, enhancers)
    else:
        transcript = from_file(Path(source).expanduser().resolve(), options, transcriber, progress, enhancers)
    return _with_coverage(transcript)

