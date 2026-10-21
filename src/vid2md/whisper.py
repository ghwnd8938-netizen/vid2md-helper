"""Speech recognition with faster-whisper (decodes audio/video via bundled PyAV; no ffmpeg needed)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .segments import Segment, Word

log = logging.getLogger("vid2md.whisper")

Progress = Callable[[float, float], None]  # (seconds transcribed, total seconds)


@dataclass(frozen=True)
class WhisperOptions:
    model: str = "large-v3-turbo"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto -> int8 on CPU, float16 on CUDA
    language: str = "ko"
    prompt: str | None = None  # glossary / style hint, e.g. English terms used in the talk
    # Greedy decoding. Measured on two 10-minute sets of Korean speech with reference text, on 2 CPUs: beam 5 left out
    # whole sentences (12% and 16% of characters wrong) where greedy kept them all (7% and 9%), and greedy is 7-14% faster.
    beam_size: int = 1
    # Candidates sampled when a window is retried at a higher temperature (faster-whisper's default is 5). The full
    # large-v3 keeps a long cache per candidate: five at once took a 60-minute recording to 4.6 GiB, over a 4 GiB server.
    best_of: int = 5
    word_timestamps: bool = False  # needed to split sentences between speakers
    # (word, pack) pairs from the whole dictionary: when given, a quick first listen picks the hints.
    candidates: tuple[tuple[str, str], ...] = ()
    # Other workers may take some of a long recording's ten-minute parts (see Share).
    share: "Share | None" = field(default=None, compare=False, repr=False)


class Share(Protocol):
    """Hands the ten-minute parts of a long recording to other idle workers. This worker offers them, then does
    whichever part it is given next until none is left, and finally collects the parts the others did."""

    def offer(self, spans: list[tuple[float, float]], prompt: str | None) -> bool: ...  # False: do them all here

    def next(self) -> int | None: ...  # a part to do here (waits while others finish theirs); None once all are in

    def progress(self, index: int, fraction: float) -> None: ...

    def done(self, index: int, segments: list[Segment]) -> None: ...

    def segments(self) -> list[Segment]: ...  # every part's, in order


@dataclass(frozen=True)
class WhisperResult:
    segments: list[Segment]
    duration: float
    provider: str
    hints: tuple[str, ...] = ()  # glossary words given to the model (only when picked automatically)


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - any failure means "no usable CUDA"
        return False


# Whisper's decoder holds 448 tokens: the prompt (glossary hint + up to 5 control tokens) plus the text it
# writes for a 30-second window. Dense Korean speech can need over 200 tokens of text, so the hint is
# capped below half, cut at a term boundary, and the text gets the rest (max_new_tokens). Going over
# 448 fails the whole job ("No position encodings are defined for positions >= 448"); giving the text less
# than the rest drops the end of dense windows (whole sentences went missing with a fixed 240 and no hint).
CONTEXT_TOKENS = 448
HINT_TOKENS = 200
CONTROL_TOKENS = 8  # sot_prev, sot, language, task, no_timestamps, with room to spare


def fit_prompt(prompt: str | None, encode: Callable[[str], list[int]] | None, limit: int = HINT_TOKENS) -> str | None:
    """Keep the leading comma-separated terms of a glossary prompt that fit in the model's prompt budget."""
    if not prompt or encode is None:
        return prompt
    kept: list[str] = []
    for term in (t.strip() for t in prompt.split(",")):
        if not term:
            continue
        if len(encode(" " + ", ".join(kept + [term]))) > limit:
            dropped = prompt.count(",") + 1 - len(kept)
            log.info("용어 힌트가 길어 앞쪽 %d개만 씁니다 (%d개 제외)", len(kept), dropped)
            break
        kept.append(term)
    return ", ".join(kept) or None


def new_tokens(prompt: str | None, encode: Callable[[str], list[int]] | None) -> int:
    """Room left for a window's text after the hint (fit_prompt has already cut the hint to HINT_TOKENS)."""
    if not prompt:
        hint = 0
    elif encode is None:
        hint = HINT_TOKENS  # can't count: assume the longest hint
    else:
        hint = len(encode(" " + prompt.strip()))
    return CONTEXT_TOKENS - hint - CONTROL_TOKENS


def _encoder(model) -> Callable[[str], list[int]] | None:
    tokenizer = getattr(model, "hf_tokenizer", None)
    if tokenizer is None:
        return None
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


SAMPLE_RATE = 16000
SAMPLE_PARTS = 6  # parts of the recording listened to first when picking hints
SAMPLE_SECONDS = 60  # length of each part

# Long recordings are recognised in parts: faster-whisper computes the features of all the audio it is
# given at once, and for an hour of speech that alone outgrows a 3 GiB server. Ten minutes at a time
# stays well under 2 GB. Each cut goes to the quietest moment near the part's end, so no word is split.
PART_SECONDS = 600
CUT_SEARCH_SECONDS = 15  # how far from the planned cut to look for a quiet moment, either way
SHARE_MIN_SECONDS = 1200  # shorter recordings are not worth another worker downloading the sound
# Shared recordings go out in five-minute parts: two workers on 25 minutes split 10:15 with ten-minute parts
# (the last one short), but close to even with five-minute ones, and nobody waits long for the last part.
SHARE_PART_SECONDS = 300


def _parts(audio, seconds: int | None = None) -> list[tuple[int, int]]:
    """(start, end) sample ranges of about `seconds` (PART_SECONDS), cut where it is quietest."""
    import numpy as np

    size, search, frame = (seconds or PART_SECONDS) * SAMPLE_RATE, CUT_SEARCH_SECONDS * SAMPLE_RATE, SAMPLE_RATE // 10
    bounds = [0]
    while len(audio) - bounds[-1] > size + search:
        low = bounds[-1] + size - search
        frames = np.asarray(audio[low:low + 2 * search], dtype="float32").reshape(-1, frame)
        quietest = int(np.argmin((frames ** 2).mean(axis=1)))
        bounds.append(low + quietest * frame + frame // 2)
    bounds.append(len(audio))
    return list(zip(bounds, bounds[1:]))


def _sample_text(model, audio, options: WhisperOptions) -> str:
    """A rough, quick transcript of a few spread-out parts (fast settings, the person's own words as hints)."""
    total = len(audio) / SAMPLE_RATE
    if total <= SAMPLE_PARTS * SAMPLE_SECONDS:
        parts = [audio]
    else:
        size = SAMPLE_SECONDS * SAMPLE_RATE
        starts = [int((k + 0.5) / SAMPLE_PARTS * len(audio) - size / 2) for k in range(SAMPLE_PARTS)]
        parts = [audio[max(0, s):max(0, s) + size] for s in starts]
    text = []
    for part in parts:
        pieces, _ = model.transcribe(part, language=options.language, beam_size=1, vad_filter=True,
                                     condition_on_previous_text=False, hotwords=options.prompt or None)
        text.extend(p.text for p in pieces)
    return " ".join(text)


def _auto_prompt(model, audio, options: WhisperOptions) -> str | None:
    from .glossary import pick_hints

    log.info("용어 힌트를 고르는 중입니다 (영상 일부를 먼저 듣습니다)")
    picked = pick_hints(_sample_text(model, audio, options), options.candidates)
    own = [t.strip() for t in (options.prompt or "").split(",") if t.strip()]
    words = own + [w for w in picked if w not in own]
    return ", ".join(words) or None


def _model(options: WhisperOptions, device: str):
    from faster_whisper import WhisperModel

    compute_type = options.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    log.info("모델 로드: %s (%s/%s) — 처음에는 모델 다운로드로 시간이 걸립니다", options.model, device, compute_type)
    return WhisperModel(options.model, device=device, compute_type=compute_type), compute_type


def _span(model, audio, start: int, end: int, options: WhisperOptions, prompt: str | None, room: int,
          on_piece: Callable[[float], None] | None) -> list[Segment]:
    """Recognise audio[start:end] (samples); times in the result are from the start of the recording."""
    offset = start / SAMPLE_RATE
    pieces, _ = model.transcribe(
        audio[start:end],
        language=options.language,
        beam_size=options.beam_size,
        best_of=options.best_of,
        vad_filter=True,
        # Long recordings can loop on repeated text when each window is
        # conditioned on the previous one; independent windows are safer.
        condition_on_previous_text=False,
        # Hotwords go in front of every 30-second window (an initial prompt would reach only the first
        # one, and on top of the hotwords it would overflow the decoder there).
        hotwords=prompt,
        max_new_tokens=room,
        word_timestamps=options.word_timestamps,
    )
    segments: list[Segment] = []
    for piece in pieces:
        text = piece.text.strip()
        if text:
            words = tuple(Word(w.start + offset, w.end + offset, w.word) for w in (piece.words or ()))
            segments.append(Segment(piece.start + offset, piece.end + offset, text, words=words))
        if on_piece:
            on_piece(piece.end)
    return segments


def _run(path: Path, options: WhisperOptions, device: str, progress: Progress | None) -> WhisperResult:
    import faster_whisper

    model, compute_type = _model(options, device)
    audio = faster_whisper.decode_audio(str(path), sampling_rate=SAMPLE_RATE)  # once: hints and every part
    prompt = options.prompt
    if options.candidates:
        prompt = _auto_prompt(model, audio, options)
    encode = _encoder(model)
    prompt = fit_prompt(prompt, encode)
    room = new_tokens(prompt, encode)
    total = len(audio) / SAMPLE_RATE
    share = options.share
    if share is not None and total >= SHARE_MIN_SECONDS:
        bounds = _parts(audio, SHARE_PART_SECONDS)
        if not share.offer([(start / SAMPLE_RATE, end / SAMPLE_RATE) for start, end in bounds], prompt):
            share, bounds = None, _parts(audio)
    else:
        share, bounds = None, _parts(audio)
    if share is not None:
        while (index := share.next()) is not None:
            start, end = bounds[index]
            length = (end - start) / SAMPLE_RATE
            report = lambda t, k=index, n=length: share.progress(k, min(t / n, 1.0))  # noqa: E731
            share.done(index, _span(model, audio, start, end, options, prompt, room, report))
        segments = share.segments()
    else:
        segments = []
        for start, end in bounds:
            offset = start / SAMPLE_RATE
            report = (lambda t, o=offset: progress(o + t, total)) if progress else None
            segments.extend(_span(model, audio, start, end, options, prompt, room, report))
    provider = f"faster-whisper {faster_whisper.__version__} ({options.model}, {device}/{compute_type})"
    hints = tuple(prompt.split(", ")) if options.candidates and prompt else ()
    return WhisperResult(segments, total, provider, hints)


def transcribe_span(path: Path, options: WhisperOptions, start: float, end: float,
                    progress: Callable[[float], None] | None = None) -> list[Segment]:
    """One part of a recording that another worker shared (seconds; its hint words are final already)."""
    import faster_whisper

    device = options.device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    model, _ = _model(options, device)
    audio = faster_whisper.decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    first, last = round(start * SAMPLE_RATE), min(round(end * SAMPLE_RATE), len(audio))
    encode = _encoder(model)
    prompt = fit_prompt(options.prompt, encode)
    length = max((last - first) / SAMPLE_RATE, 1e-9)
    report = (lambda t: progress(min(t / length, 1.0))) if progress else None
    return _span(model, audio, first, last, options, prompt, new_tokens(prompt, encode), report)


def transcribe_file(path: Path, options: WhisperOptions, progress: Progress | None = None) -> WhisperResult:
    device = options.device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    try:
        return _run(path, options, device, progress)
    except RuntimeError as error:
        # On Windows a visible GPU without the cuBLAS/cuDNN DLLs fails only
        # once decoding starts. With device=auto, fall back to CPU.
        if options.device == "auto" and device == "cuda":
            log.warning("CUDA 실행 실패, CPU로 다시 시도합니다: %s", error)
            return _run(path, options, "cpu", progress)
        raise


def stderr_progress() -> Progress:
    last = {"pct": -1}

    def report(done: float, total: float) -> None:
        if total <= 0:
            return
        pct = min(int(done / total * 100), 100)
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"\r전사 중... {pct:3d}%", end="", file=sys.stderr, flush=True)
            if pct == 100:
                print(file=sys.stderr)

    return report
