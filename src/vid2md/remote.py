"""vid2md-worker: take jobs from the Google Apps Script dashboard and transcribe them here.

The worker only makes outbound HTTPS requests (to the Apps Script web app and to
the Google Drive API), so a home PC or a server works without opening any port.

With the trusted key (WORKER_SECRET) it gets a Drive token and does everything. With the
external server key (WORKER_SECRET_LIMITED) it gets no token: only the jobs the owners let
go out, and for an upload only its MP3, in pieces through the dashboard.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import logging
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .audio import extract_mp3, media_seconds
from .markdown import render
from .pipeline import Options, build_transcript
from .segments import Segment, format_timestamp
from .slides import read_slides
from .whisper import WhisperOptions, transcribe_span

log = logging.getLogger("vid2md.worker")

DRIVE_API = "https://www.googleapis.com/drive/v3"
CONFIG_PATH = Path.home() / ".vid2md-worker.json"
PROGRESS_EVERY = 15.0  # seconds between progress reports
WAIT_EVERY = 10.0  # seconds between asks while other workers finish their parts
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
LINGER_EVERY = 15.0  # seconds between asks while a 서브 서버 waits for parts that may still come back
# A runner started for parts waits this long for them: it may start before the recording is split (the dashboard
# starts runners as soon as it knows a recording is long), and a part whose helper went quiet comes back later.
PARTS_LINGER = 240.0
PART_MARGIN = 3.0  # seconds of sound taken on each side of a part fetched by its bytes (MP3 frames start mid-way)
PARALLEL_PIECES = 3  # pieces of a sound fetched at once through the dashboard
SAFE = {"safe": True}
KEY_NAMES = {"full": "전체(WORKER_SECRET)", "limited": "제한(WORKER_SECRET_LIMITED)", "helper": "서브(WORKER_SECRET_HELPER)"}  # a log line that carries only ids, part numbers and counts: fine for a public log


class QuietLogs(logging.Filter):
    """A 서브 서버's log is public (a GitHub runner of a public repository). Every line not marked SAFE is replaced
    by its logger and level, without arguments or traceback, so no title, hint word or transcript can reach it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "safe", False):
            record.msg, record.args = "(%s %s: 내용 숨김)", (record.name, record.levelname)
            record.exc_info, record.exc_text, record.stack_info = None, None, None
        return True


def quiet_logs(handler: logging.Handler | None = None) -> None:
    """QuietLogs on the given handler (else every handler of the root logger); warnings go through it too."""
    logging.captureWarnings(True)
    for each in [handler] if handler else logging.getLogger().handlers:
        each.addFilter(QuietLogs())


class RemoteError(RuntimeError):
    """The dashboard answered with ok=false, or could not be reached."""


class Dashboard:
    """Client for the Apps Script web app (JSON posted as text/plain, redirects followed)."""

    def __init__(self, url: str, secret: str, drive_api: str = DRIVE_API, timeout: float = 120, name: str = ""):
        self.url, self.secret, self.drive_api, self.timeout = url, secret, drive_api, timeout
        # Tells two servers with the same key apart (which one holds which part of a shared recording).
        # VID2MD_WORKER_NAME marks a kind of worker (Colab's notebook sets "colab:…", which the dashboard notices).
        self.name = name or os.environ.get("VID2MD_WORKER_NAME", "").strip() or f"{socket.gethostname()}:{os.getpid()}"

    def call(self, action: str, **params: Any) -> dict[str, Any]:
        body = json.dumps({"action": action, "secret": self.secret, "worker": self.name, **params}).encode()
        request = urllib.request.Request(self.url, data=body, headers={"content-type": "text/plain;charset=utf-8"})
        try:
            # Apps Script answers POST with a 302 to the result; urllib follows it as a GET.
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise RemoteError(f"{action}: {error}") from error
        if not data.get("ok"):
            raise RemoteError(f"{action}: {data.get('error', 'unknown error')}")
        return data

    def upload(self, path: Path, folder_id: str, name: str, token: str) -> str:
        """Put a file into a Drive folder (resumable upload, sent in one request); returns its id."""
        size = path.stat().st_size
        upload_api = self.drive_api.replace("/drive/v3", "/upload/drive/v3")
        start = urllib.request.Request(
            f"{upload_api}/files?uploadType=resumable&fields=id",
            data=json.dumps({"name": name, "parents": [folder_id]}).encode(),
            headers={"authorization": f"Bearer {token}", "content-type": "application/json; charset=UTF-8",
                     "x-upload-content-length": str(size)},
        )
        with urllib.request.urlopen(start, timeout=self.timeout) as response:
            session = response.headers["location"]
        return self.put(session, path, token)

    def put(self, session: str, path: Path, token: str | None = None) -> str:
        """Send a file to a resumable upload session (one request); returns the new file's id.
        A session the dashboard opened needs no token: its address only takes this one file."""
        headers = {"content-length": str(path.stat().st_size)}
        if token:
            headers["authorization"] = f"Bearer {token}"
        with path.open("rb") as body:
            put = urllib.request.Request(session, data=body, method="PUT", headers=headers)
            with urllib.request.urlopen(put, timeout=self.timeout) as response:
                return json.loads(response.read() or b"{}").get("id", "")

    def download(self, file_id: str, token: str, out, byte_range: tuple[int, int] | None = None) -> None:
        headers = {"authorization": f"Bearer {token}"}
        if byte_range:
            headers["range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        request = urllib.request.Request(f"{self.drive_api}/files/{file_id}?alt=media", headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            while block := response.read(1 << 20):
                out.write(block)


Transcribe = Callable[..., Any]


def job_options(job: dict[str, Any], whisper: WhisperOptions) -> Options:
    """Pipeline options for a job; the keys for the optional steps come from this machine's environment."""
    picked = job.get("options") or {}
    speakers = picked.get("speakers")
    return Options(
        force_whisper=bool(job.get("force_whisper")),
        whisper=replace(whisper, prompt=job.get("prompt") or whisper.prompt,
                        # "자동 선택": the dashboard sends the whole dictionary; the recording picks from it.
                        candidates=tuple((str(w), str(p)) for w, p in job.get("glossary") or ())),
        diarize=bool(picked.get("diarize")),
        num_speakers=int(speakers) if speakers else None,
        denoise=bool(picked.get("denoise")),
        hf_token=os.environ.get("HF_TOKEN") or None,
    )


def segments_json(segments) -> str:
    """Compact [start, end, text, speaker] rows; the dashboard builds SRT/VTT/TXT from them."""
    rows = [[round(s.start, 2), round(s.end, 2), s.text, s.speaker] for s in segments]
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


class PartShare:
    """whisper.Share through the dashboard: idle workers take some of this recording's parts while this one
    works through the rest. A part whose worker goes quiet comes back here, so the job never waits on a helper."""

    def __init__(self, dashboard: Dashboard, job_id: str, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, sound: Callable[[], str | None] | None = None):
        self.dashboard, self.job_id, self.clock, self.sleep = dashboard, job_id, clock, sleep
        self.sound = sound  # puts the recording's MP3 in Drive for the 서브 서버s (which never get a video); its id
        self.mine: dict[int, list[Segment]] = {}
        self.theirs: dict[int, list[Segment]] = {}
        self.count = 0
        self.last = -PROGRESS_EVERY

    def offer(self, spans: list[tuple[float, float]], prompt: str | None) -> bool:
        audio_id = self.sound() if self.sound else None
        extra = {"audio_id": audio_id} if audio_id else {}
        try:
            self.dashboard.call("worker_split", id=self.job_id, parts=[[round(a, 3), round(b, 3)] for a, b in spans],
                                prompt=prompt or "", **extra)
        except RemoteError as error:
            log.warning("나눠 처리하지 않고 혼자 처리합니다: %s", error)
            return False
        self.count = len(spans)
        log.info("%d개 부분으로 나눴습니다. 쉬고 있는 서버가 함께 처리합니다.", len(spans))
        return True

    def next(self) -> int | None:
        while True:
            answer = self.dashboard.call("worker_next", id=self.job_id)
            if answer.get("part") is not None:
                return int(answer["part"])
            if answer.get("done"):
                for index, rows in (answer.get("parts") or {}).items():
                    self.theirs[int(index)] = [Segment(float(r[0]), float(r[1]), str(r[2])) for r in json.loads(rows)]
                return None
            self.sleep(WAIT_EVERY)  # others are finishing theirs; asking again also keeps the job alive

    def progress(self, index: int, fraction: float) -> None:
        now = self.clock()
        if now - self.last >= PROGRESS_EVERY:
            self.last = now
            try:
                self.dashboard.call("worker_progress", id=self.job_id, part=index, progress=round(fraction, 3))
            except RemoteError as error:
                log.warning("진행 상황 보고 실패: %s", error)

    def done(self, index: int, segments: list[Segment]) -> None:
        self.mine[index] = segments

    def segments(self) -> list[Segment]:
        parts = {**self.theirs, **self.mine}
        helped = len(self.theirs)
        if helped:
            log.info("다른 서버가 %d개 부분을 처리했습니다", helped)
        return [segment for index in sorted(parts) for segment in parts[index]]


class Worker:
    def __init__(self, dashboard: Dashboard, whisper: WhisperOptions, transcribe: Transcribe = build_transcript,
                 clock: Callable[[], float] = time.monotonic, span: Callable[..., list[Segment]] = transcribe_span,
                 sleep: Callable[[float], None] = time.sleep, hq_model: str = "",
                 slides: Callable[[Path], str] = read_slides, helper: bool = False, parts_only: bool = False):
        self.dashboard, self.whisper, self.transcribe, self.clock = dashboard, whisper, transcribe, clock
        self.span, self.sleep, self.slides = span, sleep, slides
        self.helper = helper  # a 서브 서버: parts only (the key allows nothing else), and its log names jobs by id
        self.parts_only = parts_only or helper  # started only to help with the parts of a long recording
        # 고품질: the full large-v3 model for the jobs that ask for it (a local path if one is set, else its name).
        self.hq_model = hq_model or os.environ.get("VID2MD_MODEL_HQ", "") or "large-v3"
        self.held: tuple[str, Path, str] | None = None  # (job id, its sound, temp dir) kept for that job's next part

    def run_once(self) -> bool:
        """Claim and process one job (or one part of another worker's long recording). False when there was none."""
        job = self.dashboard.call("worker_claim", **({"parts_only": True} if self.parts_only else {})).get("job")
        if not job or job.get("part") is None or (self.held and self.held[0] != job["id"]):
            self._drop_held()
        if not job:
            return False
        part = job.get("part")
        label = self._label(job)
        if part is not None:
            log.info("다른 서버의 작업을 돕습니다: %s (%d번째 부분)", label, int(part["index"]) + 1, extra=SAFE)
        else:
            log.info("작업 시작: %s (%s)", job["title"], job["kind"])
        extra = {"part": int(part["index"])} if part is not None else {}
        try:
            if part is not None:
                self._process_part(job, part)
            else:
                self._process(job)
        except KeyboardInterrupt:
            # Ctrl+C, or SIGTERM when the platform restarts or moves this worker: the job is not at fault, so it
            # goes straight back to the queue for another worker (instead of waiting STALE_MINUTES there).
            log.warning("중지 신호를 받아 작업을 대기열로 돌려놓습니다: %s", label, extra=SAFE)
            try:
                self.dashboard.call("worker_release", id=job["id"], **extra)
            except RemoteError:
                log.exception("작업을 돌려놓지 못했습니다 (30분 뒤 자동으로 대기열로 돌아갑니다)")
            self._drop_held()
            raise
        except Exception as error:  # noqa: BLE001 - every failure is reported to the dashboard
            if self.helper:
                log.error("작업 실패: %s (%s)", label, type(error).__name__, extra=SAFE)
            else:
                log.exception("작업 실패: %s", label)
            try:
                self.dashboard.call("worker_fail", id=job["id"], error=str(error) or type(error).__name__, **extra)
            except RemoteError:
                log.exception("실패 보고도 실패했습니다")
        return True

    def _label(self, job: dict[str, Any]) -> str:
        return f"작업 #{str(job['id'])[:8]}" if self.helper else str(job["title"])

    def _drop_held(self) -> None:
        if self.held:
            shutil.rmtree(self.held[2], ignore_errors=True)
            self.held = None

    def _process_part(self, job: dict[str, Any], part: dict[str, Any]) -> None:
        """One part of a long recording that another worker is transcribing: the same sound, from start to end."""
        index = int(part["index"])
        last = {"t": -PROGRESS_EVERY}

        def report(fraction: float) -> None:
            now = self.clock()
            if now - last["t"] >= PROGRESS_EVERY:
                last["t"] = now
                try:
                    self.dashboard.call("worker_progress", id=job["id"], part=index, progress=round(fraction, 3))
                except RemoteError as error:
                    log.warning("진행 상황 보고 실패: %s", error)

        start, end = float(part["start"]), float(part["end"])
        whisper = replace(self._whisper_for(job), prompt=job.get("prompt") or None, candidates=(), share=None)
        rate = int(job.get("cbr") or 0)
        if rate and job["kind"] == "upload":
            # An MP3 the workers made (constant bit rate): only the bytes of this part, a little either side.
            self._drop_held()
            with tempfile.TemporaryDirectory(prefix="vid2md-part-") as work:
                first = max(0.0, start - PART_MARGIN)
                window = (int(first * rate), int((end + PART_MARGIN) * rate))
                sound = self._fetch_window(job, window, Path(work) / "part.mp3")
                offset = window[0] / rate
                report(0.0)
                segments = [Segment(s.start + offset, s.end + offset, s.text, s.speaker)
                            for s in self.span(sound, whisper, start - offset, end - offset, report)]
        else:
            if not self.held:
                work = tempfile.mkdtemp(prefix="vid2md-part-")
                try:
                    self.held = (job["id"], self._fetch_sound(job, Path(work)), work)
                except BaseException:
                    shutil.rmtree(work, ignore_errors=True)
                    raise
            report(0.0)
            segments = self.span(self.held[1], whisper, start, end, report)
        self.dashboard.call("worker_part_done", id=job["id"], part=index, segments=segments_json(segments))
        log.info("부분 처리 완료: %s (%d번째)", self._label(job), index + 1, extra=SAFE)

    def _whisper_for(self, job: dict[str, Any]) -> WhisperOptions:
        if (job.get("options") or {}).get("hq"):
            return replace(self.whisper, model=self.hq_model, best_of=1)  # one retry candidate: see WhisperOptions
        return self.whisper

    def _read_slides(self, job: dict[str, Any], media: Path, report: Callable[..., None]) -> tuple[str, ...]:
        """화면 글자: read the video's slides and hand the text in; returns the words to use as hint candidates.
        A failure only costs the hints, never the job."""
        if media.suffix.lower() in AUDIO_SUFFIXES:
            return ()
        report("화면 글자 읽는 중", force=True)
        try:
            text = self.slides(media)
            if not text:
                return ()
            return tuple(self.dashboard.call("worker_slides", id=job["id"], text=text).get("words") or ())
        except Exception as error:  # noqa: BLE001
            log.warning("화면 글자를 읽지 못했습니다: %s", error)
            return ()

    def _fetch_window(self, job: dict[str, Any], window: tuple[int, int], target: Path) -> Path:
        """Bytes window[0]..window[1] of the job's MP3: from Drive with a trusted key, else through the dashboard."""
        with target.open("wb") as out:
            if job.get("access_token"):
                self.dashboard.download(job["chunks"][0], job["access_token"], out, byte_range=window)
            else:
                count, index = 1, 0
                while index < count:
                    answer = self.dashboard.call("worker_audio", id=job["id"], index=index, **{"from": window[0], "to": window[1]})
                    count = int(answer.get("chunks") or 1)
                    out.write(base64.b64decode(answer["data"]))
                    index += 1
        return target

    def _fetch_sound(self, job: dict[str, Any], work: Path) -> Path:
        if job["kind"] == "youtube":
            from .youtube import YouTubeClient

            return YouTubeClient().download_audio(job["source"], work)
        suffix = ".mp3" if job.get("input") == "mp3" else Path(job["source"]).suffix.lower() or ".bin"
        media = work / f"media{suffix}"
        if job.get("access_token"):
            self._download_pieces(job, media)
        else:
            self._download_audio(job, media)
        return media

    def _process(self, job: dict[str, Any]) -> None:
        last = {"t": -PROGRESS_EVERY}

        def report(stage: str, progress: float = 0.0, force: bool = False) -> None:
            now = self.clock()
            if force or now - last["t"] >= PROGRESS_EVERY:
                last["t"] = now
                try:
                    self.dashboard.call("worker_progress", id=job["id"], stage=stage, progress=round(progress, 3))
                except RemoteError as error:
                    log.warning("진행 상황 보고 실패: %s", error)

        options = job_options(job, self._whisper_for(job))
        place: dict[str, Path] = {}
        early: dict[str, str] = {}

        def sound() -> str | None:
            # The 서브 서버s never get a video: a long upload's MP3 goes to Drive before it is shared (the transcript
            # then keeps that one). Only a trusted worker saves into Drive this way; an MP3 already made is used as it is.
            audio = job.get("audio") or {}
            if "media" not in place or not audio.get("folder_id") or job.get("input") == "mp3":
                return None
            report("음성 파일 만드는 중", force=True)
            early.update(self._keep_audio(audio, place["media"], place["work"] / "audio.mp3"))
            return early.get("audio_id")

        share = None
        if not (options.diarize or options.denoise):
            # Speaker labels and voice isolation need the whole recording in one place; the rest can be shared.
            share = PartShare(self.dashboard, job["id"], self.clock, self.sleep, sound)
            options = replace(options, whisper=replace(options.whisper, share=share))
        on_progress = lambda done, total: report("음성 인식 중", done / total if total else 0)  # noqa: E731
        kept: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix="vid2md-job-") as work:
            if job["kind"] == "upload":
                mp3 = job.get("input") == "mp3"  # the MP3 made earlier, not the video
                report("음성 받는 중" if mp3 else "영상 받는 중", force=True)
                suffix = ".mp3" if mp3 else Path(job["source"]).suffix.lower() or ".bin"
                media = Path(work) / f"media{suffix}"
                if job.get("access_token"):
                    self._download_pieces(job, media)
                else:  # the external server: through the dashboard, this job only
                    self._download_audio(job, media)
                place.update(media=media, work=Path(work))
                if share:
                    self._announce_length(job, media)
                if (job.get("options") or {}).get("slides") and job.get("access_token") and not mp3:
                    words = self._read_slides(job, media, report)
                    if words:
                        whisper = options.whisper
                        extra = tuple((w, "화면") for w in words)
                        options = replace(options, whisper=replace(whisper, candidates=whisper.candidates + extra))
                if job.get("prepare"):
                    # The external server may take this one: keep the original, make the MP3, hand it back.
                    kept = self._keep(job, media, Path(work), report)
                    if kept.get("audio_id"):
                        self.dashboard.call("worker_prepared", id=job["id"], **kept)
                        log.info("외부 서버로 넘겼습니다: %s", job["title"])
                        return
                report("음성 인식 준비 중", force=True)
                transcript = self.transcribe(str(media), options, progress=on_progress)
                transcript = replace(transcript, source=replace(transcript.source, title=job["title"], locator=job["source"]))
                if not job.get("prepare"):
                    kept = self._keep(job, media, Path(work), report, audio_kept=early or None)
            else:
                report("영상 정보 확인 중", force=True)
                transcript = self.transcribe(job["source"], options, progress=on_progress)

        duration = transcript.source.duration
        self.dashboard.call(
            "worker_complete",
            id=job["id"],
            markdown=render(transcript),
            title=transcript.source.title,
            method=transcript.method,
            confidence=transcript.confidence,
            duration=format_timestamp(duration) if duration else "",
            segments=segments_json(transcript.segments),
            **kept,
        )
        log.info("작업 완료: %s", transcript.source.title)

    def _announce_length(self, job: dict[str, Any], media: Path) -> None:
        """A recording that will be shared: its length lets the dashboard start the helpers before it is split."""
        seconds = media_seconds(media)
        if not seconds:
            return
        try:
            self.dashboard.call("worker_progress", id=job["id"], stage="음성 인식 준비 중", progress=0, seconds=round(seconds))
        except RemoteError as error:
            log.warning("진행 상황 보고 실패: %s", error)

    def _keep(self, job: dict[str, Any], media: Path, work: Path, report: Callable[..., None],
              audio_kept: dict[str, str] | None = None) -> dict[str, str]:
        kept: dict[str, str] = {}
        if job.get("original"):
            report("원본 영상 보관 중", force=True)
            kept = self._keep_original(job["original"], media)
        if audio_kept is not None:  # made already, for the 서브 서버s
            kept.update(audio_kept)
        elif job.get("audio"):
            report("음성 파일 만드는 중", force=True)
            kept.update(self._keep_audio(job["audio"], media, work / "audio.mp3"))
        return kept

    def _keep_original(self, original: dict[str, str], media: Path) -> dict[str, str]:
        """Save the whole upload as one playable file in Drive. A failure does not fail the job:
        the dashboard then keeps the uploaded pieces, so nothing is lost."""
        try:
            if original.get("upload_url"):  # the external server: an address for this file only; the dashboard finds it
                self.dashboard.put(original["upload_url"], media)
                return {}
            token = self.dashboard.call("worker_token")["access_token"]  # transcription may outlast the old one
            return {"original_id": self.dashboard.upload(media, original["folder_id"], original["name"], token)}
        except Exception as error:  # noqa: BLE001
            log.warning("원본 영상을 보관하지 못했습니다: %s", error)
            return {"original_error": str(error) or type(error).__name__}

    def _keep_audio(self, audio: dict[str, str], media: Path, target: Path) -> dict[str, str]:
        """An MP3 of the sound next to the transcript. Like the original, a failure does not fail the job."""
        try:
            extract_mp3(media, target)
            if audio.get("upload_url"):
                self.dashboard.put(audio["upload_url"], target)
                return {}
            token = self.dashboard.call("worker_token")["access_token"]
            return {"audio_id": self.dashboard.upload(target, audio["folder_id"], audio["name"], token)}
        except Exception as error:  # noqa: BLE001
            log.warning("음성 파일을 만들지 못했습니다: %s", error)
            return {"audio_error": str(error) or type(error).__name__}

    def _download_pieces(self, job: dict[str, Any], target: Path) -> None:
        token = job["access_token"]
        with target.open("wb") as out:
            for file_id in job["chunks"]:
                try:
                    self.dashboard.download(file_id, token, out)
                except urllib.error.HTTPError as error:
                    if error.code != 401:
                        raise
                    token = self.dashboard.call("worker_token")["access_token"]  # expired after an hour
                    self.dashboard.download(file_id, token, out)

    def _download_audio(self, job: dict[str, Any], target: Path) -> None:
        """The external server's way in: the job's sound (its MP3 or the uploaded recording) through the dashboard,
        a few pieces at once (each answer takes seconds; the dashboard's limit on answers keeps the pieces small)."""
        from concurrent.futures import ThreadPoolExecutor

        def piece(index: int) -> bytes:
            return base64.b64decode(self.dashboard.call("worker_audio", id=job["id"], index=index)["data"])

        with target.open("wb") as out, ThreadPoolExecutor(PARALLEL_PIECES) as pool:
            for data in pool.map(piece, range(int(job["audio_chunks"]))):
                out.write(data)

    def drain(self, attempts: int = 3, backoff: float = 10.0, linger: float = 0.0) -> int:
        """Process jobs until the queue is empty; returns how many were processed. With linger, keep asking that
        long after the last job (a 서브 서버: a part whose helper went quiet comes back after a while)."""
        processed, failures = 0, 0
        idle_since = self.clock() if linger else 0.0
        while True:
            try:
                if not self.run_once():
                    if not linger or self.clock() - idle_since >= linger:
                        return processed
                    self.sleep(LINGER_EVERY)
                    continue
                processed += 1
                failures = 0
                idle_since = self.clock() if linger else 0.0
            except RemoteError as error:
                failures += 1
                if failures >= attempts:
                    raise
                log.warning("대시보드 연결 문제, %.0f초 뒤 다시: %s", backoff * failures, error)
                time.sleep(backoff * failures)

    def run_forever(self, idle_seconds: float = 20.0, max_backoff: float = 300.0) -> None:
        delay = idle_seconds
        while True:
            try:
                busy = self.run_once()
                delay = idle_seconds
                if busy:
                    continue
            except RemoteError as error:
                log.warning("대시보드 연결 문제, %.0f초 뒤 다시: %s", delay, error)
                delay = min(delay * 2, max_backoff)
            time.sleep(delay)


def load_config(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(path: Path, data: dict[str, str]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def key_is(dashboard: Dashboard, expected: str, attempts: int = 3, wait: float = 10.0) -> bool:
    """A server meant for the limited key refuses to start with the full one (it would get a Drive token).
    If the dashboard cannot be reached, the worker starts anyway and keeps trying as usual."""
    for attempt in range(attempts):
        try:
            key = dashboard.call("worker_ping").get("key")
        except RemoteError as error:
            log.warning("키 종류를 확인하지 못했습니다 (%d/%d): %s", attempt + 1, attempts, error)
            time.sleep(wait)
            continue
        if key == expected:
            return True
        # Only the kinds of key (and the property each comes from) are named: no value, so a public log may show it.
        log.error("이 서버에는 %s 키를 넣어야 하는데 %s 키가 들어 있습니다. 비밀 값을 바꿀 때까지 시작하지 않습니다.",
                  KEY_NAMES.get(expected, expected), KEY_NAMES.get(str(key), str(key)), extra=SAFE)
        return False
    return True


def stop_on_sigterm(*_: Any) -> None:
    """A container is stopped with SIGTERM; treat it like Ctrl+C so a running job is handed back."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vid2md-worker", description="구글 대시보드의 작업을 받아 이 컴퓨터에서 변환합니다.")
    parser.add_argument("--setup", action="store_true", help="대시보드 주소와 작업 서버 비밀 키를 저장")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--once", action="store_true", help="작업 하나만 처리하고 종료")
    parser.add_argument("--drain", action="store_true", help="쌓인 작업을 모두 처리하면 종료 (Colab 등)")
    parser.add_argument("--model", default=None, help="Whisper 모델 (기본: 설정값 또는 large-v3-turbo)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--idle", type=float, default=20.0, help="대기열이 비었을 때 다시 확인할 간격(초)")
    parser.add_argument("--helper", action="store_true",
                        help="서브 서버: 서브 전용 키로 긴 녹음의 부분만 돕고, 로그에는 작업 번호만 남김 (공개 저장소용)")
    parser.add_argument("--parts-only", action="store_true",
                        help="긴 녹음의 부분만 돕고 작업 전체는 가져가지 않음 (비공개 서브 서버용)")
    parser.add_argument("--linger", type=float, default=None,
                        help="할 일이 없을 때 더 기다리는 시간(초). 서브 서버 기본 240, --drain 기본 0")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    if args.helper:
        quiet_logs()
        try:
            return _main(args)
        except Exception as error:  # noqa: BLE001 - a public log gets the kind of error only
            log.error("서브 서버가 멈췄습니다 (%s)", type(error).__name__, extra=SAFE)
            return 1
    return _main(args)


def _main(args: argparse.Namespace) -> int:

    config = load_config(args.config)
    # Environment variables win over the file, so Colab secrets or systemd can supply them.
    for key, env in (("url", "VID2MD_API_URL"), ("secret", "VID2MD_WORKER_SECRET"), ("model", "VID2MD_MODEL")):
        if os.environ.get(env):
            config[key] = os.environ[env].strip()
    if args.setup:
        url = input("Apps Script 웹앱 주소(https://script.google.com/macros/s/.../exec): ").strip()
        secret = getpass.getpass("작업 서버 비밀 키(WORKER_SECRET, 외부 서버면 WORKER_SECRET_LIMITED): ").strip()
        model = input(f"Whisper 모델 [{config.get('model', 'large-v3-turbo')}]: ").strip() or config.get("model", "large-v3-turbo")
        config = {"url": url, "secret": secret, "model": model}
        try:
            Dashboard(url, secret).call("worker_ping")
        except RemoteError as error:
            print(f"연결 확인 실패: {error}", file=sys.stderr)
            return 1
        save_config(args.config, config)
        print(f"저장했습니다: {args.config}\n이제 vid2md-worker 를 실행하세요.")
        return 0
    if not config.get("url") or not config.get("secret"):
        print("먼저 vid2md-worker --setup 으로 대시보드 주소와 비밀 키를 저장하세요 "
              "(또는 VID2MD_API_URL, VID2MD_WORKER_SECRET 환경 변수).", file=sys.stderr)
        return 1

    dashboard = Dashboard(config["url"], config["secret"], drive_api=os.environ.get("VID2MD_DRIVE_API", DRIVE_API))
    expected = os.environ.get("VID2MD_EXPECT_KEY", "").strip() or ("helper" if args.helper else "")
    if expected and not key_is(dashboard, expected):
        return 2
    worker = Worker(
        dashboard,
        WhisperOptions(model=args.model or config.get("model", "large-v3-turbo"), device=args.device),
        helper=args.helper,
        parts_only=args.parts_only,
    )
    signal.signal(signal.SIGTERM, stop_on_sigterm)
    if args.once:
        worker.run_once()
        return 0
    if args.drain or args.helper or args.parts_only:
        linger = args.linger if args.linger is not None else (PARTS_LINGER if args.helper or args.parts_only else 0.0)
        try:
            count = worker.drain(linger=linger) if linger else worker.drain()
        except KeyboardInterrupt:
            log.info("중지했습니다.", extra=SAFE)
            return 0
        except RemoteError as error:
            if args.helper:  # the message may name the dashboard's address
                log.error("대시보드에 연결하지 못했습니다 (%s)", type(error).__name__, extra=SAFE)
            else:
                log.error("대시보드에 연결하지 못했습니다: %s", error)
            return 1
        log.info("대기열이 비었습니다. 처리한 작업: %d개", count, extra=SAFE)
        return 0
    log.info("대시보드에서 작업을 기다립니다 (중지: Ctrl+C)")
    try:
        worker.run_forever(idle_seconds=args.idle)
    except KeyboardInterrupt:
        log.info("중지했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
