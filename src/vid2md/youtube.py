"""YouTube access through yt-dlp: metadata, caption tracks, audio download."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .segments import Segment, parse_subtitles

log = logging.getLogger("vid2md.youtube")

_URL = re.compile(r"^https?://", re.IGNORECASE)
_PREFERRED_EXTS = ("vtt", "srt")


class SourceError(RuntimeError):
    """The source could not be read; the message is meant for the user."""


def is_url(value: str) -> bool:
    return bool(_URL.match(value.strip()))


@dataclass(frozen=True)
class CaptionTrack:
    kind: str  # "manual" | "auto"
    lang: str
    ext: str
    url: str


def _lang_matches(key: str, lang: str) -> bool:
    return key == lang or key.startswith(lang + "-")


def _pick_format(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ext in _PREFERRED_EXTS:
        for entry in entries:
            if entry.get("ext") == ext and entry.get("url"):
                return entry
    return None


def choose_caption_track(info: dict[str, Any], lang: str = "ko") -> CaptionTrack | None:
    """Pick the most trustworthy caption track in `lang`.

    Order: uploaded captions, then the auto-generated track in the video's
    original language. Auto captions in `lang` for a video spoken in another
    language are machine translations, so they are never chosen.
    """
    manual = info.get("subtitles") or {}
    for key in sorted(manual, key=lambda k: (k != lang, k)):
        if _lang_matches(key, lang) and "live_chat" not in key:
            entry = _pick_format(manual[key])
            if entry:
                return CaptionTrack("manual", key, entry["ext"], entry["url"])

    auto = info.get("automatic_captions") or {}
    spoken = (info.get("language") or "").lower()
    candidates = [f"{lang}-orig"]
    if _lang_matches(spoken, lang):
        candidates.append(lang)
    for key in candidates:
        if key in auto:
            entry = _pick_format(auto[key])
            if entry:
                return CaptionTrack("auto", key, entry["ext"], entry["url"])
    return None


def _options(**extra: Any) -> dict[str, Any]:
    # Route yt-dlp output through logging (stderr) so stdout stays clean for
    # the MCP stdio transport.
    return {"quiet": True, "no_warnings": False, "noprogress": True, "logger": log, **extra}


def _friendly(error: Exception) -> SourceError:
    text = str(error)
    if "not a bot" in text or "429" in text:
        return SourceError(
            "YouTube가 요청을 거부했습니다(봇 확인 또는 요청 과다). 잠시 후 다시 시도하고, "
            "`pip install -U \"yt-dlp[default]\"`로 yt-dlp를 최신으로 유지하세요. "
            f"원본 메시지: {text}"
        )
    return SourceError(f"YouTube 정보를 가져오지 못했습니다: {text}")


class YouTubeClient:
    """Thin yt-dlp wrapper; replaceable in tests."""

    def extract_info(self, url: str) -> dict[str, Any]:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError

        try:
            with YoutubeDL(_options(skip_download=True)) as ydl:
                return ydl.sanitize_info(ydl.extract_info(url, download=False))
        except DownloadError as error:
            raise _friendly(error) from error

    def fetch_captions(self, track: CaptionTrack) -> list[Segment]:
        from yt_dlp import YoutubeDL

        with YoutubeDL(_options()) as ydl:
            text = ydl.urlopen(track.url).read().decode("utf-8", errors="replace")
        return parse_subtitles(text)

    def download_audio(self, url: str, directory: Path) -> Path:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError

        options = _options(format="bestaudio/best", outtmpl=str(directory / "%(id)s.%(ext)s"))
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                path = Path(ydl.prepare_filename(info))
        except DownloadError as error:
            raise _friendly(error) from error
        if not path.exists():
            raise SourceError(f"오디오 파일이 생성되지 않았습니다: {path}")
        return path

    @staticmethod
    def version() -> str:
        from yt_dlp.version import __version__

        return __version__
