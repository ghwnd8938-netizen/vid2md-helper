"""Write the Markdown file safely on Windows and elsewhere."""

from __future__ import annotations

import re
from pathlib import Path

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(title: str, max_length: int = 120) -> str:
    """Turn a video title into a valid Windows file name (without extension)."""
    name = " ".join(_INVALID.sub(" ", title).split())[:max_length].rstrip(" .")
    if not name:
        name = "transcript"
    if name.split(".")[0].upper() in _RESERVED:
        name = f"_{name}"
    return name


def write_markdown(text: str, path: Path, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"이미 파일이 있습니다: {path} (덮어쓰려면 --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: the Windows default code page (cp949) cannot hold every character.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)
    return path
