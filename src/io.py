"""Image discovery and JSON writing. Original files are never modified."""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    files = [
        p
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No images found under {path}")
    return files


def ensure_output_dir(output_path: str | Path) -> Path:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
