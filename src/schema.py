"""Shared types and JSON schema helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


LABELS = ("header", "footer", "main_text", "side_text", "filler")

# Colors are BGR for OpenCV overlays.
LABEL_COLORS = {
    "header": (210, 160, 40),
    "footer": (180, 80, 180),
    "main_text": (70, 180, 70),
    "side_text": (40, 140, 230),
    "filler": (50, 50, 220),
}


@dataclass
class Region:
    id: int
    label: str
    bbox: list[int]  # [x1, y1, x2, y2] in original image pixels
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageResult:
    image: str
    width: int
    height: int
    page_box: list[int]  # [x, y, w, h]
    regions: list[Region] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "page_box": self.page_box,
            "regions": [r.to_dict() for r in self.regions],
        }


def clip_box(box: Sequence[int], width: int, height: int) -> list[int]:
    """Clip [x1, y1, x2, y2] to image bounds and order corners."""
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def clip_box_to_page(box: Sequence[int], page_box: Sequence[int]) -> list[int]:
    """Keep a box inside the detected page rectangle [x, y, w, h]."""
    px, py, pw, ph = (int(v) for v in page_box)
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(px, min(px + pw, x1))
    y1 = max(py, min(py + ph, y1))
    x2 = max(px, min(px + pw, x2))
    y2 = max(py, min(py + ph, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = (int(v) for v in box)
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: Sequence[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = (int(v) for v in box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = (int(v) for v in a)
    bx1, by1, bx2, by2 = (int(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def merge_boxes(boxes: Iterable[Sequence[int]]) -> list[int] | None:
    boxes = [list(b) for b in boxes if box_area(b) > 0]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def clamp_confidence(value: float) -> float:
    return max(0.05, min(0.99, float(value)))
