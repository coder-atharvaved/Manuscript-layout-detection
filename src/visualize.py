"""Draw labeled boxes onto a copy of the input image."""

from __future__ import annotations

import cv2

from .schema import LABEL_COLORS, Region


def annotate(bgr: np.ndarray, regions: list[Region], page_box: list[int] | None = None) -> np.ndarray:
    canvas = bgr.copy()
    if page_box:
        x, y, w, h = page_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)

    for region in regions:
        x1, y1, x2, y2 = region.bbox
        color = LABEL_COLORS.get(region.label, (0, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{region.label} {region.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 6)
        cv2.rectangle(canvas, (x1, max(0, ty - th - 4)), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _legend(canvas, regions)
    return canvas


def _legend(canvas: np.ndarray, regions: list[Region]) -> None:
    labels = []
    for r in regions:
        if r.label not in labels:
            labels.append(r.label)
    if not labels:
        return
    x, y = 12, 18
    for label in labels:
        color = LABEL_COLORS.get(label, (0, 255, 255))
        cv2.rectangle(canvas, (x, y - 10), (x + 14, y + 4), color, -1)
        cv2.putText(canvas, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 18
