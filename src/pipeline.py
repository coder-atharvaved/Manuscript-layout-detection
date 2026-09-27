"""End-to-end detection for one image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .page import find_page_box, deskew_binary
from .preprocess import binarize, enhance_gray, suppress_bleedthrough, unsharp_mask
from .regions import lines_to_regions
from .schema import PageResult, box_area, clip_box
from .textmap import extract_components, group_lines
from .visualize import annotate


def detect_image(image_path: str | Path) -> tuple[PageResult, np.ndarray]:
    path = Path(image_path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = bgr.shape[:2]
    page_box = find_page_box(bgr)
    raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Preprocessing pipeline: enhance -> sharpen -> suppress bleed-through -> binarize -> deskew
    gray = enhance_gray(bgr)
    gray = unsharp_mask(gray)           # Mitigate blur
    gray = suppress_bleedthrough(gray)  # Reduce bleed-through from reverse side
    binary = binarize(gray)
    binary, _ = deskew_binary(binary)   # Correct skew

    stats = extract_components(binary, page_box)
    lines = group_lines(stats, page_box)
    fillers = _border_filler_boxes(raw_gray, page_box)
    regions = lines_to_regions(lines, page_box, w, h, border_fillers=fillers)
    result = PageResult(
        image=path.name,
        width=w,
        height=h,
        page_box=page_box,
        regions=regions,
    )
    overlay = annotate(bgr, regions, page_box)
    return result, overlay


def _border_filler_boxes(gray: np.ndarray, page_box: list[int]) -> list[list[int]]:
    """Catalog / credit lines: outside the leaf, or photo top/bottom on full-frame pages."""
    h, w = gray.shape[:2]
    px, py, pw, ph = page_box
    full_frame = ph >= 0.92 * h and pw >= 0.92 * w
    if full_frame:
        band = max(16, int(0.09 * h))
        zones = [(0, band), (max(0, h - band), h)]
    else:
        zones = []
        if py > 8:
            zones.append((0, py))
        bottom = py + ph
        if h - bottom > 8:
            zones.append((bottom, h))

    boxes: list[list[int]] = []
    for y0, y1 in zones:
        strip = gray[y0:y1]
        if strip.size == 0:
            continue
        _, ink = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if ink.mean() > 160:
            ink = cv2.bitwise_not(ink)
        num, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        min_area = max(8, (w * max(1, y1 - y0)) // 12000)
        cluster = []
        for i in range(1, num):
            x, y, bw, bh, area = (int(v) for v in stats[i][:5])
            y += y0
            if area < min_area or bh > 0.08 * h or bw > 0.98 * w:
                continue
            cy = y + bh / 2.0
            if full_frame and not (cy < 0.08 * h or cy > 0.935 * h):
                continue
            cluster.append([x, y, x + bw, y + bh])
        if cluster:
            cxs = [0.5 * (b[0] + b[2]) for b in cluster]
            med = sorted(cxs)[len(cxs) // 2]
            cluster = [b for b in cluster if abs(0.5 * (b[0] + b[2]) - med) < 0.38 * w]
        if cluster:
            box = clip_box(
                [
                    min(b[0] for b in cluster),
                    min(b[1] for b in cluster),
                    max(b[2] for b in cluster),
                    max(b[3] for b in cluster),
                ],
                w,
                h,
            )
            if box_area(box) > 50 and (box[2] - box[0]) > 30:
                boxes.append(box)
    return boxes
