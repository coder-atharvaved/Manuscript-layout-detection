"""Connected components and text-line grouping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextLine:
    bbox: list[int]  # x1,y1,x2,y2
    area: int
    n_cc: int
    mean_height: float
    mean_width: float
    mean_solidity: float
    density: float


def extract_components(binary: np.ndarray, page_box: list[int]) -> np.ndarray:
    """Return stats array for ink blobs inside the page, filtered by size."""
    x, y, w, h = page_box
    roi = np.zeros_like(binary)
    roi[y : y + h, x : x + w] = binary[y : y + h, x : x + w]
    num, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num <= 1:
        return np.zeros((0, 5), dtype=np.int32)

    page_area = max(1, w * h)
    min_area = max(8, page_area // 25000)
    max_area = page_area // 3
    keep = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw < 2 or bh < 2:
            continue
        if bw > 0.95 * w and bh > 0.95 * h:
            continue
        keep.append(stats[i])
    if not keep:
        return np.zeros((0, 5), dtype=np.int32)
    return np.stack(keep, axis=0)


def group_lines(stats: np.ndarray, page_box: list[int]) -> list[TextLine]:
    if stats.size == 0:
        return []

    boxes = []
    for row in stats:
        x, y, w, h, area = (int(v) for v in row[:5])
        boxes.append((x, y, x + w, y + h, area, h, w))

    boxes.sort(key=lambda b: (b[1], b[0]))
    heights = [b[5] for b in boxes]
    med_h = float(np.median(heights)) if heights else 12.0
    y_tol = max(4.0, 0.65 * med_h)

    clusters: list[list[tuple]] = []
    for box in boxes:
        placed = False
        cy = (box[1] + box[3]) / 2.0
        for cluster in clusters:
            cys = [((c[1] + c[3]) / 2.0) for c in cluster]
            if abs(cy - float(np.mean(cys))) <= y_tol:
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    px, py, pw, ph = page_box
    lines: list[TextLine] = []
    for cluster in clusters:
        x1 = min(c[0] for c in cluster)
        y1 = min(c[1] for c in cluster)
        x2 = max(c[2] for c in cluster)
        y2 = max(c[3] for c in cluster)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        area = sum(c[4] for c in cluster)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mean_h = float(np.mean([c[5] for c in cluster]))
        mean_w = float(np.mean([c[6] for c in cluster]))
        density = area / float(bw * bh)
        lines.append(
            TextLine(
                bbox=[x1, y1, x2, y2],
                area=int(area),
                n_cc=len(cluster),
                mean_height=mean_h,
                mean_width=mean_w,
                mean_solidity=min(1.0, density * 1.8),
                density=density,
            )
        )

    # Drop lines that are almost the full page frame.
    filtered = []
    for line in lines:
        lw = line.bbox[2] - line.bbox[0]
        lh = line.bbox[3] - line.bbox[1]
        if lw > 0.98 * pw and lh > 0.4 * ph:
            continue
        filtered.append(line)
    return filtered
