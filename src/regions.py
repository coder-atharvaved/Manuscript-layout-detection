"""Turn text lines into labeled layout regions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import (
    Region,
    box_area,
    box_center,
    clamp_confidence,
    clip_box,
    clip_box_to_page,
    iou,
    merge_boxes,
)
from .script import is_latin_line, latin_score
from .textmap import TextLine


def split_into_columns(lines: list[TextLine], page_box: list[int]) -> list[list[TextLine]]:
    """Split lines into columns based on horizontal gaps.
    Returns list of column-line-groups.
    """
    if not lines:
        return []

    px, py, pw, ph = page_box
    # Compute median character width from line mean_width
    widths = [ln.mean_width for ln in lines if ln.mean_width > 0]
    if not widths:
        return [lines]
    median_char_w = float(np.median(widths))
    # Column gap threshold: 2.5x median char width (adjustable)
    col_gap_thresh = max(median_char_w * 2.5, 0.08 * pw)

    # Sort lines by x-center
    lines_with_cx = [(ln, (ln.bbox[0] + ln.bbox[2]) / 2.0) for ln in lines]
    lines_with_cx.sort(key=lambda x: x[1])

    columns: list[list[TextLine]] = []
    current_col: list[TextLine] = [lines_with_cx[0][0]]
    last_cx = lines_with_cx[0][1]

    for ln, cx in lines_with_cx[1:]:
        if cx - last_cx > col_gap_thresh:
            columns.append(current_col)
            current_col = [ln]
        else:
            current_col.append(ln)
        last_cx = cx
    columns.append(current_col)

    # Filter out tiny columns (noise)
    columns = [c for c in columns if len(c) >= 2 or box_area(c[0].bbox) > 200]
    return columns if columns else [lines]


def lines_to_regions(
    lines: list[TextLine],
    page_box: list[int],
    image_w: int,
    image_h: int,
    border_fillers: list[list[int]] | None = None,
) -> list[Region]:
    labeled: list[tuple[str, list[int], float, bool]] = []

    body_candidates: list[TextLine] = []
    for line in lines:
        if is_latin_line(line, page_box, image_h):
            conf = clamp_confidence(0.5 + 0.45 * latin_score(line, page_box, image_h))
            labeled.append(("filler", line.bbox, conf, False))
        else:
            body_candidates.append(line)

    for extra in border_fillers or []:
        labeled.append(("filler", extra, 0.88, False))

    if not body_candidates:
        labeled.append(("main_text", page_box_as_xyxy(page_box), 0.4, True))
    else:
        # Split body candidates into columns for multi-column layout support
        columns = split_into_columns(body_candidates, page_box)

        px, py, pw, ph = page_box
        filler_boxes = [b for lab, b, _, _ in labeled if lab == "filler"]

        for col_lines in columns:
            if not col_lines:
                continue
            # Find core for this column
            core = merge_boxes(ln.bbox for ln in col_lines) or page_box_as_xyxy(page_box)
            core = _shrink_from_fillers(core, filler_boxes, page_box)

            labeled.append(("main_text", core, 0.9, True))

            # Classify header/footer/side_text relative to this column's core
            for ln in col_lines:
                if iou(ln.bbox, core) > 0.12:
                    continue
                if _contained(ln.bbox, core, pad=6):
                    continue
                if box_area(ln.bbox) < 80:
                    continue
                cx, cy = box_center(ln.bbox)
                width_frac = (ln.bbox[2] - ln.bbox[0]) / max(1, pw)
                height_frac = (ln.bbox[3] - ln.bbox[1]) / max(1, ph)
                rel_x = (cx - px) / max(1, pw)

                if ln.bbox[3] <= core[1] + 4 and width_frac < 0.7:
                    labeled.append(("header", ln.bbox, 0.74, True))
                elif ln.bbox[1] >= core[3] - 4 and width_frac < 0.7:
                    labeled.append(("footer", ln.bbox, 0.74, True))
                elif width_frac < 0.25 and height_frac > 0.06 and (rel_x < 0.20 or rel_x > 0.80):
                    if ln.bbox[2] <= core[0] + 30 or ln.bbox[0] >= core[2] - 30:
                        labeled.append(("side_text", ln.bbox, 0.73, True))

    regions: list[Region] = []
    seen: set[tuple] = set()
    rid = 1
    for label, bbox, conf, clip_page in labeled:
        box = clip_box(bbox, image_w, image_h)
        if clip_page:
            box = clip_box_to_page(box, page_box)
            box = clip_box(box, image_w, image_h)
        if box_area(box) < 40:
            continue
        if label == "side_text" and (box[2] - box[0]) < 12:
            continue
        key = (label, tuple(box))
        if key in seen:
            continue
        seen.add(key)
        regions.append(Region(id=rid, label=label, bbox=box, confidence=clamp_confidence(conf)))
        rid += 1
    return _merge_same_label(regions, image_w, image_h)


def page_box_as_xyxy(page_box: list[int]) -> list[int]:
    x, y, w, h = page_box
    return [x, y, x + w, y + h]


def _contained(inner: list[int], outer: list[int], pad: int = 0) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[2] <= outer[2] + pad
        and inner[3] <= outer[3] + pad
    )


def _shrink_from_fillers(core: list[int], fillers: list[list[int]], page_box: list[int]) -> list[int]:
    """Keep main_text from swallowing catalog banners that overlap the core."""
    x1, y1, x2, y2 = core
    px, py, pw, ph = page_box
    core_h = max(1, y2 - y1)
    for fb in fillers:
        fx1, fy1, fx2, fy2 = fb
        horizontal = not (fx2 < x1 or fx1 > x2)
        if not horizontal:
            continue
        if fy2 <= y1 + 0.25 * core_h:
            y1 = max(y1, fy2 + 2)
        if fy1 >= y2 - 0.25 * core_h:
            y2 = min(y2, fy1 - 2)
    y1 = max(py, y1)
    y2 = min(py + ph, y2)
    if y2 - y1 < 20:
        return core
    return [x1, y1, x2, y2]


def _merge_same_label(regions: list[Region], image_w: int, image_h: int) -> list[Region]:
    groups: dict[str, list[Region]] = defaultdict(list)
    for r in regions:
        groups[r.label].append(r)

    merged: list[Region] = []
    rid = 1
    for label, items in groups.items():
        if label == "main_text":
            box = merge_boxes(r.bbox for r in items)
            conf = max(r.confidence for r in items)
            if box:
                merged.append(
                    Region(id=rid, label=label, bbox=clip_box(box, image_w, image_h), confidence=conf)
                )
                rid += 1
            continue

        used = [False] * len(items)
        for i, a in enumerate(items):
            if used[i]:
                continue
            cluster = [a]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(items):
                    if used[j]:
                        continue
                    if any(iou(c.bbox, b.bbox) > 0.05 or _horizontally_aligned(c.bbox, b.bbox) for c in cluster):
                        cluster.append(b)
                        used[j] = True
                        changed = True
            box = merge_boxes(r.bbox for r in cluster)
            conf = max(r.confidence for r in cluster)
            if box:
                merged.append(
                    Region(id=rid, label=label, bbox=clip_box(box, image_w, image_h), confidence=conf)
                )
                rid += 1
    merged.sort(key=lambda r: (r.bbox[1], r.bbox[0], r.label))
    for i, r in enumerate(merged, start=1):
        r.id = i
    return merged


def _horizontally_aligned(a: list[int], b: list[int], gap: int = 24) -> bool:
    vertical_overlap = not (a[3] < b[1] - gap or b[3] < a[1] - gap)
    return vertical_overlap
