"""Latin vs manuscript-script cues without a required OCR runtime."""

from __future__ import annotations

from .textmap import TextLine


def latin_score(line: TextLine, page_box: list[int], image_h: int) -> float:
    """Higher means more likely English / catalog / pencil-style filler."""
    x1, y1, x2, y2 = line.bbox
    px, py, pw, ph = page_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0

    score = 0.0
    outside = x2 < px or x1 > px + pw or y2 < py or y1 > py + ph
    if outside:
        score += 0.5

    near_photo_edge = y1 < 0.07 * image_h or y2 > 0.93 * image_h
    near_page_edge = y2 < py + 0.035 * ph or y1 > py + 0.965 * ph
    if near_photo_edge:
        score += 0.2
    if near_page_edge and line.density < 0.22:
        score += 0.2

    aspect = width / float(height)
    cc_per_width = line.n_cc / max(1.0, width / max(line.mean_height, 1.0))
    if cc_per_width > 1.8 and line.mean_height < 0.04 * max(ph, 1):
        score += 0.2
    if 6.0 < aspect < 50 and line.mean_height < 0.04 * ph and line.density < 0.25:
        score += 0.15
    rel_cx = (cx - px) / max(1, pw)
    if 0.15 < rel_cx < 0.85 and width > 0.3 * pw and height < 0.055 * max(image_h, 1) and line.density < 0.28:
        score += 0.2

    # Dense, tall strokes on the writing surface are manuscript body, not catalog type.
    on_page = (y1 >= py - 8) and (y2 <= py + ph + 8)
    if on_page and line.density >= 0.22 and line.mean_height >= 0.028 * ph:
        score -= 0.45
    return float(max(0.0, min(1.0, score)))


def is_latin_line(line: TextLine, page_box: list[int], image_h: int) -> bool:
    return latin_score(line, page_box, image_h) >= 0.5
