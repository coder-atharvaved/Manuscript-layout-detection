"""Isolate the manuscript page/leaf from scanner background."""

from __future__ import annotations

import cv2
import numpy as np


def find_page_box(bgr: np.ndarray) -> list[int]:
    """Return [x, y, w, h] of the writing surface in original coordinates."""
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return [0, 0, w, h]

    mask = _foreground_mask(bgr)
    coverage = float(mask.mean()) / 255.0
    if coverage > 0.92:
        margin_x = max(2, w // 200)
        margin_y = max(2, h // 200)
        return [margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, w, h]

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.08 * w * h:
        return [0, 0, w, h]

    x, y, bw, bh = cv2.boundingRect(contour)
    pad_x = max(2, bw // 80)
    pad_y = max(2, bh // 80)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)
    return [int(x), int(y), int(bw), int(bh)]


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    band = max(4, min(h, w) // 40)
    corners = np.concatenate(
        [
            bgr[:band, :band].reshape(-1, 3),
            bgr[:band, w - band :].reshape(-1, 3),
            bgr[h - band :, :band].reshape(-1, 3),
            bgr[h - band :, w - band :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
    flat = dist.astype(np.uint8) if dist.max() <= 255 else cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If the leaf is lighter than a dark table, Otsu may invert; pick the more central class.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    cy, cx = h / 2.0, w / 2.0
    on = np.hypot(xs.mean() - cx, ys.mean() - cy)
    off_ys, off_xs = np.where(mask == 0)
    if len(off_xs):
        off = np.hypot(off_xs.mean() - cx, off_ys.mean() - cy)
        if off < on:
            mask = cv2.bitwise_not(mask)
    return mask


def deskew_binary(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect and correct skew using minAreaRect on ink pixels.
    Returns (deskewed_binary, angle_degrees). Angle is 0 if no significant skew.
    """
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return binary, 0.0

    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    # minAreaRect returns angle in [-90, 0]; normalize to [-45, 45]
    if angle < -45:
        angle += 90

    if abs(angle) < 0.5:
        return binary, 0.0

    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle
