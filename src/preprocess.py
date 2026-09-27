"""Contrast enhancement for faded / uneven manuscript pages."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def enhance_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, amount: float = 1.5, radius: int = 1) -> np.ndarray:
    """Sharpen image to mitigate blur using unsharp masking."""
    if radius < 1:
        radius = 1
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def suppress_bleedthrough(gray: np.ndarray) -> np.ndarray:
    """Reduce ghost text from reverse side using dual-threshold subtraction."""
    win = max(15, (min(gray.shape) // 30) | 1)
    if win % 2 == 0:
        win += 1
    # Heavy threshold (main ink) - higher k = stricter
    thresh_heavy = threshold_sauvola(gray, window_size=win, k=0.35)
    # Light threshold (catches bleed-through) - lower k = more sensitive
    thresh_light = threshold_sauvola(gray, window_size=win, k=0.1)
    # Bleed-through = pixels that are ink at light threshold but not at heavy
    bleed = ((gray < thresh_light) & (gray >= thresh_heavy)).astype(np.uint8) * 255
    # Dilate bleed mask to cover stroke widths
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bleed = cv2.dilate(bleed, kernel)
    # Subtract half the bleed-through intensity
    cleaned = cv2.subtract(gray, cv2.divide(bleed, 2))
    return cleaned


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink-as-white binary mask using Sauvola (handles uneven lighting)."""
    win = max(15, (min(gray.shape) // 30) | 1)
    if win % 2 == 0:
        win += 1
    thresh = threshold_sauvola(gray, window_size=win, k=0.2)
    ink = (gray < thresh).astype(np.uint8) * 255
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, open_k)
