"""
color_identification.py
-----------------------
HSV-based colour-strip detection on a ladle ROI.

Each ladle carries a set of colour-coded alumina rings arranged
vertically.  Reading the rings top-to-bottom and mapping each colour
to a digit produces the ladle's unique ID number.

Default mapping  (tunable via config):
    Green  → 1
    Blue   → 2
    Brown  → 3
"""

from __future__ import annotations

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional

# ── Load config ──────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def _load_color_config(cfg_path: Path = _CONFIG_PATH) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["color_detection"]

_color_cfg = _load_color_config()

# Build runtime structures from config
COLOR_HSV_RANGES: dict[str, list[dict]] = {}
COLOR_TO_DIGIT:   dict[str, int]        = {}

for _color_name, _color_data in _color_cfg["colors"].items():
    COLOR_TO_DIGIT[_color_name] = _color_data["digit"]
    COLOR_HSV_RANGES[_color_name] = [
        {
            "lower": np.array(r["lower"], dtype=np.uint8),
            "upper": np.array(r["upper"], dtype=np.uint8),
        }
        for r in _color_data["ranges"]
    ]

MIN_STRIP_AREA           = _color_cfg["min_strip_area"]
COLOR_PERCENTAGE_THRESHOLD = _color_cfg["percentage_threshold"]


# ── Core functions ────────────────────────────────────────────────────────────

def _build_color_mask(hsv_roi: np.ndarray, color_name: str) -> np.ndarray:
    """Return a cleaned binary mask for one colour in an HSV ROI.

    Uses an elliptical structuring element for morphological ops —
    better suited to the circular cross-section of alumina rings than
    a square kernel.
    """
    ranges = COLOR_HSV_RANGES[color_name]
    mask: Optional[np.ndarray] = None
    for r in ranges:
        m = cv2.inRange(hsv_roi, r["lower"], r["upper"])
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_strips(
    bgr_roi:   np.ndarray,
    roi_x1:    int,
    roi_y1:    int,
) -> list[tuple]:
    """
    Detect colour strips inside a ladle bounding-box crop.

    Args:
        bgr_roi:  BGR image crop of the detected ladle.
        roi_x1:   Left edge of the crop in the original frame (for global coords).
        roi_y1:   Top edge of the crop in the original frame.

    Returns:
        List of (y_center_global, color_name, rect_in_frame, color_pct, area, contours_global)
        sorted top-to-bottom (ascending y).

        contours_global is a list of contour arrays already translated to
        full-frame coordinates — pass directly to cv2.drawContours() for
        visualising exactly which pixels triggered each strip detection.
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return []

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    strips: list[tuple] = []

    for color_name in COLOR_HSV_RANGES:
        mask = _build_color_mask(hsv, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_STRIP_AREA:
                continue

            rect = cv2.minAreaRect(cnt)          # ((cx, cy), (w, h), angle)
            (cx, cy), (cw, ch), angle = rect
            cw = max(cw, 1.0)
            ch = max(ch, 1.0)
            color_pct = (area / (cw * ch)) * 100.0

            if color_pct >= COLOR_PERCENTAGE_THRESHOLD:
                y_global    = cy + roi_y1
                rect_global = (
                    (int(cx + roi_x1), int(cy + roi_y1)),
                    (int(cw), int(ch)),
                    angle,
                )
                # Translate contour to full-frame coords for overlay drawing
                cnt_global = cnt + np.array([[roi_x1, roi_y1]])
                strips.append((y_global, color_name, rect_global, color_pct, area, [cnt_global]))

    strips.sort(key=lambda s: s[0])   # top-to-bottom
    return strips


def strips_to_ladle_id(strips: list[tuple]) -> tuple[Optional[int], list[str]]:
    """
    Convert an ordered list of detected strips to a ladle ID.

    Reads each strip's colour, maps it to its digit, concatenates the
    digits, and parses the result as an integer.

    Returns:
        (ladle_id, digit_list)
        ladle_id is None if no valid digits were found.
    """
    if not strips:
        return None, []

    # Strip tuple is (y, color, rect, pct, area, contours) — unpack with *rest
    digits = [
        str(COLOR_TO_DIGIT[color])
        for _, color, *_rest in strips
        if color in COLOR_TO_DIGIT
    ]

    if digits:
        try:
            return int("".join(digits)), digits
        except ValueError:
            pass

    return None, digits
