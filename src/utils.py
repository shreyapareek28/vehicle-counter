"""
Shared helpers used by every module.
"""
from __future__ import annotations

import os
from datetime import datetime

# Vehicle class names we care about. Works with the COCO-pretrained model and
# with custom models fine-tuned on Indian traffic (add your own class names here).
VEHICLE_NAMES = {
    "bicycle", "car", "motorcycle", "bus", "truck",
    "auto-rickshaw", "autorickshaw", "e-rickshaw", "rickshaw", "van", "tempo",
}

# BGR colours per class (fallback colour is used for unknown classes)
CLASS_COLORS = {
    "car": (255, 140, 0),
    "motorcycle": (0, 200, 255),
    "bus": (60, 60, 230),
    "truck": (0, 180, 90),
    "bicycle": (200, 80, 200),
}
DEFAULT_COLOR = (180, 180, 180)


def color_for(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


def parse_line(text: str, width: int, height: int):
    """Parse 'x1,y1,x2,y2' given as fractions (0-1) of the frame into pixels."""
    try:
        x1, y1, x2, y2 = [float(v) for v in text.split(",")]
    except ValueError as exc:
        raise ValueError("--line must look like 0.05,0.6,0.95,0.6") from exc
    return (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))


def make_run_dir(base: str, name: str | None = None) -> str:
    name = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path
