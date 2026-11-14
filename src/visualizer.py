"""
MODULE 3a - VISUALISATION       (Owner: Member 3)
-------------------------------------------------
Draws everything the viewer sees: boxes, IDs, trails, the counting line and
a live statistics panel.
"""
from __future__ import annotations

import cv2
import numpy as np

from .utils import color_for


class Visualizer:
    def __init__(self, line_p1, line_p2, show_trails: bool = True):
        self.p1, self.p2 = line_p1, line_p2
        self.show_trails = show_trails
        self._flash = 0          # frames left for line "flash" after a count

    def flash_line(self, frames: int = 6):
        self._flash = frames

    def draw(self, frame: np.ndarray, detections, history, counter, fps: float) -> np.ndarray:
        out = frame.copy()

        # counting line
        line_color = (0, 255, 255) if self._flash > 0 else (0, 0, 255)
        cv2.line(out, self.p1, self.p2, line_color, 3)
        self._flash = max(0, self._flash - 1)

        # vehicles
        for d in detections:
            color = color_for(d.cls_name)
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.cls_name} #{d.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.show_trails:
                pts = history.trail(d.track_id)
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(out, a, b, color, 2)

        self._draw_panel(out, counter, fps)
        return out

    def _draw_panel(self, img, counter, fps):
        rows = [("VEHICLE COUNTER", None)]
        for cls, d in sorted(counter.counts.items()):
            rows.append((f"{cls:<11} IN:{d['IN']:<3} OUT:{d['OUT']:<3}", color_for(cls)))
        dirs = counter.totals_by_direction()
        rows.append((f"TOTAL {counter.total}  (IN {dirs['IN']} / OUT {dirs['OUT']})", (0, 255, 255)))
        rows.append((f"FPS {fps:.1f}", (200, 200, 200)))

        line_h, pad = 22, 10
        w, h = 285, pad * 2 + line_h * len(rows)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        for i, (text, col) in enumerate(rows):
            y = 10 + pad + (i + 1) * line_h - 6
            cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if i else 0.6, col or (255, 255, 255), 1, cv2.LINE_AA)
