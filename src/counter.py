"""
MODULE 2b - LINE COUNTER        
-------------------------------------------------
Counts each tracked vehicle exactly once when it crosses a virtual line,
separately for every vehicle type and for both directions.

How the crossing test works
  side(P) = sign of the 2-D cross product of the line vector and (P - line start)
  A vehicle crossed when side() flips between two consecutive frames AND the
  crossing point lies within the line segment.
"""
from __future__ import annotations

from collections import defaultdict

Point = tuple[float, float]


class LineCounter:
    DIRECTIONS = ("IN", "OUT")

    def __init__(self, p1: Point, p2: Point, flip: bool = False):
        self.p1, self.p2 = p1, p2
        self.flip = flip
        self._last_side: dict[int, int] = {}
        self._counted: set[int] = set()
        self.counts = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self.events: list[dict] = []

    # ---- geometry -------------------------------------------------------
    def _side(self, pt: Point) -> int:
        (x1, y1), (x2, y2) = self.p1, self.p2
        cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        return (cross > 0) - (cross < 0)

    def _within_segment(self, pt: Point, margin: float = 0.05) -> bool:
        (x1, y1), (x2, y2) = self.p1, self.p2
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return False
        u = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / length_sq
        return -margin <= u <= 1 + margin

    # ---- main API -------------------------------------------------------
    def update(self, track_id: int, cls_name: str, point: Point,
               frame_idx: int, time_s: float) -> dict | None:
        """Feed one tracked vehicle. Returns an event dict if it just got counted."""
        side = self._side(point)
        prev = self._last_side.get(track_id)
        if side != 0:
            self._last_side[track_id] = side

        if (track_id in self._counted or prev is None or side == 0
                or side == prev or not self._within_segment(point)):
            return None

        direction = "IN" if (prev < 0 < side) else "OUT"
        if self.flip:
            direction = "OUT" if direction == "IN" else "IN"

        self._counted.add(track_id)
        self.counts[cls_name][direction] += 1
        event = {"frame": frame_idx, "time_s": round(time_s, 2),
                 "track_id": track_id, "vehicle_type": cls_name,
                 "direction": direction}
        self.events.append(event)
        return event

    # ---- summaries ------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self._counted)

    def totals_by_class(self) -> dict[str, int]:
        return {c: d["IN"] + d["OUT"] for c, d in self.counts.items()}

    def totals_by_direction(self) -> dict[str, int]:
        return {k: sum(d[k] for d in self.counts.values()) for k in self.DIRECTIONS}
