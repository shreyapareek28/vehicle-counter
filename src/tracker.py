"""
MODULE 2a - TRACK HISTORY       (Owner: Member 2)
-------------------------------------------------
Keeps the recent trajectory of every vehicle ID. Used for
  * drawing motion trails
  * cleaning up IDs that have disappeared (memory-safe on long videos)
"""
from __future__ import annotations

from collections import defaultdict, deque


class TrackHistory:
    def __init__(self, max_len: int = 30, max_age: int = 90):
        self.max_len = max_len
        self.max_age = max_age              # frames before an ID is forgotten
        self.points = defaultdict(lambda: deque(maxlen=max_len))
        self.last_seen: dict[int, int] = {}

    def update(self, track_id: int, point: tuple[float, float], frame_idx: int):
        self.points[track_id].append((int(point[0]), int(point[1])))
        self.last_seen[track_id] = frame_idx

    def trail(self, track_id: int):
        return list(self.points[track_id])

    def cleanup(self, frame_idx: int):
        stale = [t for t, f in self.last_seen.items() if frame_idx - f > self.max_age]
        for t in stale:
            self.points.pop(t, None)
            self.last_seen.pop(t, None)
