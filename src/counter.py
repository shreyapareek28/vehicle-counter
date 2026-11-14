
"""
MODULE 2b - IMPROVED LINE COUNTER
-------------------------------------------------
Counts tracked vehicles when their movement trajectory
intersects a virtual counting line.

Features:
- Accurate line-segment intersection
- IN / OUT direction detection
- One count per track ID
- Vehicle-type classification
- Diagonal counting line support
"""

from __future__ import annotations

from collections import defaultdict

Point = tuple[float, float]


class LineCounter:

    DIRECTIONS = ("IN", "OUT")

    def __init__(self, p1: Point, p2: Point, flip: bool = False):

        self.p1 = p1
        self.p2 = p2
        self.flip = flip

        # Store previous valid position of each vehicle
        self._last_point: dict[int, Point] = {}

        # Prevent duplicate counting
        self._counted: set[int] = set()

        # Vehicle counts
        self.counts = defaultdict(
            lambda: {"IN": 0, "OUT": 0}
        )

        self.events: list[dict] = []

    # -------------------------------------------------
    # GEOMETRY
    # -------------------------------------------------

    @staticmethod
    def _cross(a: Point, b: Point) -> float:

        return a[0] * b[1] - a[1] * b[0]

    def _side(self, pt: Point) -> int:

        x1, y1 = self.p1
        x2, y2 = self.p2

        cross = (
            (x2 - x1) * (pt[1] - y1)
            - (y2 - y1) * (pt[0] - x1)
        )

        return (cross > 0) - (cross < 0)

    def _get_intersection(
        self,
        a: Point,
        b: Point,
        margin: float = 0.02
    ) -> Point | None:

        """
        Finds the intersection between:

        1. Vehicle movement segment A -> B
        2. Counting line P1 -> P2

        Returns the intersection point only if it lies
        within the counting line segment.
        """

        rx = b[0] - a[0]
        ry = b[1] - a[1]

        sx = self.p2[0] - self.p1[0]
        sy = self.p2[1] - self.p1[1]

        denominator = self._cross((rx, ry), (sx, sy))

        # Parallel or overlapping segments
        if abs(denominator) < 1e-9:
            return None

        qpx = self.p1[0] - a[0]
        qpy = self.p1[1] - a[1]

        t = self._cross((qpx, qpy), (sx, sy)) / denominator

        u = self._cross((qpx, qpy), (rx, ry)) / denominator

        # Vehicle must actually move across the intersection
        if not 0.0 <= t <= 1.0:
            return None

        # Intersection must lie on the finite counting line
        if not -margin <= u <= 1.0 + margin:
            return None

        intersection_x = a[0] + t * rx
        intersection_y = a[1] + t * ry

        return intersection_x, intersection_y

    # -------------------------------------------------
    # MAIN COUNTING FUNCTION
    # -------------------------------------------------

    def update(
        self,
        track_id: int,
        cls_name: str,
        point: Point,
        frame_idx: int,
        time_s: float
    ) -> dict | None:

        """
        Processes one vehicle in one frame.

        Returns an event dictionary if a vehicle is counted.
        Otherwise returns None.
        """

        # Already counted this vehicle
        if track_id in self._counted:
            return None

        current_side = self._side(point)

        # Ignore points exactly on the line.
        # Keep the last valid position from either side.
        if current_side == 0:
            return None

        previous_point = self._last_point.get(track_id)

        # Save current position
        self._last_point[track_id] = point

        # First observation: establish starting position
        if previous_point is None:
            return None

        previous_side = self._side(previous_point)

        # Vehicle has not changed sides
        if previous_side == current_side:
            return None

        # Check actual trajectory intersection
        intersection = self._get_intersection(
            previous_point,
            point
        )

        if intersection is None:
            return None

        # Determine direction
        direction = (
            "IN"
            if previous_side < 0 and current_side > 0
            else "OUT"
        )

        if self.flip:
            direction = (
                "OUT" if direction == "IN" else "IN"
            )

        # Mark vehicle as counted
        self._counted.add(track_id)

        self.counts[cls_name][direction] += 1

        event = {
            "frame": frame_idx,
            "time_s": round(time_s, 2),
            "track_id": track_id,
            "vehicle_type": cls_name,
            "direction": direction,
            "crossing_point": (
                round(intersection[0], 2),
                round(intersection[1], 2)
            )
        }

        self.events.append(event)

        return event

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    @property
    def total(self) -> int:

        return len(self._counted)

    def totals_by_class(self) -> dict[str, int]:

        return {
            cls: data["IN"] + data["OUT"]
            for cls, data in self.counts.items()
        }

    def totals_by_direction(self) -> dict[str, int]:

        return {
            direction: sum(
                data[direction]
                for data in self.counts.values()
            )
            for direction in self.DIRECTIONS
        }
