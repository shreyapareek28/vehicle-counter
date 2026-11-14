"""
MODULE 1 - DETECTION            (Owner: Member 1)
-------------------------------------------------
Wraps the YOLOv8 model. Given a frame it returns the vehicles found in it,
already filtered to vehicle classes and with tracker IDs attached.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from .utils import VEHICLE_NAMES


@dataclass
class Detection:
    track_id: int
    cls_name: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, float(y2)


class VehicleDetector:
    """YOLOv8 detector with built-in multi-object tracking (ByteTrack / BoT-SORT)."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 640, device: str | None = None,
                 tracker: str = "bytetrack.yaml"):
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.tracker = device, tracker
        # keep only vehicle classes (works for COCO and custom-trained weights)
        self.class_ids = [i for i, n in self.model.names.items()
                          if n.lower() in VEHICLE_NAMES]
        if not self.class_ids:
            raise ValueError(
                "No vehicle classes found in this model. Names present: "
                f"{list(self.model.names.values())}")

    @property
    def class_names(self) -> list[str]:
        return [self.model.names[i] for i in self.class_ids]

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame, persist=True, tracker=self.tracker, conf=self.conf,
            iou=self.iou, imgsz=self.imgsz, classes=self.class_ids,
            device=self.device, verbose=False)[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:   # nothing tracked in this frame
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        return [Detection(int(i), self.model.names[int(c)], float(p), tuple(int(v) for v in b))
                for b, i, c, p in zip(xyxy, ids, cls, confs)]
