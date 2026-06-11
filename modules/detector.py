"""
modules/detector.py
─────────────────────────────────────────────────────────
Module 2 — Vehicle Detection (YOLOv8)

Wraps YOLOv8 inference, filters for vehicle classes, and
returns Detection objects ready for the tracker and depth
estimator.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Detection:
    """
    A single vehicle detection in image space.
    bbox: [x1, y1, x2, y2] in pixels
    """
    bbox:      np.ndarray          # [x1, y1, x2, y2]
    conf:      float
    class_id:  int
    class_name: str = ""

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def tlwh(self) -> np.ndarray:
        """Top-left-width-height format (used by DeepSORT)."""
        return np.array([self.bbox[0], self.bbox[1], self.width, self.height])

    @property
    def area(self) -> float:
        return self.width * self.height


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Parameters
    ----------
    cfg : dict
        Pipeline config (cfg["detection"] section used).

    Notes
    -----
    CARLA-specific: the NPC (Audi TT) is always ahead of ego in this
    dataset and appears in the upper-center of the 1920×1080 frame.
    `pick_best()` returns the most-confident detection in the expected
    frontal zone, filtering out distant or irrelevant detections.
    """

    COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, cfg: dict):
        self.cfg = cfg["detection"]
        self.img_w = cfg["carla"]["image_width"]
        self.img_h = cfg["carla"]["image_height"]
        self._model = None
        self._load_model()

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 on one BGR frame; return list of vehicle detections."""
        if self._model is None:
            return []

        results = self._model(
            frame,
            classes=self.cfg["vehicle_classes"],
            conf=self.cfg["confidence_threshold"],
            iou=self.cfg["iou_threshold"],
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls)
                h_px   = y2 - y1

                # Filter tiny detections (far-away or false positives)
                if h_px < self.cfg["min_bbox_height_px"]:
                    continue
                # Filter detections below road level in image
                if y1 > self.img_h * 0.85:
                    continue

                detections.append(Detection(
                    bbox=np.array([x1, y1, x2, y2]),
                    conf=float(box.conf),
                    class_id=cls_id,
                    class_name=self.COCO_VEHICLE_CLASSES.get(cls_id, "vehicle"),
                ))

        return detections

    def pick_best(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Return the most likely NPC target from a list of detections.

        Strategy:
          1. Prefer detections in the front-center column of the image
             (ego's NPC is directly ahead throughout this dataset).
          2. Among those, pick highest confidence.
        """
        if not detections:
            return None

        cx_center   = self.img_w / 2
        center_band = self.img_w * 0.35          # ±35% of image width
        front_dets  = [d for d in detections
                       if abs(d.cx - cx_center) < center_band]

        pool = front_dets if front_dets else detections
        return max(pool, key=lambda d: d.conf)

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            weights = self.cfg["model_weights"]
            self._model = YOLO(weights)
            print(f"[Detector] YOLOv8 loaded — weights: {weights}")
        except ImportError:
            print("[Detector] WARNING: ultralytics not installed. "
                  "Run: pip install ultralytics")
        except Exception as exc:
            print(f"[Detector] WARNING: could not load model — {exc}")
