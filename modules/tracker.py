"""
modules/tracker.py
─────────────────────────────────────────────────────────
Module 4 — Multi-Object Tracking (DeepSORT)

Maintains consistent track IDs across frames so that the
Kalman filter always updates the same target vehicle, even
through brief occlusions or missed detections.

For the single-NPC scenario in this dataset, a simpler
centroid-based fallback is also provided (no external deps).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .detector import Detection


@dataclass
class Track:
    """Active track with its current bounding box and detection."""
    track_id:  int
    bbox:      np.ndarray     # [x1, y1, x2, y2]
    det:       Optional[Detection]  # best matching detection this frame
    age:       int = 0        # frames since last confirmed match

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2)


class VehicleTracker:
    """
    Multi-object tracker wrapper.

    Tries DeepSORT first; if the library is not available falls back to a
    lightweight IoU-based centroid tracker that works fine for this
    single-actor scenario.

    Parameters
    ----------
    cfg : dict   Full pipeline config dict.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["tracking"]
        self._backend: str = "none"
        self._ds_tracker = None
        self._simple_tracks: List[_SimpleTrack] = []
        self._next_id = 1
        self._init_deepsort()

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Feed detections for the current frame; return confirmed tracks.

        Parameters
        ----------
        detections : list of Detection objects from the detector.
        frame      : BGR image (used by DeepSORT appearance encoder).

        Returns
        -------
        List of Track objects with bboxes in pixel space.
        """
        if self._backend == "deepsort":
            return self._update_deepsort(detections, frame)
        else:
            return self._update_simple(detections)

    def pick_target(self, tracks: List[Track], frame_w: int) -> Optional[Track]:
        """
        From all active tracks, return the one most likely to be the
        NPC vehicle directly ahead of the ego.

        Strategy: closest-to-centre-x track, among confirmed tracks
        (age == 0 means matched this frame).
        """
        fresh = [t for t in tracks if t.age == 0]
        if not fresh:
            fresh = tracks
        if not fresh:
            return None
        cx_mid = frame_w / 2
        return min(fresh, key=lambda t: abs(t.cx - cx_mid))

    # ── DeepSORT backend ─────────────────────────────────────────────────────

    def _init_deepsort(self):
        try:
            import sys
            sys.path.append("deep_sort")
            from deep_sort import nn_matching
            from deep_sort.tracker import Tracker as DSTracker

            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine",
                self.cfg["max_cosine_distance"],
                None,
            )
            self._ds_tracker = DSTracker(
                metric,
                max_age=self.cfg["max_age"],
                n_init=self.cfg["n_init"],
            )
            self._backend = "deepsort"
            print("[Tracker] DeepSORT initialised")
        except Exception as exc:
            print(f"[Tracker] DeepSORT not available ({exc}). "
                  "Using simple IoU tracker.")
            self._backend = "simple"

    def _update_deepsort(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        from deep_sort.detection import Detection as DSDetection

        ds_dets = []
        for d in detections:
            # Stub feature vector (replace with a real Re-ID encoder if needed)
            feature = np.zeros(self.cfg["feature_dim"], dtype=np.float32)
            ds_dets.append(DSDetection(d.tlwh, d.conf, feature))

        self._ds_tracker.predict()
        self._ds_tracker.update(ds_dets)

        tracks = []
        for t in self._ds_tracker.tracks:
            if not t.is_confirmed():
                continue
            x1, y1, w, h = t.to_tlwh()
            bbox = np.array([x1, y1, x1 + w, y1 + h])
            # Match back to a Detection (closest centre)
            best_det = _match_det_to_bbox(detections, bbox)
            tracks.append(Track(
                track_id=t.track_id,
                bbox=bbox,
                det=best_det,
                age=t.time_since_update,
            ))
        return tracks

    # ── Simple IoU centroid fallback ─────────────────────────────────────────

    def _update_simple(self, detections: List[Detection]) -> List[Track]:
        """
        Minimal IoU-based tracker — sufficient for 1 actor.
        Matches detections to existing tracks by IoU; unmatched tracks age out.
        """
        IOU_THRESH = 0.3
        MAX_AGE    = self.cfg["max_age"]

        # Age all tracks
        for t in self._simple_tracks:
            t.age += 1

        unmatched_dets = list(detections)
        for t in self._simple_tracks:
            if not unmatched_dets:
                break
            best_iou, best_det = 0.0, None
            for d in unmatched_dets:
                iou = _iou(t.bbox, d.bbox)
                if iou > best_iou:
                    best_iou, best_det = iou, d
            if best_iou >= IOU_THRESH and best_det is not None:
                t.bbox = best_det.bbox.copy()
                t.det  = best_det
                t.age  = 0
                unmatched_dets.remove(best_det)

        # Spawn new tracks for unmatched detections
        for d in unmatched_dets:
            self._simple_tracks.append(_SimpleTrack(
                track_id=self._next_id,
                bbox=d.bbox.copy(),
                det=d,
                age=0,
            ))
            self._next_id += 1

        # Remove stale tracks
        self._simple_tracks = [t for t in self._simple_tracks if t.age < MAX_AGE]

        return [Track(track_id=t.track_id, bbox=t.bbox, det=t.det, age=t.age)
                for t in self._simple_tracks]


# ── Internal helpers ──────────────────────────────────────────────────────────

@dataclass
class _SimpleTrack:
    track_id: int
    bbox:     np.ndarray
    det:      Optional[Detection]
    age:      int


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _match_det_to_bbox(detections: List[Detection], bbox: np.ndarray) -> Optional[Detection]:
    if not detections:
        return None
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return min(detections, key=lambda d: (d.cx - cx)**2 + (d.cy - cy)**2)
