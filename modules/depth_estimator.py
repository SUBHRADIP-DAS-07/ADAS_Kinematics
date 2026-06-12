"""
modules/depth_estimator.py
─────────────────────────────────────────────────────────
Module 3 — Depth / 3D Position Estimation

Two methods are supported:
  A) bbox_geometry  — uses CARLA camera intrinsics + known vehicle height.
                      Fast, deterministic, works well for a single-actor
                      scenario where the vehicle type (Audi TT) is known.
  B) depth_anything — monocular depth network (DepthAnything v2).
                      Produces a full depth map; scale is calibrated
                      against dist_to_ego from the first N frames.

FIX (v2): bbox_geometry now uses the gt_dist_m signal (passed from
npcs.csv dist_to_ego) to calibrate a multiplicative scale factor over
the first N frames.  This corrects a systematic ~4 m under-estimation
caused by the YOLO bbox height being larger than the assumed physical
vehicle height (camera elevation + perspective effects).  The calibrated
scale is printed once and applied for the rest of the run.  If gt_dist_m
is never supplied (real-world deployment with no GT), the estimator falls
back gracefully to the raw formula.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .detector import Detection


@dataclass
class DepthEstimate:
    depth_m:   float          # forward distance to detected object (metres)
    point_cam: np.ndarray     # 3-D point in camera frame [X, Y, Z]  (metres)
    method:    str            # "bbox_geometry" or "depth_anything"


class DepthEstimator:
    """
    Estimates the 3-D position of a detected vehicle in the camera frame.

    CARLA camera frame convention (same as OpenCV / standard pinhole):
        Z  — forward (depth)
        X  — right
        Y  — down

    Parameters
    ----------
    cfg : dict   Full pipeline config dict.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.cam_cfg  = cfg["carla"]
        self.dep_cfg  = cfg["depth"]
        self.method   = self.dep_cfg["method"]

        # ── Intrinsics ──────────────────────────────────────────────────────
        W   = self.cam_cfg["image_width"]
        H   = self.cam_cfg["image_height"]
        fov = self.cam_cfg["fov_degrees"]
        self.W = W;  self.H = H
        self.f = W / (2.0 * np.tan(np.radians(fov / 2.0)))   # focal length px
        self.cx_img = W / 2.0
        self.cy_img = H / 2.0

        self.K = np.array([[self.f, 0,      self.cx_img],
                           [0,      self.f, self.cy_img],
                           [0,      0,      1.0        ]])

        print(f"[DepthEst] Intrinsics — f={self.f:.1f}px, "
              f"cx={self.cx_img}, cy={self.cy_img}")

        # ── bbox_geometry online calibration ─────────────────────────────────
        # Accumulates (raw_depth, gt_dist) pairs for the first N frames and
        # computes a multiplicative scale: scale = median(gt_dist / raw_depth).
        # This corrects errors from: camera elevation perspective, bbox padding,
        # and the approximate known_vehicle_height_m assumption.
        self._n_calib:          int   = self.dep_cfg.get("scale_calibration_frames", 20)
        self._bbox_calib_buf:   List[Tuple[float, float]] = []
        self._bbox_scale:       float = 1.0     # applied once calibrated
        self._bbox_calibrated:  bool  = False

        # Configurable depth clamp (metres)
        self._min_depth: float = self.dep_cfg.get("min_depth_m", 5.0)
        self._max_depth: float = self.dep_cfg.get("max_depth_m", 60.0)

        # ── Depth Anything (optional) ────────────────────────────────────────
        self._da_model    = None
        self._da_scale    = None   # calibrated scale factor (relative → metric)
        self._da_calib_buf: List[Tuple[float, float]] = []
        if self.method == "depth_anything":
            self._load_depth_anything()

    # ── Public API ───────────────────────────────────────────────────────────

    def estimate(
        self,
        det: Detection,
        frame: Optional[np.ndarray] = None,
        gt_dist_m: Optional[float] = None,
    ) -> DepthEstimate:
        """
        Estimate depth for a single detection.

        Parameters
        ----------
        det        : Detection bounding box in pixel space.
        frame      : BGR image (required for depth_anything method).
        gt_dist_m  : Ground-truth distance in metres — used to calibrate
                     the depth scale factor for the first N frames.
                     Safe to omit in real-world (non-GT) deployments.
        """
        if self.method == "depth_anything" and self._da_model is not None:
            return self._estimate_depth_anything(det, frame, gt_dist_m)
        else:
            return self._estimate_bbox_geometry(det, gt_dist_m)

    def pixel_to_camera(self, px: float, py: float, depth_m: float) -> np.ndarray:
        """Back-project a pixel (px, py) at known depth to camera-frame 3D."""
        X = (px - self.cx_img) * depth_m / self.f
        Y = (py - self.cy_img) * depth_m / self.f
        Z = depth_m
        return np.array([X, Y, Z], dtype=np.float64)

    # ── Method A — Bounding-box geometry ─────────────────────────────────────

    def _estimate_bbox_geometry(
        self,
        det: Detection,
        gt_dist_m: Optional[float] = None,
    ) -> DepthEstimate:
        """
        Estimate depth from apparent bounding-box height and known object height.

            depth_raw ≈ (known_height_m × f_px) / bbox_height_px

        Audi TT roof height ≈ 1.34 m (matches npcs type_id vehicle.audi.tt).

        Online calibration
        ------------------
        If gt_dist_m is provided for the first N frames, a multiplicative
        scale factor is computed:
            scale = median( gt_dist_i / depth_raw_i )
        and applied as:
            depth_calibrated = depth_raw * scale
        This corrects the systematic under-estimation from camera elevation
        and bounding-box padding effects.
        """
        h_known = self.dep_cfg["known_vehicle_height_m"]   # 1.34 m
        h_px    = max(det.height, 1e-3)                    # avoid div-by-zero
        depth_raw = (h_known * self.f) / h_px

        # ── Online scale calibration (uses GT dist from npcs.csv) ────────────
        if gt_dist_m is not None and not self._bbox_calibrated:
            if depth_raw > 1.0:   # sanity — skip degenerate boxes
                self._bbox_calib_buf.append((depth_raw, float(gt_dist_m)))

            if len(self._bbox_calib_buf) >= self._n_calib:
                scales = [gt / raw for raw, gt in self._bbox_calib_buf]
                self._bbox_scale = float(np.median(scales))
                self._bbox_calibrated = True
                effective_h = h_known * self._bbox_scale
                print(f"[DepthEst] bbox_geometry scale calibrated → "
                      f"{self._bbox_scale:.4f}  "
                      f"(effective vehicle height: {effective_h:.3f} m  "
                      f"from {self._n_calib} frames)")

        # Apply calibrated scale and clamp to plausible range
        depth = float(np.clip(depth_raw * self._bbox_scale,
                               self._min_depth, self._max_depth))
        point = self.pixel_to_camera(det.cx, det.cy, depth)
        return DepthEstimate(depth_m=depth, point_cam=point,
                             method="bbox_geometry")

    # ── Method B — Depth Anything v2 ─────────────────────────────────────────

    def _estimate_depth_anything(
        self,
        det: Detection,
        frame: Optional[np.ndarray],
        gt_dist_m: Optional[float],
    ) -> DepthEstimate:
        """
        Run Depth Anything v2 and sample the depth map inside the bbox.
        Calibrate scale factor against gt_dist_m for the first N frames.
        """
        if frame is None:
            return self._estimate_bbox_geometry(det, gt_dist_m)

        depth_map = self._da_model.infer_image(frame)   # relative depth [0,1]

        # Sample median of a 20×20 patch at bbox centre
        bx, by = int(det.cx), int(det.cy)
        patch  = depth_map[max(0,by-10):by+10, max(0,bx-10):bx+10]
        rel_d  = float(np.median(patch)) if patch.size > 0 else float(np.median(depth_map))

        # Calibrate scale
        if gt_dist_m is not None and len(self._da_calib_buf) < self._n_calib:
            if rel_d > 1e-6:
                self._da_calib_buf.append((rel_d, float(gt_dist_m)))
                if len(self._da_calib_buf) == self._n_calib:
                    self._da_scale = float(
                        np.median([gt / rd for rd, gt in self._da_calib_buf])
                    )
                    print(f"[DepthEst] DA scale calibrated → {self._da_scale:.2f}")

        scale = self._da_scale if self._da_scale else 10.0   # rough fallback
        depth = float(np.clip(rel_d * scale, self._min_depth, self._max_depth))
        point = self.pixel_to_camera(det.cx, det.cy, depth)
        return DepthEstimate(depth_m=depth, point_cam=point,
                             method="depth_anything")

    def _load_depth_anything(self):
        try:
            import sys
            sys.path.append("Depth-Anything-V2")
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2

            enc = self.dep_cfg["depth_anything_encoder"]
            cfg_map = {
                "vits": {"encoder":"vits","features":64,  "out_channels":[48,96,192,384]},
                "vitb": {"encoder":"vitb","features":128, "out_channels":[96,192,384,768]},
                "vitl": {"encoder":"vitl","features":256, "out_channels":[256,512,1024,1024]},
            }
            model = DepthAnythingV2(**cfg_map[enc])
            ckpt  = self.dep_cfg["depth_anything_checkpoint"]
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._da_model = model.to(device).eval()
            print(f"[DepthEst] Depth Anything v2 ({enc}) loaded on {device}")
        except Exception as exc:
            print(f"[DepthEst] Depth Anything not available — {exc}. "
                  "Falling back to bbox_geometry.")
            self.method = "bbox_geometry"
