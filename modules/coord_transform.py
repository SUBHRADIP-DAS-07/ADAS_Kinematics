"""
modules/coord_transform.py
─────────────────────────────────────────────────────────
Module 5 — Coordinate Transform (CARLA-specific)

Converts a 3-D point in the camera frame to a 2-D position
in the CARLA world frame, using the ego vehicle's pose.

CARLA coordinate system (UE4 / left-handed):
  X — forward (east in default map)
  Y — right   (south in default map)
  Z — up

Camera convention (OpenCV pinhole, mounted facing forward):
  Z_cam — forward (depth)
  X_cam — right
  Y_cam — down

Ego yaw in ego.csv is in DEGREES, increasing as the vehicle
turns left (counter-clockwise viewed from above), which is
the CARLA convention. We validate this against the dataset:
  • yaw range: -0.28 → 19.6 °
  • ego travels primarily in +X world direction with a slight
    +Y component as it turns left (yaw increases), consistent
    with CARLA's left-hand sign convention.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class WorldPoint:
    """A 2-D position in the CARLA world frame (metres)."""
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


class CoordTransformer:
    """
    Transforms points from camera space → CARLA world space.

    Steps
    -----
    1. Camera frame (OpenCV)  →  Ego/vehicle frame
       X_ego =  Z_cam   (forward)
       Y_ego =  X_cam   (right)
       Z_ego = -Y_cam   (up, flip down-axis)

    2. Ego frame  →  CARLA world frame
       Rotate by ego yaw (degrees → radians, left-hand convention):
       X_world = ego.x + X_ego * cos(yaw) - Y_ego * sin(yaw)
       Y_world = ego.y + X_ego * sin(yaw) + Y_ego * cos(yaw)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        print("[CoordTransform] CARLA left-hand coord system initialised")

    # ── Public API ───────────────────────────────────────────────────────────

    def camera_to_world(self, point_cam: np.ndarray, ego) -> WorldPoint:
        """
        Parameters
        ----------
        point_cam : np.ndarray [X_cam, Y_cam, Z_cam] — camera frame (metres)
        ego       : EgoState (from data_loader)

        Returns
        -------
        WorldPoint  2-D world position (metres, CARLA frame)
        """
        X_cam, Y_cam, Z_cam = point_cam

        # Step 1 — camera frame → ego/vehicle frame
        X_ego =  Z_cam    # forward
        Y_ego =  X_cam    # right
        # Z_ego = -Y_cam  # up (unused for 2-D prediction)

        # Step 2 — ego frame → world frame (CARLA yaw in degrees)
        yaw_rad = np.radians(ego.yaw)
        X_world = ego.x + X_ego * np.cos(yaw_rad) - Y_ego * np.sin(yaw_rad)
        Y_world = ego.y + X_ego * np.sin(yaw_rad) + Y_ego * np.cos(yaw_rad)

        return WorldPoint(x=float(X_world), y=float(Y_world))

    def relative_to_ego(self, world_pt: WorldPoint, ego) -> np.ndarray:
        """
        Express a world point in the ego vehicle frame (metres).
        Useful for model training and visualisation.

        Returns [dx_forward, dy_lateral]
        """
        dx = world_pt.x - ego.x
        dy = world_pt.y - ego.y
        yaw_rad = np.radians(ego.yaw)
        forward =  dx * np.cos(yaw_rad) + dy * np.sin(yaw_rad)
        lateral = -dx * np.sin(yaw_rad) + dy * np.cos(yaw_rad)
        return np.array([forward, lateral])

    def world_to_image(self, world_pt: WorldPoint, ego, K: np.ndarray) -> np.ndarray:
        """
        Project a world point back into image pixel coordinates (for overlay).

        Parameters
        ----------
        world_pt : WorldPoint
        ego      : EgoState
        K        : 3×3 camera intrinsic matrix

        Returns
        -------
        np.ndarray  [u, v] pixel coordinates (float)
        """
        yaw_rad = np.radians(ego.yaw)

        # World → ego frame
        dx = world_pt.x - ego.x
        dy = world_pt.y - ego.y
        X_ego =  dx * np.cos(yaw_rad) + dy * np.sin(yaw_rad)
        Y_ego = -dx * np.sin(yaw_rad) + dy * np.cos(yaw_rad)
        Z_ego = 0.0   # assume same ground plane

        # Ego frame → camera frame (inverse of cam→ego)
        Z_cam = X_ego    # forward
        X_cam = Y_ego    # right
        Y_cam = -Z_ego   # down

        if Z_cam <= 0.1:        # behind camera, skip
            return np.array([-1.0, -1.0])

        f  = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        u  = f * (X_cam / Z_cam) + cx
        v  = f * (Y_cam / Z_cam) + cy
        return np.array([u, v])

    # ── Sanity check (call once after pipeline is loaded) ────────────────────

    def validate_transform(self, ego, npc_gt) -> float:
        """
        Use the known NPC world position to back-check the transform chain.
        Returns lateral error in metres between estimated and GT position
        if you pass a synthetic camera point at the GT depth.

        Primarily used during development to verify sign conventions.
        """
        import math
        gt_dist   = npc_gt.dist_to_ego
        # Synthesise a camera point at the ground-truth range, centred
        dummy_cam = np.array([0.0, 0.0, gt_dist])
        est       = self.camera_to_world(dummy_cam, ego)
        err       = math.sqrt((est.x - npc_gt.x)**2 + (est.y - npc_gt.y)**2)
        return err
