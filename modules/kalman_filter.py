"""
modules/kalman_filter.py
─────────────────────────────────────────────────────────
Module 6 — Kalman Filter State Estimation

Fuses noisy per-frame world-position measurements (from
detection → depth → coordinate transform) into a smooth
state estimate [x, y, vx, vy] for the NPC vehicle.

Constant-velocity motion model:
    x_k  = x_{k-1} + vx_{k-1} * dt
    y_k  = y_{k-1} + vy_{k-1} * dt
    vx_k = vx_{k-1}
    vy_k = vy_{k-1}

dt is taken from sim_time deltas (irregular in this dataset),
so F and Q are rebuilt every step.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from filterpy.kalman import KalmanFilter as _FPKalmanFilter
    _HAS_FILTERPY = True
except ImportError:
    _HAS_FILTERPY = False


@dataclass
class KFState:
    x: float
    y: float
    vx: float
    vy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])


class NpcKalmanFilter:
    """
    Wraps filterpy's KalmanFilter (or a minimal NumPy fallback) to track
    the NPC's [x, y, vx, vy] state in CARLA world coordinates.

    Parameters
    ----------
    cfg : dict   Full pipeline config dict (uses cfg["kalman"]).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["kalman"]
        self._initialised = False
        self._use_filterpy = _HAS_FILTERPY

        if self._use_filterpy:
            self._kf = self._build_filterpy()
        else:
            print("[KalmanFilter] filterpy not installed — using NumPy fallback. "
                  "Run: pip install filterpy")
            self._state = np.zeros(4)        # [x, y, vx, vy]
            self._P     = np.eye(4) * self.cfg["initial_uncertainty"]

    # ── Public API ───────────────────────────────────────────────────────────

    def initialise(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Seed the filter state with the first measurement."""
        if self._use_filterpy:
            self._kf.x = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
            self._kf.P = np.eye(4) * self.cfg["initial_uncertainty"]
        else:
            self._state = np.array([x, y, vx, vy], dtype=np.float64)
            self._P     = np.eye(4) * self.cfg["initial_uncertainty"]
        self._initialised = True

    def step(self, measured_xy: Optional[np.ndarray], dt: float) -> KFState:
        """
        Run one predict (+ optional update) cycle.

        Parameters
        ----------
        measured_xy : np.ndarray [x, y] world position measurement, or
                      None if no detection this frame (predict only).
        dt          : time delta in seconds since the previous frame.

        Returns
        -------
        KFState  current filtered estimate.
        """
        if dt <= 0:
            dt = self.cfg["default_dt"]

        if not self._initialised:
            if measured_xy is None:
                # Can't initialise without a measurement — return zeros
                return KFState(0.0, 0.0, 0.0, 0.0)
            self.initialise(float(measured_xy[0]), float(measured_xy[1]))
            return KFState(float(measured_xy[0]), float(measured_xy[1]), 0.0, 0.0)

        if self._use_filterpy:
            return self._step_filterpy(measured_xy, dt)
        else:
            return self._step_numpy(measured_xy, dt)

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    # ── filterpy backend ─────────────────────────────────────────────────────

    def _build_filterpy(self) -> "_FPKalmanFilter":
        kf = _FPKalmanFilter(dim_x=4, dim_z=2)

        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float64)   # dt baked in via .F[0,2]/.F[1,3] each step

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=np.float64)

        kf.R *= self.cfg["observation_noise_var"]
        kf.P *= self.cfg["initial_uncertainty"]
        kf.Q *= self.cfg["process_noise_var"]
        return kf

    def _step_filterpy(self, measured_xy: Optional[np.ndarray], dt: float) -> KFState:
        # update dt-dependent transition entries
        self._kf.F[0, 2] = dt
        self._kf.F[1, 3] = dt

        self._kf.predict()
        if measured_xy is not None:
            self._kf.update(np.array(measured_xy, dtype=np.float64).reshape(2, 1))

        x, y, vx, vy = self._kf.x.flatten()
        return KFState(float(x), float(y), float(vx), float(vy))

    # ── NumPy fallback backend ───────────────────────────────────────────────

    def _step_numpy(self, measured_xy: Optional[np.ndarray], dt: float) -> KFState:
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float64)
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=np.float64)
        Q = np.eye(4) * self.cfg["process_noise_var"]
        R = np.eye(2) * self.cfg["observation_noise_var"]

        # Predict
        x_pred = F @ self._state
        P_pred = F @ self._P @ F.T + Q

        if measured_xy is not None:
            z = np.array(measured_xy, dtype=np.float64)
            y_resid = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_new = x_pred + K @ y_resid
            P_new = (np.eye(4) - K @ H) @ P_pred
        else:
            x_new, P_new = x_pred, P_pred

        self._state, self._P = x_new, P_new
        x, y, vx, vy = x_new
        return KFState(float(x), float(y), float(vx), float(vy))
