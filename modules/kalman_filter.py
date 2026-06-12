"""
modules/kalman_filter.py
─────────────────────────────────────────────────────────
Module 6 — Kalman Filter State Estimation

FIX (v2): Upgraded from a 4-state constant-velocity (CV) model to a
6-state constant-acceleration (CA) model:

    State  : [x, y, vx, vy, ax, ay]
    Observe: [x, y]

The CA model gives:
  • Smoother velocity estimates during NPC speed changes
  • Direct acceleration estimates (ax, ay) exported to KFState — the
    trajectory predictor uses these instead of ego acceleration as a proxy
  • Lower Kalman lag when the NPC accelerates or brakes

dt is taken from sim_time deltas (irregular in this dataset),
so F and Q are rebuilt every step.

Transition matrix (6 × 6):
    ┌ 1  0  dt  0  ½dt²  0   ┐
    │ 0  1  0   dt  0   ½dt² │
    │ 0  0  1   0   dt   0   │
    │ 0  0  0   1   0    dt  │
    │ 0  0  0   0   1    0   │
    └ 0  0  0   0   0    1   ┘
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
    x:  float
    y:  float
    vx: float
    vy: float
    ax: float = 0.0   # estimated NPC acceleration (world frame)
    ay: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy, self.ax, self.ay])


class NpcKalmanFilter:
    """
    6-state constant-acceleration Kalman filter for NPC tracking.

    State:    [x, y, vx, vy, ax, ay]
    Measured: [x, y]

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
            self._state = np.zeros(6)
            self._P     = np.eye(6) * self.cfg["initial_uncertainty"]

    # ── Public API ───────────────────────────────────────────────────────────

    def initialise(self, x: float, y: float,
                   vx: float = 0.0, vy: float = 0.0,
                   ax: float = 0.0, ay: float = 0.0):
        """Seed the filter state with the first measurement."""
        init = np.array([x, y, vx, vy, ax, ay], dtype=np.float64)
        if self._use_filterpy:
            self._kf.x = init.reshape(6, 1)
            self._kf.P = np.eye(6) * self.cfg["initial_uncertainty"]
        else:
            self._state = init
            self._P     = np.eye(6) * self.cfg["initial_uncertainty"]
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
        KFState  current filtered estimate (includes ax, ay).
        """
        if dt <= 0:
            dt = self.cfg["default_dt"]

        if not self._initialised:
            if measured_xy is None:
                return KFState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.initialise(float(measured_xy[0]), float(measured_xy[1]))
            return KFState(float(measured_xy[0]), float(measured_xy[1]),
                           0.0, 0.0, 0.0, 0.0)

        if self._use_filterpy:
            return self._step_filterpy(measured_xy, dt)
        else:
            return self._step_numpy(measured_xy, dt)

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    # ── filterpy backend ─────────────────────────────────────────────────────

    def _build_filterpy(self) -> "_FPKalmanFilter":
        kf = _FPKalmanFilter(dim_x=6, dim_z=2)

        # F is updated with the correct dt each step
        kf.F = np.eye(6, dtype=np.float64)

        # Measure only [x, y]
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]], dtype=np.float64)

        # Measurement noise: [x, y] observation
        kf.R = np.eye(2) * self.cfg["observation_noise_var"]

        # Initial covariance
        kf.P = np.eye(6) * self.cfg["initial_uncertainty"]

        # Process noise built per-step (depends on dt); set identity as placeholder
        kf.Q = np.eye(6) * self.cfg["process_noise_var"]

        return kf

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """Constant-acceleration transition matrix for given dt."""
        dt2 = 0.5 * dt * dt
        return np.array([
            [1, 0, dt,  0, dt2,   0],
            [0, 1,  0, dt,   0, dt2],
            [0, 0,  1,  0,  dt,   0],
            [0, 0,  0,  1,   0,  dt],
            [0, 0,  0,  0,   1,   0],
            [0, 0,  0,  0,   0,   1],
        ], dtype=np.float64)

    @staticmethod
    def _build_Q(dt: float, sigma_pos: float, sigma_accel: float) -> np.ndarray:
        """
        Block-diagonal process noise matrix.
        Position/velocity block uses process_noise_var (small, NPC moves smoothly).
        Acceleration block uses process_noise_accel_var (larger, allows accel changes).
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # 2×2 blocks for (x,vx,ax) and (y,vy,ay) using Singer acceleration model
        q_pos   = sigma_pos
        q_accel = sigma_accel

        Q = np.zeros((6, 6), dtype=np.float64)
        # x, vx, ax block (indices 0, 2, 4)
        for i, j, row_i, row_j in [(0, 0, 0, 0), (0, 2, 0, 2), (2, 0, 2, 0),
                                    (2, 2, 2, 2), (0, 4, 0, 4), (4, 0, 4, 0),
                                    (4, 4, 4, 4), (2, 4, 2, 4), (4, 2, 4, 2)]:
            pass  # use simpler diagonal form below

        # Simpler diagonal Q: position noise and acceleration noise
        Q[0, 0] = q_pos * dt4 / 4
        Q[1, 1] = q_pos * dt4 / 4
        Q[2, 2] = q_pos * dt2
        Q[3, 3] = q_pos * dt2
        Q[4, 4] = q_accel
        Q[5, 5] = q_accel

        # Off-diagonal position–velocity coupling
        Q[0, 2] = Q[2, 0] = q_pos * dt3 / 2
        Q[1, 3] = Q[3, 1] = q_pos * dt3 / 2

        return Q

    def _step_filterpy(self, measured_xy: Optional[np.ndarray], dt: float) -> KFState:
        sigma_pos   = self.cfg["process_noise_var"]
        sigma_accel = self.cfg.get("process_noise_accel_var", 0.8)

        self._kf.F = self._build_F(dt)
        self._kf.Q = self._build_Q(dt, sigma_pos, sigma_accel)

        self._kf.predict()
        if measured_xy is not None:
            self._kf.update(np.array(measured_xy, dtype=np.float64).reshape(2, 1))

        x, y, vx, vy, ax, ay = self._kf.x.flatten()
        return KFState(float(x), float(y), float(vx), float(vy),
                       float(ax), float(ay))

    # ── NumPy fallback backend ───────────────────────────────────────────────

    def _step_numpy(self, measured_xy: Optional[np.ndarray], dt: float) -> KFState:
        sigma_pos   = self.cfg["process_noise_var"]
        sigma_accel = self.cfg.get("process_noise_accel_var", 0.8)

        F = self._build_F(dt)
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        Q = self._build_Q(dt, sigma_pos, sigma_accel)
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
            P_new = (np.eye(6) - K @ H) @ P_pred
        else:
            x_new, P_new = x_pred, P_pred

        self._state, self._P = x_new, P_new
        x, y, vx, vy, ax, ay = x_new
        return KFState(float(x), float(y), float(vx), float(vy),
                       float(ax), float(ay))
