"""
modules/predictor.py
─────────────────────────────────────────────────────────
Module 7 — Trajectory Prediction

Given the current Kalman-filtered NPC state [x, y, vx, vy]
(and optionally ego acceleration), predict the NPC's world
position for the next `horizon_frames` steps.

Three model families are provided:
  • Constant Velocity (CV)      — baseline
  • Constant Acceleration (CA)  — uses ego ax, ay as a proxy
  • LSTM (optional)             — learned from npcs.csv trajectories
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .kalman_filter import KFState


@dataclass
class TrajectoryPrediction:
    model_name: str
    points:     np.ndarray   # shape (horizon, 2) — world [x, y] per future step
    dts:        np.ndarray   # shape (horizon,)   — cumulative dt for each point


class TrajectoryPredictor:
    """
    Produces future-position predictions from a filtered NPC state.

    Parameters
    ----------
    cfg : dict   Full pipeline config dict.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg["prediction"]
        self.horizon = self.cfg["horizon_frames"]
        self.models  = self.cfg["models"]

        self._lstm = None
        if "lstm" in self.models:
            self._load_lstm(cfg)

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_all(
        self,
        state: KFState,
        dt: float,
        ego_ax: float = 0.0,
        ego_ay: float = 0.0,
        history: Optional[np.ndarray] = None,
    ) -> List[TrajectoryPrediction]:
        """
        Run every configured prediction model and return all results.

        Parameters
        ----------
        state    : current filtered NPC state.
        dt       : nominal time step (seconds) between future frames.
        ego_ax, ego_ay : ego acceleration (world frame), used by CA model
                         as a proxy for NPC acceleration.
        history  : optional (T, 4) array of past [x,y,vx,vy] states for LSTM.
        """
        out = []
        for name in self.models:
            if name == "constant_velocity":
                out.append(self.predict_cv(state, dt))
            elif name == "constant_acceleration":
                out.append(self.predict_ca(state, dt, ego_ax, ego_ay))
            elif name == "lstm" and self._lstm is not None and history is not None:
                out.append(self.predict_lstm(history, dt))
        return out

    def predict_cv(self, state: KFState, dt: float) -> TrajectoryPrediction:
        """Constant-velocity straight-line projection."""
        dts = np.arange(1, self.horizon + 1) * dt
        pts = np.stack([
            state.x + state.vx * dts,
            state.y + state.vy * dts,
        ], axis=1)
        return TrajectoryPrediction("constant_velocity", pts, dts)

    def predict_ca(self, state: KFState, dt: float,
                    ax: float = 0.0, ay: float = 0.0) -> TrajectoryPrediction:
        """
        Constant-acceleration projection.

        Uses the ego vehicle's acceleration as a (rough) proxy for the
        NPC's acceleration — appropriate for car-following scenarios
        where the NPC tends to mirror the lead vehicle's behaviour.
        """
        dts = np.arange(1, self.horizon + 1) * dt
        pts = np.stack([
            state.x + state.vx * dts + 0.5 * ax * dts ** 2,
            state.y + state.vy * dts + 0.5 * ay * dts ** 2,
        ], axis=1)
        return TrajectoryPrediction("constant_acceleration", pts, dts)

    def predict_lstm(self, history: np.ndarray, dt: float) -> TrajectoryPrediction:
        """
        LSTM-based prediction.

        Parameters
        ----------
        history : np.ndarray shape (obs_window, 4) — [x, y, vx, vy] history,
                  most recent last.
        """
        import torch
        with torch.no_grad():
            x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)
            pred = self._lstm(x).squeeze(0).numpy()  # (horizon, 2) — relative offsets

        # LSTM is trained to predict offsets relative to the last observed position
        last_xy = history[-1, :2]
        pts = last_xy[None, :] + pred
        dts = np.arange(1, self.horizon + 1) * dt
        return TrajectoryPrediction("lstm", pts, dts)

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_lstm(self, cfg: dict):
        try:
            import torch
            from .lstm_model import TrajLSTM   # see modules/lstm_model.py

            lstm_cfg = cfg["lstm"]
            model = TrajLSTM(
                input_dim=4,
                hidden_size=lstm_cfg["hidden_size"],
                num_layers=lstm_cfg["num_layers"],
                horizon=self.horizon,
            )
            ckpt = lstm_cfg["checkpoint"]
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            self._lstm = model
            print(f"[Predictor] LSTM loaded from {ckpt}")
        except Exception as exc:
            print(f"[Predictor] LSTM not available — {exc}. "
                  "Skipping LSTM predictions (train via scripts/train_lstm.py).")
            self.models = [m for m in self.models if m != "lstm"]
