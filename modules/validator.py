"""
modules/validator.py
─────────────────────────────────────────────────────────
Module 8 — Validation against npcs.csv ground truth

Computes ADE / FDE / RMSE for each prediction model, per
frame and aggregated over the whole sequence.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from .predictor import TrajectoryPrediction


@dataclass
class FrameMetrics:
    frame_idx:  int
    model_name: str
    ade:        float   # average displacement error over the horizon
    fde:        float   # final displacement error (last predicted step)
    rmse:       float   # root-mean-square error over the horizon
    n_valid:    int      # number of horizon steps that had ground truth


class Validator:
    """
    Compares predicted trajectories against npcs.csv ground truth.

    Parameters
    ----------
    cfg     : dict          Full pipeline config dict.
    npcs_df : pd.DataFrame  Ground-truth NPC dataframe, indexed by frame_idx.
    """

    def __init__(self, cfg: dict, npcs_df: pd.DataFrame):
        self.cfg     = cfg["validation"]
        self.npcs_df = npcs_df
        self._records: List[FrameMetrics] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(self, frame_idx: int, prediction: TrajectoryPrediction) -> Optional[FrameMetrics]:
        """
        Evaluate a single prediction against ground truth for frames
        [frame_idx+1 .. frame_idx+horizon].

        Returns None if no ground-truth frames are available in that range
        (e.g. near the end of the sequence).
        """
        if frame_idx < self.cfg["skip_warmup_frames"]:
            return None

        errors = []
        for i, (px, py) in enumerate(prediction.points):
            gt_idx = frame_idx + i + 1
            if gt_idx not in self.npcs_df.index:
                break
            gx = float(self.npcs_df.at[gt_idx, "x"])
            gy = float(self.npcs_df.at[gt_idx, "y"])
            errors.append(np.hypot(px - gx, py - gy))

        if not errors:
            return None

        errors = np.array(errors)
        metrics = FrameMetrics(
            frame_idx=frame_idx,
            model_name=prediction.model_name,
            ade=float(errors.mean()),
            fde=float(errors[-1]),
            rmse=float(np.sqrt((errors ** 2).mean())),
            n_valid=len(errors),
        )
        self._records.append(metrics)
        return metrics

    def summary(self) -> pd.DataFrame:
        """
        Aggregate metrics per model across all evaluated frames.

        Returns
        -------
        pd.DataFrame with columns: model, ADE_mean, FDE_mean, RMSE_mean,
        ADE_std, FDE_std, RMSE_std, n_frames
        """
        if not self._records:
            return pd.DataFrame(columns=[
                "model", "ADE_mean", "FDE_mean", "RMSE_mean",
                "ADE_std", "FDE_std", "RMSE_std", "n_frames"
            ])

        df = self.to_dataframe()
        rows = []
        for model_name, g in df.groupby("model_name"):
            rows.append({
                "model":     model_name,
                "ADE_mean":  g["ade"].mean(),
                "FDE_mean":  g["fde"].mean(),
                "RMSE_mean": g["rmse"].mean(),
                "ADE_std":   g["ade"].std(),
                "FDE_std":   g["fde"].std(),
                "RMSE_std":  g["rmse"].std(),
                "n_frames":  len(g),
            })
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """All per-frame metrics as a flat DataFrame."""
        return pd.DataFrame([{
            "frame_idx":  r.frame_idx,
            "model_name": r.model_name,
            "ade":        r.ade,
            "fde":        r.fde,
            "rmse":       r.rmse,
            "n_valid":    r.n_valid,
        } for r in self._records])

    def save(self, path: str):
        self.to_dataframe().to_csv(path, index=False)
        print(f"[Validator] Per-frame metrics saved → {path}")

    def save_summary(self, path: str):
        self.summary().to_csv(path, index=False)
        print(f"[Validator] Summary metrics saved → {path}")
        print(self.summary().to_string(index=False))
