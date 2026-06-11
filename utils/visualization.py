"""
utils/visualization.py
─────────────────────────────────────────────────────────
Drawing helpers for the annotated output video and
matplotlib plots of trajectories / error curves.
"""

from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from modules.detector import Detection
from modules.tracker import Track
from modules.coord_transform import CoordTransformer, WorldPoint
from modules.predictor import TrajectoryPrediction


MODEL_COLORS = {
    "constant_velocity":     (0, 255, 255),   # yellow (BGR)
    "constant_acceleration": (255, 0, 255),   # magenta
    "lstm":                  (0, 165, 255),   # orange
}


# ── Video frame annotation ────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, det: Optional[Detection]) -> np.ndarray:
    if det is None:
        return frame
    x1, y1, x2, y2 = [int(v) for v in det.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{det.class_name} {det.conf:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_track(frame: np.ndarray, track: Optional[Track]) -> np.ndarray:
    if track is None:
        return frame
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, f"ID {track.track_id}", (x1, max(0, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    return frame


def draw_predictions(
    frame: np.ndarray,
    predictions: List[TrajectoryPrediction],
    ego, transformer: CoordTransformer, K: np.ndarray,
) -> np.ndarray:
    """Project each predicted future point back into the image and draw it."""
    for pred in predictions:
        color = MODEL_COLORS.get(pred.model_name, (200, 200, 200))
        for (wx, wy) in pred.points:
            uv = transformer.world_to_image(WorldPoint(x=wx, y=wy), ego, K)
            u, v = uv
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                cv2.circle(frame, (int(u), int(v)), 4, color, -1)
        # legend dot
    _draw_legend(frame, [p.model_name for p in predictions])
    return frame


def draw_info_panel(frame: np.ndarray, frame_idx: int, ego, npc_gt,
                     est_dist: Optional[float] = None) -> np.ndarray:
    lines = [f"Frame: {frame_idx}",
             f"Ego speed: {ego.speed:.2f} m/s",
             f"Ego yaw: {ego.yaw:.2f} deg"]
    if npc_gt is not None:
        lines.append(f"GT dist_to_ego: {npc_gt.dist_to_ego:.2f} m")
    if est_dist is not None:
        lines.append(f"Est. depth: {est_dist:.2f} m")

    y0 = 30
    for i, line in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _draw_legend(frame: np.ndarray, model_names: List[str]):
    x0, y0 = frame.shape[1] - 320, 30
    for i, name in enumerate(model_names):
        color = MODEL_COLORS.get(name, (200, 200, 200))
        y = y0 + i * 28
        cv2.circle(frame, (x0, y - 5), 6, color, -1)
        cv2.putText(frame, name, (x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)


# ── Video writer helper ───────────────────────────────────────────────────────

class VideoWriter:
    def __init__(self, path: str, size, fps: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_trajectory_overview(ego_df: pd.DataFrame, npcs_df: pd.DataFrame,
                              results_df: pd.DataFrame, out_dir: str):
    """Bird's-eye-view plot of ego path, NPC GT path, and estimated NPC path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.plot(ego_df["x"], ego_df["y"], label="Ego (GT)", color="green", lw=2)
    plt.plot(npcs_df["x"], npcs_df["y"], label="NPC (GT)", color="blue", lw=2)

    if "est_x" in results_df.columns:
        plt.plot(results_df["est_x"], results_df["est_y"],
                 label="NPC (Estimated/Filtered)", color="red",
                 lw=1.5, linestyle="--")

    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.title("Bird's-Eye View — Trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)
    out_path = Path(out_dir) / "trajectory_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Trajectory overview saved → {out_path}")


def plot_error_curves(metrics_df: pd.DataFrame, out_dir: str):
    """Plot ADE / FDE / RMSE per frame for each prediction model."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        print("[Viz] No metrics to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for metric, ax in zip(["ade", "fde", "rmse"], axes):
        for model_name, g in metrics_df.groupby("model_name"):
            ax.plot(g["frame_idx"], g[metric], label=model_name)
        ax.set_ylabel(metric.upper() + " (m)")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Prediction Error Over Time")
    out_path = Path(out_dir) / "error_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Error curves saved → {out_path}")


def plot_position_estimate_vs_gt(results_df: pd.DataFrame, npcs_df: pd.DataFrame, out_dir: str):
    """Compare estimated NPC x/y over time vs ground truth."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, col, label in zip(axes, ["x", "y"], ["X position (m)", "Y position (m)"]):
        ax.plot(npcs_df.index, npcs_df[col], label="Ground truth", color="blue")
        if f"est_{col}" in results_df.columns:
            ax.plot(results_df["frame_idx"], results_df[f"est_{col}"],
                    label="Estimated (filtered)", color="red", linestyle="--")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("NPC Position Estimate vs Ground Truth")
    out_path = Path(out_dir) / "position_vs_gt.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Position comparison saved → {out_path}")
