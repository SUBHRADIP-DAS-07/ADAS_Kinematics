"""
main.py
─────────────────────────────────────────────────────────
ADAS Kinematic Prediction Pipeline — Main Runner

Wires together all 8 modules:
  1. DataLoader        — ego.csv / npcs.csv / video.avi sync
  2. VehicleDetector    — YOLOv8
  3. DepthEstimator     — bbox geometry / Depth Anything v2
  4. VehicleTracker     — DeepSORT / IoU fallback
  5. CoordTransformer   — camera -> CARLA world
  6. NpcKalmanFilter    — state estimation [x,y,vx,vy]
  7. TrajectoryPredictor— CV / CA / LSTM
  8. Validator          — ADE / FDE / RMSE vs npcs.csv

Usage
-----
    python main.py --config config/config.yaml
    python main.py --config config/config.yaml --no-video   # skip annotated video (faster)
    python main.py --config config/config.yaml --max-frames 100
"""

from __future__ import annotations
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from modules.data_loader import DataLoader
from modules.detector import VehicleDetector
from modules.depth_estimator import DepthEstimator
from modules.tracker import VehicleTracker
from modules.coord_transform import CoordTransformer, WorldPoint
from modules.kalman_filter import NpcKalmanFilter
from modules.predictor import TrajectoryPredictor
from modules.validator import Validator

from utils.visualization import (
    draw_detection, draw_track, draw_predictions, draw_info_panel,
    VideoWriter, plot_trajectory_overview, plot_error_curves,
    plot_position_estimate_vs_gt,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(cfg: dict, max_frames: int | None = None, write_video: bool = True):
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["plots_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Initialise modules ───────────────────────────────────────────────────
    loader      = DataLoader(cfg)
    detector    = VehicleDetector(cfg)
    depth_est   = DepthEstimator(cfg)
    tracker     = VehicleTracker(cfg)
    transformer = CoordTransformer(cfg)
    kf          = NpcKalmanFilter(cfg)
    predictor   = TrajectoryPredictor(cfg)
    validator   = Validator(cfg, loader.npcs_df)

    K = depth_est.K
    img_w, img_h = loader.video_size

    # ── Output video writer ──────────────────────────────────────────────────
    writer = None
    if write_video:
        writer = VideoWriter(cfg["output"]["annotated_video"],
                             (img_w, img_h), loader.video_fps)

    history_buf = []   # rolling buffer of [x,y,vx,vy] for LSTM
    obs_window  = cfg.get("lstm", {}).get("obs_window", 20)

    results = []

    frame_indices = loader.frame_indices
    if max_frames:
        frame_indices = frame_indices[:max_frames]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for frame_idx in tqdm(frame_indices, desc="Processing frames"):
        ego   = loader.get_ego(frame_idx)
        npc_gt = loader.get_npc(frame_idx)
        frame  = loader.get_frame(frame_idx)
        if frame is None:
            continue

        dt = loader.get_dt(frame_idx)

        # 2. Detection
        detections = detector.detect(frame)

        # 4. Tracking
        tracks = tracker.update(detections, frame)
        target = tracker.pick_target(tracks, img_w)
        det    = target.det if target is not None else detector.pick_best(detections)

        # 3. Depth + 3D point (camera frame)
        measured_world = None
        est_depth = None
        if det is not None:
            depth_result = depth_est.estimate(
                det, frame=frame,
                gt_dist_m=(npc_gt.dist_to_ego if npc_gt is not None else None),
            )
            est_depth = depth_result.depth_m

            # 5. Coordinate transform: camera -> world
            world_pt = transformer.camera_to_world(depth_result.point_cam, ego)
            measured_world = world_pt.to_array()

        # 6. Kalman filter update
        state = kf.step(measured_world, dt)

        # Update LSTM history buffer (relative-to-ego representation)
        rel = transformer.relative_to_ego(WorldPoint(x=state.x, y=state.y), ego)
        history_buf.append([rel[0], rel[1], state.vx, state.vy])
        if len(history_buf) > obs_window:
            history_buf.pop(0)

        history_arr = (np.array(history_buf) if len(history_buf) == obs_window else None)

        # 7. Trajectory prediction
        predictions = predictor.predict_all(
            state, dt=dt, ego_ax=ego.ax, ego_ay=ego.ay, history=history_arr,
        )

        # 8. Validation
        for pred in predictions:
            validator.evaluate(frame_idx, pred)

        # ── Record results row ──────────────────────────────────────────────
        row = {
            "frame_idx": frame_idx,
            "est_x": state.x, "est_y": state.y,
            "est_vx": state.vx, "est_vy": state.vy,
            "est_depth": est_depth,
            "det_conf": det.conf if det is not None else np.nan,
        }
        if npc_gt is not None:
            row.update({"gt_x": npc_gt.x, "gt_y": npc_gt.y,
                        "gt_dist_to_ego": npc_gt.dist_to_ego})
        results.append(row)

        # ── Annotate video frame ────────────────────────────────────────────
        if writer is not None:
            frame = draw_detection(frame, det)
            if target is not None:
                frame = draw_track(frame, target)
            frame = draw_predictions(frame, predictions, ego, transformer, K)
            frame = draw_info_panel(frame, frame_idx, ego, npc_gt, est_depth)
            writer.write(frame)

    if writer is not None:
        writer.release()
    loader.close()

    # ── Save results & metrics ───────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg["output"]["results_csv"], index=False)
    print(f"\n[Main] Per-frame results saved → {cfg['output']['results_csv']}")

    validator.save(str(Path(cfg["output"]["dir"]) / "metrics_per_frame.csv"))
    validator.save_summary(cfg["output"]["metrics_csv"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_trajectory_overview(loader.ego_df, loader.npcs_df, results_df,
                             cfg["output"]["plots_dir"])
    plot_error_curves(validator.to_dataframe(), cfg["output"]["plots_dir"])
    plot_position_estimate_vs_gt(results_df, loader.npcs_df, cfg["output"]["plots_dir"])

    print("\n[Main] Pipeline complete.")
    if writer is not None:
        print(f"[Main] Annotated video → {cfg['output']['annotated_video']}")


def main():
    parser = argparse.ArgumentParser(description="ADAS Kinematic Prediction Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames processed (debugging)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the annotated output video")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, max_frames=args.max_frames, write_video=not args.no_video)


if __name__ == "__main__":
    main()
