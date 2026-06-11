"""
scripts/train_lstm.py
─────────────────────────────────────────────────────────
Optional — train the LSTM trajectory predictor on npcs.csv.

Builds (observation, target) pairs from the NPC ground-truth
trajectory expressed in the EGO-relative frame, then trains
modules.lstm_model.TrajLSTM to predict future relative offsets.

Usage
-----
    python scripts/train_lstm.py --config config/config.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from modules.coord_transform import CoordTransformer, WorldPoint
from modules.data_loader import EgoState
from modules.lstm_model import TrajLSTM


def build_dataset(ego_df: pd.DataFrame, npcs_df: pd.DataFrame,
                   transformer: CoordTransformer, obs_window: int, horizon: int):
    """
    Returns
    -------
    X : (N, obs_window, 4)  — [rel_x, rel_y, vx, vy] history
    Y : (N, horizon, 2)     — [d_rel_x, d_rel_y] future offsets relative
                              to the last observed position
    """
    common_idx = sorted(set(ego_df.index) & set(npcs_df.index))

    rel_xy, vxvy = [], []
    for idx in common_idx:
        e = ego_df.loc[idx]
        n = npcs_df.loc[idx]
        ego = EgoState(
            frame_idx=idx, sim_time=e["sim_time"],
            x=e["x"], y=e["y"], z=e["z"],
            vx=e["vx"], vy=e["vy"], vz=e["vz"],
            pitch=e["pitch"], yaw=e["yaw"], roll=e["roll"],
            ax=e["ax"], ay=e["ay"], az=e["az"],
        )
        rel = transformer.relative_to_ego(WorldPoint(x=n["x"], y=n["y"]), ego)
        rel_xy.append(rel)
        vxvy.append([n["vx"], n["vy"]])

    rel_xy = np.array(rel_xy)   # (T, 2)
    vxvy   = np.array(vxvy)     # (T, 2)
    state  = np.concatenate([rel_xy, vxvy], axis=1)   # (T, 4)

    X, Y = [], []
    T = len(state)
    for t in range(T - obs_window - horizon):
        x_seq = state[t : t+obs_window]                     # (obs_window, 4)
        last_xy = x_seq[-1, :2]
        future_xy = state[t+obs_window : t+obs_window+horizon, :2]  # (horizon, 2)
        y_seq = future_xy - last_xy[None, :]                 # offsets
        X.append(x_seq)
        Y.append(y_seq)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import torch
    import torch.nn as nn

    ego_df  = pd.read_csv(cfg["data"]["ego_csv"]).set_index("frame_idx")
    npcs_df = pd.read_csv(cfg["data"]["npcs_csv"]).set_index("frame_idx")
    transformer = CoordTransformer(cfg)

    obs_window = cfg["lstm"]["obs_window"]
    horizon    = cfg["prediction"]["horizon_frames"]

    X, Y = build_dataset(ego_df, npcs_df, transformer, obs_window, horizon)
    print(f"[TrainLSTM] Dataset: X={X.shape}, Y={Y.shape}")

    if len(X) < 10:
        raise RuntimeError(
            "Not enough samples to train an LSTM with this obs_window/horizon. "
            "Reduce obs_window or horizon_frames in config.yaml, or collect "
            "more frames of data."
        )

    # Simple train/val split
    n_val = max(1, int(0.15 * len(X)))
    X_train, Y_train = X[:-n_val], Y[:-n_val]
    X_val,   Y_val   = X[-n_val:], Y[-n_val:]

    model = TrajLSTM(
        input_dim=4,
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        horizon=horizon,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lstm"]["learning_rate"])
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    Y_train_t = torch.tensor(Y_train)
    X_val_t   = torch.tensor(X_val)
    Y_val_t   = torch.tensor(Y_val)

    epochs = cfg["lstm"]["train_epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, Y_val_t)
            print(f"[TrainLSTM] epoch {epoch:4d}/{epochs} "
                  f"train_loss={loss.item():.5f}  val_loss={val_loss.item():.5f}")

    ckpt_path = Path(cfg["lstm"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[TrainLSTM] Model saved → {ckpt_path}")
    print("[TrainLSTM] Add 'lstm' to prediction.models in config.yaml to enable it.")


if __name__ == "__main__":
    main()
