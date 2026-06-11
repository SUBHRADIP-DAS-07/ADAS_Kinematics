"""
modules/data_loader.py
─────────────────────────────────────────────────────────
Module 1 — Data Ingestion & Frame Synchronisation

Loads ego.csv, npcs.csv, and video.avi. Provides a unified
iterator that yields (frame_idx, ego_state, npc_gt, bgr_frame)
tuples synchronised on frame_idx.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class EgoState:
    """One row of ego.csv as a typed struct."""
    frame_idx: int
    sim_time:  float
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    pitch: float; yaw: float; roll: float
    ax: float; ay: float; az: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm([self.vx, self.vy]))


@dataclass
class NpcState:
    """One row of npcs.csv as a typed struct (ground-truth target)."""
    frame_idx: int
    sim_time:  float
    actor_id:  int
    type_id:   str
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    dist_to_ego: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


# ── DataLoader ───────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and synchronises all three data sources.

    Usage
    -----
    loader = DataLoader(cfg)
    for frame_idx, ego, npc, frame in loader:
        ...  # npc is None if npcs.csv has no entry for this frame
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._ego_df  = self._load_ego(cfg["data"]["ego_csv"])
        self._npcs_df = self._load_npcs(cfg["data"]["npcs_csv"])
        self._cap     = cv2.VideoCapture(cfg["data"]["video"])
        self._validate()

    # ── Public API ───────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Tuple[int, EgoState, Optional[NpcState], np.ndarray]]:
        for frame_idx in self._ego_df.index:
            ego   = self._get_ego(frame_idx)
            npc   = self._get_npc(frame_idx)
            frame = self._get_video_frame(frame_idx)
            if frame is None:
                continue
            yield frame_idx, ego, npc, frame

    def __len__(self) -> int:
        return len(self._ego_df)

    def get_ego(self, frame_idx: int) -> EgoState:
        return self._get_ego(frame_idx)

    def get_npc(self, frame_idx: int) -> Optional[NpcState]:
        return self._get_npc(frame_idx)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._get_video_frame(frame_idx)

    def get_dt(self, frame_idx: int) -> float:
        """Time delta (seconds) from previous frame. Uses sim_time for accuracy."""
        prev_idx = max(self._ego_df.index.min(), frame_idx - 1)
        dt = float(self._ego_df.at[frame_idx, "sim_time"] -
                   self._ego_df.at[prev_idx,  "sim_time"])
        return dt if dt > 0 else self.cfg["kalman"]["default_dt"]

    @property
    def ego_df(self) -> pd.DataFrame:
        return self._ego_df

    @property
    def npcs_df(self) -> pd.DataFrame:
        return self._npcs_df

    @property
    def frame_indices(self):
        return self._ego_df.index.tolist()

    @property
    def video_fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def video_size(self) -> Tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def close(self):
        self._cap.release()

    def __del__(self):
        try:
            self._cap.release()
        except Exception:
            pass

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_ego(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.set_index("frame_idx")
        print(f"[DataLoader] ego.csv  — {len(df)} frames, "
              f"cols: {list(df.columns)}")
        return df

    def _load_npcs(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.set_index("frame_idx")
        n_actors = df["actor_id"].nunique() if "actor_id" in df.columns else "?"
        print(f"[DataLoader] npcs.csv — {len(df)} frames, "
              f"{n_actors} unique actor(s)")
        return df

    def _get_ego(self, frame_idx: int) -> EgoState:
        row = self._ego_df.loc[frame_idx]
        return EgoState(
            frame_idx=frame_idx,
            sim_time=float(row["sim_time"]),
            x=float(row["x"]),   y=float(row["y"]),   z=float(row["z"]),
            vx=float(row["vx"]), vy=float(row["vy"]), vz=float(row["vz"]),
            pitch=float(row["pitch"]), yaw=float(row["yaw"]), roll=float(row["roll"]),
            ax=float(row["ax"]), ay=float(row["ay"]), az=float(row["az"]),
        )

    def _get_npc(self, frame_idx: int) -> Optional[NpcState]:
        if frame_idx not in self._npcs_df.index:
            return None
        row = self._npcs_df.loc[frame_idx]
        return NpcState(
            frame_idx=frame_idx,
            sim_time=float(row["sim_time"]),
            actor_id=int(row["actor_id"]),
            type_id=str(row["type_id"]),
            x=float(row["x"]),   y=float(row["y"]),   z=float(row["z"]),
            vx=float(row["vx"]), vy=float(row["vy"]), vz=float(row["vz"]),
            dist_to_ego=float(row["dist_to_ego"]),
        )

    def _get_video_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        # frame_idx starts at 1; VideoCapture is 0-based
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = self._cap.read()
        return frame if ret else None

    def _validate(self):
        n_ego   = len(self._ego_df)
        n_npcs  = len(self._npcs_df)
        n_video = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[DataLoader] Video  — {n_video} frames @ {fps:.1f} fps, "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        # warn if counts diverge
        if abs(n_ego - n_video) > 2:
            print(f"[DataLoader] WARNING: ego rows ({n_ego}) vs "
                  f"video frames ({n_video}) differ by > 2")
        print(f"[DataLoader] Ready — {n_ego} ego / {n_npcs} npc rows aligned.")
