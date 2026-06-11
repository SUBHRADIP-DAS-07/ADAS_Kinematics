"""
modules/lstm_model.py
─────────────────────────────────────────────────────────
Optional — LSTM Trajectory Predictor (PyTorch)

Architecture
------------
Input  : sequence of (obs_window, 4) — [x, y, vx, vy] in the EGO
         relative frame (see dataset.py for frame conversion).
Output : (horizon, 2) — predicted [dx, dy] OFFSETS relative to the
         last observed position.

Training is done with scripts/train_lstm.py against npcs.csv
trajectories. This file only defines the network so it can be
imported by both training and inference code without circular
dependencies.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class TrajLSTM(nn.Module):
        """
        Simple sequence-to-vector LSTM trajectory predictor.

        Parameters
        ----------
        input_dim   : number of input features per time step (default 4: x,y,vx,vy)
        hidden_size : LSTM hidden state size
        num_layers  : number of stacked LSTM layers
        horizon     : number of future steps to predict
        """

        def __init__(self, input_dim: int = 4, hidden_size: int = 64,
                     num_layers: int = 2, horizon: int = 10):
            super().__init__()
            self.horizon = horizon
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, horizon * 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            x : (B, T, input_dim)
            returns : (B, horizon, 2)
            """
            out, (h_n, _) = self.lstm(x)
            last_hidden = out[:, -1, :]              # (B, hidden_size)
            offsets = self.head(last_hidden)          # (B, horizon*2)
            return offsets.view(-1, self.horizon, 2)  # (B, horizon, 2)

else:

    class TrajLSTM:  # type: ignore
        """Placeholder when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for TrajLSTM. "
                "Install with: pip install torch"
            )
