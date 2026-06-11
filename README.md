# ADAS Kinematic Prediction Pipeline

Predicts a target (NPC) vehicle's kinematics from ego-vehicle kinematics
(`ego.csv`) and a forward-facing CARLA-simulator video (`video.avi`),
validated against ground-truth NPC kinematics (`npcs.csv`).

This README assumes **Windows 11 + VS Code's integrated terminal**
(PowerShell), using a standard Python `venv` — no Conda required.

---

## 0. Prerequisites

- **Python 3.10 or 3.11** installed and on PATH
  (check with `python --version` in the VS Code terminal)
- **Git for Windows** (https://git-scm.com/download/win) — needed for
  `scripts/setup_repos.ps1`
- VS Code with the **Python extension** installed

---

## 1. Open the project in VS Code

```powershell
cd path\to\adas_pipeline
code .
```

Open a new terminal in VS Code: **Terminal → New Terminal** (defaults
to PowerShell on Windows 11).

---

## 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **If you get an error** like *"running scripts is disabled on this
> system"*, run this once (per user, no admin needed) and retry:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

Once activated, your prompt should show `(venv)`. VS Code may also
prompt *"Select Interpreter"* — pick the one inside `.\venv\`.

---

## 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch**: needed for the optional LSTM trajectory
> predictor (`scripts/train_lstm.py`, `modules/lstm_model.py`) and
> for Depth Anything v2. Install the CPU build with:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> or follow https://pytorch.org/get-started/locally/ for a GPU build.

---

## 4. Clone external repos (DeepSORT, optional Depth Anything v2)

```powershell
.\scripts\setup_repos.ps1                     # DeepSORT only
.\scripts\setup_repos.ps1 -WithDepthAnything  # + Depth Anything v2
```

If `deep_sort\` is not present, `tracker.py` automatically falls back
to a built-in IoU tracker — no code changes needed, you can skip this
step entirely for a quick first run.

---

## 5. Add your data

Place the three input files in `data\`:

```
data\ego.csv
data\npcs.csv
data\video.avi
```

---

## 6. Run the pipeline

```powershell
python main.py --config config\config.yaml
```

Useful flags:

```powershell
python main.py --config config\config.yaml --max-frames 50   # quick test
python main.py --config config\config.yaml --no-video        # skip annotated video (faster)
```

The first run will auto-download the YOLOv8 weights (`yolov8m.pt`,
~50 MB) into the project folder.

---

## 7. Outputs (in `outputs\`)

- `results.csv` — per-frame estimated NPC state vs ground truth
- `metrics.csv` — aggregated ADE / FDE / RMSE per prediction model
- `metrics_per_frame.csv` — per-frame error breakdown
- `annotated.avi` — video with detection box, track ID, and predicted trajectories overlaid
- `plots\trajectory_overview.png` — bird's-eye view of ego/NPC/estimated paths
- `plots\error_curves.png` — ADE/FDE/RMSE over time
- `plots\position_vs_gt.png` — estimated vs ground-truth x/y over time

---

## 8. (Optional) Train the LSTM predictor

```powershell
python scripts\train_lstm.py --config config\config.yaml
```

Then add `"lstm"` to `prediction.models` in `config\config.yaml` and
re-run `main.py`.

> Note: the bundled LSTM predicts in the ego-relative frame; if you
> enable it, double-check `predict_lstm()` in `modules\predictor.py`
> converts its output back to world coordinates before it's used by
> the validator/visualizer (see code comments).

---

## 9. CARLA-specific notes

- Camera FOV assumed 90° (CARLA default) — change `carla.fov_degrees`
  in `config\config.yaml` if your recording used a different value.
- `depth.known_vehicle_height_m = 1.34` matches `vehicle.audi.tt`
  (the NPC type in `npcs.csv`). Update if your NPC vehicle differs.
- Yaw is in degrees, left-hand (UE4) convention — already handled in
  `modules\coord_transform.py`.
- `npcs.csv`'s `dist_to_ego` column is used as a free sanity check /
  scale calibration signal for depth estimation.

---

## 10. Troubleshooting (Windows-specific)

| Issue | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `git` not recognized | Install Git for Windows, restart VS Code terminal |
| `cv2.VideoWriter` produces a 0 KB / unplayable `annotated.avi` | The XVID codec sometimes isn't available; install the K-Lite Codec Pack, or change the fourcc in `utils\visualization.py` (`VideoWriter`) from `"XVID"` to `"mp4v"` and rename the output to `.mp4` |
| `pip install` fails on `tensorflow` | Only required if using the DeepSORT appearance encoder; the built-in IoU tracker works without it — remove `tensorflow` from `requirements.txt` if you don't need DeepSORT |
| Long file paths error during `git clone` | Run once: `git config --system core.longpaths true` (in an elevated PowerShell) |
