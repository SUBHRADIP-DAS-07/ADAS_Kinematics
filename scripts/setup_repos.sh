#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# scripts/setup_repos.sh
#
# Clones external repositories used by the pipeline.
# Run from the project root: bash scripts/setup_repos.sh
# ─────────────────────────────────────────────────────────
set -e

echo "== ADAS pipeline — external repo setup =="

# 1. DeepSORT (Module 4 — tracking). Skip if you're happy with the
#    built-in IoU fallback tracker (tracker.py auto-falls-back).
if [ ! -d "deep_sort" ]; then
    echo "[setup] Cloning DeepSORT..."
    git clone https://github.com/nwojke/deep_sort.git
else
    echo "[setup] deep_sort/ already exists, skipping."
fi

# 2. Depth Anything v2 (Module 3 — only needed if
#    depth.method == "depth_anything" in config.yaml)
if [ "$1" == "--with-depth-anything" ]; then
    if [ ! -d "Depth-Anything-V2" ]; then
        echo "[setup] Cloning Depth-Anything-V2..."
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git
        echo "[setup] Download a checkpoint manually from:"
        echo "        https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
        echo "        and place it at checkpoints/depth_anything_v2_vits.pth"
    else
        echo "[setup] Depth-Anything-V2/ already exists, skipping."
    fi
else
    echo "[setup] Skipping Depth-Anything-V2 (pass --with-depth-anything to include it)."
fi

echo "== Done. =="
