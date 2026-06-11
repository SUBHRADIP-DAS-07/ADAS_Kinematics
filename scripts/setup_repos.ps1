# ─────────────────────────────────────────────────────────
# scripts/setup_repos.ps1
#
# Clones external repositories used by the pipeline.
# Run from the project root in PowerShell (or VS Code terminal):
#
#   .\scripts\setup_repos.ps1
#   .\scripts\setup_repos.ps1 -WithDepthAnything
#
# Requires Git for Windows: https://git-scm.com/download/win
# ─────────────────────────────────────────────────────────
param(
    [switch]$WithDepthAnything
)

Write-Host "== ADAS pipeline - external repo setup =="

# 1. DeepSORT (Module 4 - tracking). Skip if you're happy with the
#    built-in IoU fallback tracker (tracker.py auto-falls-back).
if (-not (Test-Path "deep_sort")) {
    Write-Host "[setup] Cloning DeepSORT..."
    git clone https://github.com/nwojke/deep_sort.git
} else {
    Write-Host "[setup] deep_sort/ already exists, skipping."
}

# 2. Depth Anything v2 (Module 3 - only needed if
#    depth.method == "depth_anything" in config.yaml)
if ($WithDepthAnything) {
    if (-not (Test-Path "Depth-Anything-V2")) {
        Write-Host "[setup] Cloning Depth-Anything-V2..."
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git
        Write-Host "[setup] Download a checkpoint manually from:"
        Write-Host "        https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
        Write-Host "        and place it at checkpoints\depth_anything_v2_vits.pth"
    } else {
        Write-Host "[setup] Depth-Anything-V2/ already exists, skipping."
    }
} else {
    Write-Host "[setup] Skipping Depth-Anything-V2 (pass -WithDepthAnything to include it)."
}

Write-Host "== Done. =="
