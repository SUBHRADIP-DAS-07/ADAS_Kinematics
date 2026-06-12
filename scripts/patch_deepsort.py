"""
scripts/patch_deepsort.py
─────────────────────────────────────────────────────────
Safety-net patch for the cloned deep_sort library.

Run once after  .\scripts\setup_repos.ps1  (or setup_repos.sh):

    python scripts/patch_deepsort.py

What it does
────────────
1. deep_sort/deep_sort/nn_matching.py
   Guards zero-norm feature vectors in the cosine-distance helper.
   Zero-norm vectors arise when the Re-ID encoder returns an all-zeros
   array (our stub); dividing by their norm produces NaN which
   propagates into the cost matrix and crashes linear_sum_assignment.

2. deep_sort/deep_sort/linear_assignment.py
   Sanitises the cost matrix just before scipy's linear_sum_assignment,
   replacing any residual NaN / inf with a large finite cost (1e5) so
   the solver treats those pairs as "do not match" rather than crashing.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root

# ── helpers ──────────────────────────────────────────────────────────────────

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")
    print(f"  ✓ patched  {p.relative_to(ROOT)}")

def _already(text: str, sentinel: str) -> bool:
    return sentinel in text

# ── patch 1 — nn_matching.py  ────────────────────────────────────────────────

NN_PATH = ROOT / "deep_sort" / "deep_sort" / "nn_matching.py"

OLD_NORM = (
    "  a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)\n"
    "  b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)"
)

NEW_NORM = """\
  # Guard against zero-norm vectors (e.g. stub all-zeros Re-ID features).
  # Dividing by a zero norm produces NaN which later crashes
  # linear_sum_assignment with "matrix contains invalid numeric entries".
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  norm_a = np.linalg.norm(a, axis=1, keepdims=True)
  norm_b = np.linalg.norm(b, axis=1, keepdims=True)
  a = a / np.where(norm_a < 1e-12, 1.0, norm_a)
  b = b / np.where(norm_b < 1e-12, 1.0, norm_b)"""

def patch_nn_matching():
    if not NN_PATH.exists():
        print(f"  ⚠  not found: {NN_PATH}  — run setup_repos first")
        return
    src = _read(NN_PATH)
    if _already(src, "Guard against zero-norm"):
        print(f"  –  already patched: {NN_PATH.relative_to(ROOT)}")
        return
    if OLD_NORM not in src:
        # Try with 4-space indent (some clones differ)
        old4 = OLD_NORM.replace("  a", "    a").replace("  b", "    b")
        new4 = NEW_NORM.replace("  #", "    #").replace("  a", "    a").replace("  b", "    b").replace("  norm", "    norm")
        if old4 in src:
            _write(NN_PATH, src.replace(old4, new4))
            return
        print(f"  ⚠  pattern not found in {NN_PATH.name} — check indentation manually")
        return
    _write(NN_PATH, src.replace(OLD_NORM, NEW_NORM))

# ── patch 2 — linear_assignment.py  ─────────────────────────────────────────

LA_PATH = ROOT / "deep_sort" / "deep_sort" / "linear_assignment.py"

OLD_LSA = "    indices = np.asarray(linear_sum_assignment(cost_matrix)).T"

NEW_LSA = """\
    # Sanitise: replace NaN / inf with a large finite cost so scipy's
    # Hungarian solver never receives invalid numeric entries.
    cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1e5)
    indices = np.asarray(linear_sum_assignment(cost_matrix)).T"""

def patch_linear_assignment():
    if not LA_PATH.exists():
        print(f"  ⚠  not found: {LA_PATH}  — run setup_repos first")
        return
    src = _read(LA_PATH)
    if _already(src, "Sanitise: replace NaN"):
        print(f"  –  already patched: {LA_PATH.relative_to(ROOT)}")
        return
    if OLD_LSA not in src:
        print(f"  ⚠  pattern not found in {LA_PATH.name} — check manually")
        return
    _write(LA_PATH, src.replace(OLD_LSA, NEW_LSA))

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Patching deep_sort library...")
    patch_nn_matching()
    patch_linear_assignment()
    print("Done.")
