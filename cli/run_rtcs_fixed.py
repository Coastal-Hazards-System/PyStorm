"""
cli/run_rtcs_fixed.py
======================
RTCS Selection (fixed k) — Select a Representative TC Subset (RTCS) with a
fixed number of storms that best fills both the TC parameter space (X) and
the hydrodynamic response space (Y).

Usage
-----
  1. Set CONFIG overrides below (or leave empty to use all defaults).
  2. Run:  python cli/run_rtcs_selection.py

Input:   data/processed/tc_data.h5  (produced by cli/preprocess.py)
Outputs: data/processed/outputs/    (selected_storms.csv, selection_metrics.csv,
                                     pca_yspace_initial.png, pca_yspace_final.png)

All default values are defined in config/defaults.py.
Only keys you want to change need to appear in CONFIG below.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.orch.python.workflows.rtcs_selection import run_rtcs_selection

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK  — only override what you need; omit the rest
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":    str(_ROOT / "data/processed/tc_data.h5"),
    "output_dir": str(_ROOT / "data/processed/outputs"),

    # Additional storms to select ON TOP of pre-selected (total = pre + additional)
    "k_additional": 200,

    # Node subsampling (None = use all nodes)
    "node_stride": 100,

    # TC parameter SPLOM — 0-based column indices in X (Param_MT)
    "splom_params": [6, 7, 8, 9],
    "splom_labels": ["Dp (hPa)", "Rm (km)", "Vt (km/h)", "Hd (deg)"],
    "splom_title":  "TC Parameters",

    # Pre-selected storms to build upon (shown as blue in SPLOM)
    "pre_selected_csv": str(_ROOT / "data/raw/LPV/LACS_100_TC_Subset.csv"),
    # "pre_selected_indices": [0, 5, 42],   # alternative: explicit list

    # Joint matrix weights
    # "alpha_default": 1.0,
    # "beta_default":  1.0,
}
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    indices, metrics = run_rtcs_selection(CONFIG)
