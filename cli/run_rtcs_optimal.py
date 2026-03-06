"""
cli/run_rtcs_optimal.py
========================
RTCS Selection (optimal k) — Sweep subset size from k_initial to k_max,
computing DSW weights and reconstructing hazard curves at each step to find
the optimal number of storms.  Quantifies bias relative
to benchmark HCs at the full AER table and at reporting return periods.

Usage
-----
  1. Run cli/preprocess.py first to produce data/processed/tc_data.h5.
  2. Set CONFIG overrides below.
  3. Run:  python cli/run_growth_evaluation.py

Input:   data/processed/tc_data.h5  (must contain /HC benchmark curves)
Outputs: data/processed/outputs/    (selected_storms.csv, growth_history.csv,
                                     growth_evaluation.png)

All default values are defined in config/defaults.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.orch.python.workflows.growth_evaluation import run_growth_evaluation

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK  — only override what you need; omit the rest
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":    str(_ROOT / "data/processed/tc_data.h5"),
    "output_dir": str(_ROOT / "data/processed/outputs"),

    # Node subsampling (None = use all nodes)
    "node_stride": 100,

    # Growth loop bounds
    "k_initial": 20,
    "k_max":    100,
    # "k_step":    5,

    # Return periods (years) to report bias at
    "bias_report_rp": [10, 100, 1000],

    # Pre-selected storms to build upon (optional — pick one or neither)
    # "pre_selected_csv":     str(_ROOT / "data/processed/outputs/selected_storms.csv"),
    # "pre_selected_indices": [0, 5, 42],   # explicit list of original indices

    # Optional RMSE stopping threshold (m); None = run to k_max
    # "rmse_threshold": 0.20,

    # Joint matrix weights
    # "alpha_default": 1.0,
    # "beta_default":  1.0,
}
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    indices, history_df = run_growth_evaluation(CONFIG)
