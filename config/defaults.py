"""
config/defaults.py
====================
Default configuration for the RTCS (Representative TC Subset) selection pipeline.

Cross-cutting: consumed by both the API layer and the orchestration workflow.
Neither owns it.
"""

import numpy as np

RTCS_SELECTION_DEFAULTS: dict = {
    # ── Input (HDF5 preferred; CSV fallback when h5_path is None) ─────────────
    "h5_path":  "tc_data.h5",
    "X_path":   "X_parameters.csv",
    "Y_path":   "Y_surges.csv",
    "HC_path":  "HC_benchmark.csv",

    # ── CSV column handling (ignored when loading from HDF5) ──────────────────
    "X_columns":       None,
    "Y_columns":       None,
    "storm_id_column": None,

    # ── Pre-selected storms ───────────────────────────────────────────────────
    # Storms that must appear in every selection.  Supply one or neither.
    "pre_selected_csv":     None,   # path to CSV with 'original_index' column
    "pre_selected_indices": None,   # explicit list of original storm indices

    # ── Node subsampling ──────────────────────────────────────────────────────
    "node_stride": None,   # int → keep every Nth node; None → use all nodes

    # ── X column selection ─────────────────────────────────────────────────────
    # 0-based column indices into X used for the joint matrix, k-medoids, and
    # space-filling metrics.  None → use all columns.
    # Default [5,6,7,8] = [Hd, Dp, Rm, Vt].  Extended: [3,4,5,6,7,8] adds lat/lon.
    "x_select_columns": [5, 6, 7, 8],

    # ── PCA ───────────────────────────────────────────────────────────────────
    "pca_variance_threshold": 0.95,

    # ── Mode 1: subset sizing ─────────────────────────────────────────────────
    # k_additional: storms to select ON TOP of any pre-selected forced storms.
    # Total medoids passed to k-medoids = len(forced) + k_additional.
    "k_additional": 100,

    # ── Mode 2: growth loop ───────────────────────────────────────────────────
    "k_initial": 20,
    "k_max":    100,
    "k_step":     5,

    # ── Stopping thresholds ───────────────────────────────────────────────────
    "coverage_threshold":    0.90,
    "discrepancy_threshold": 0.20,
    "rmse_threshold":        None,   # float (m) to activate; None to disable

    # ── Y-space coverage ──────────────────────────────────────────────────────
    "n_coverage_clusters": 20,

    # ── Joint matrix weights ──────────────────────────────────────────────────
    "alpha_default": 10.0,
    "beta_default":  0.1,

    # Alpha/beta grid for DSW-based optimization (None = skip, use defaults)
    "alpha_beta_grid": [
        (0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0),
        (5.0, 1.0), (10.0, 1.0), (20.0, 1.0),
        (0.1, 0.1), (1.0, 0.1), (5.0, 0.1), (10.0, 0.1), (20.0, 0.1),
        (0.1, 0.5), (1.0, 0.5), (5.0, 0.5), (10.0, 0.5),
    ],

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    "alpha_sweep":   [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "k_sensitivity": 100,

    # ── HC reconstruction ─────────────────────────────────────────────────────
    # Overridden automatically by /HC attrs["aer_levels"] when loading HDF5.
    "TBL_AER": 1.0 / np.array([
        0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000,
        2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000,
    ]),
    "dry_threshold":   0.0,
    # Return periods (years) at which to report mean nodal bias in Mode 2.
    # Maps to AER = 1/RP; nearest column in TBL_AER is used.
    "bias_report_rp":  [10, 100, 1000],
    # Minimum number of wet storms required at a node for it to contribute
    # to the global DSW average.  Excludes mostly-dry nodes that bias the
    # DSW signal on large meshes with sparse surge coverage.
    "min_wet_storms": 2,

    # ── Reproducibility ───────────────────────────────────────────────────────
    "random_seed": 42,

    # ── TC parameter SPLOM ────────────────────────────────────────────────────
    # Column names or 0-based indices to plot; None → first 4 columns.
    "splom_params": None,
    # Display labels with units (same length as splom_params); None → col names.
    "splom_labels": None,
    # Title printed above the grid.
    "splom_title":  "TC Parameters",

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir": "outputs",
}
