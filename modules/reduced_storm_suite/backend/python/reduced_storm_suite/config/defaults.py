"""Default configuration for the reduced_storm_suite module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

import numpy as np

RSS_SELECTION_DEFAULTS: dict = {
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
    "pre_selected_csv":     None,
    "pre_selected_indices": None,

    # ── Node subsampling ──────────────────────────────────────────────────────
    "node_stride": None,

    # ── X column selection ─────────────────────────────────────────────────────
    "x_select_columns": [5, 6, 7, 8],

    # ── PCA ───────────────────────────────────────────────────────────────────
    "pca_variance_threshold": 0.95,
    # How to handle NaN (dry-node markers) in Y before PCA:
    #   "drop_always_dry" - drop all-NaN columns, zero-fill remaining NaN (default)
    #   "zero"            - replace every NaN with 0.0 (preserves node count)
    #   "node_mean"       - impute NaN with the per-node mean of wet values
    #   "wet_only"        - drop any column with at least one NaN
    #   "wet_ratio_floor" - drop nodes wet for < pca_min_wet_fraction of
    #                       storms, then zero-fill remaining NaN
    "pca_dry_strategy":       "drop_always_dry",
    # Only used when pca_dry_strategy == "wet_ratio_floor": minimum fraction
    # of storms (0-1) a node must be wet for to be retained.
    "pca_min_wet_fraction":   0.2,

    # ── Subset selection method ────────────────────────────────────────────────
    # How the k storms are chosen from the joint feature space:
    #   "kmedoids" - PAM, minimizes total distance to the nearest medoid.
    #                Density-following: medoids concentrate where storms are
    #                dense (the weak-storm apex), under-sampling the rare
    #                intense storms. Best for a representative subset (default).
    #   "maximin"  - greedy farthest-point. Space-filling: spreads the subset
    #                uniformly across the feature space and deliberately reaches
    #                the extremes/tail. Best for even PC-space coverage.
    "selection_method": "kmedoids",

    # ── Fixed k: subset sizing ─────────────────────────────────────────────────
    "k_additional": 100,

    # ── Sub-RSS (optional second-stage subset) ───────────────────────────────
    "k_sub_rss":     None,
    "sub_rss_mode": "within",

    # ── Optimal k: growth loop ───────────────────────────────────────────────────
    "k_min":    20,
    "k_max":   100,
    "k_step":    5,

    # ── RMSE tolerance (optimal mode selection criterion, units: m) ───────────
    "rmse_threshold":        0.10,

    # ── Informational thresholds (logged; NOT used as stopping criteria) ──────
    "coverage_threshold":    0.90,
    "discrepancy_threshold": 0.20,

    # ── Y-space coverage ──────────────────────────────────────────────────────
    "n_coverage_clusters": 20,

    # ── Joint matrix weights ──────────────────────────────────────────────────
    "alpha_default": 10.0,
    "beta_default":  0.1,

    "alpha_beta_grid": [
        (0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0),
        (5.0, 1.0), (10.0, 1.0), (20.0, 1.0),
        (0.1, 0.1), (1.0, 0.1), (5.0, 0.1), (10.0, 0.1), (20.0, 0.1),
        (0.1, 0.5), (1.0, 0.5), (5.0, 0.5), (10.0, 0.5),
    ],

    # α/β grid sweep performance knobs.
    # ab_search_workers: None=auto (cpu_count), 1=sequential, N=use N processes
    # ab_search_node_sample: None=all nodes; int=evaluate sweep on a random
    #   subset of nodes (selection accuracy ~unchanged for N >= ~2000)
    "ab_search_workers":     None,
    "ab_search_node_sample": None,

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    "alpha_sweep":   [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "k_sensitivity": 100,

    # ── HC reconstruction ─────────────────────────────────────────────────────
    "TBL_AER": 1.0 / np.array([
        0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000,
        2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000,
    ]),
    "dry_threshold":   0.0,
    "qbm_win_frac":  0.10,
    "qbm_ramp_frac": 0.03,
    "qbm_mode": "aer",
    "qbm_aer_mode": "631",
    "dsw_method": 3,
    "bias_report_aer": [10, 100, 1000],   # AER levels (labelled by MRI year N; AER = 1/N)
    "min_wet_storms": 2,

    # ── Reproducibility ───────────────────────────────────────────────────────
    "random_seed": 42,

    # ── TC parameter SPLOM ────────────────────────────────────────────────────
    "splom_params": None,
    "splom_labels": None,
    "splom_title":  "TC Parameters",

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir": "outputs",
}
