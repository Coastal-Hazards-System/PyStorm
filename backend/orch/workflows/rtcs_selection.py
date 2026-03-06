"""
backend/orch/workflows/rtcs_selection.py
==========================================
RTCS (Representative TC Subset) selection pipeline orchestration.

Owns the pipeline sequence:
  load → PCA → joint matrix → growth loop → sensitivity → plots → save

Calls engine modules for all compute; owns only the sequencing and logging.

Public API
----------
  run_pipeline(cfg)  -> (final_indices, final_metrics, history, sens_df)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.io.store import read_store
from backend.engines.sampling.pca import reduce_output
from backend.engines.sampling.joint_matrix import build_joint_matrix
from backend.engines.sampling.kmedoids import select_kmedoids
from backend.engines.sampling.metrics import evaluate_sf_metrics
from backend.engines.weights.dsw import evaluate_hc_metrics
from backend.orch.postproc.plots import (
    plot_growth_history, plot_sensitivity, plot_pca_coverage,
)
from config.defaults import SUBSET_SELECTION_DEFAULTS


# ---------------------------------------------------------------------------
# Data loading shim  (bridges store / CSV to pipeline arrays)
# ---------------------------------------------------------------------------

def _load_pipeline_data(cfg: dict):
    """
    Load X, Y, HC_bench from HDF5 store or CSV fallback.
    Overrides cfg["TBL_AER"] from /HC attrs when loading from HDF5.
    Returns: X, Y, HC_bench, storm_ids, x_cols
    """
    h5 = cfg.get("h5_path")
    if h5:
        print(f"    Source : HDF5  ({h5})")
        data = read_store(Path(h5))
        X         = data.X
        Y         = data.Y
        HC_bench  = data.HC
        storm_ids = data.storm_ids
        x_cols    = data.param_names
        if data.aer_levels is not None:
            cfg["TBL_AER"] = data.aer_levels
    else:
        print(f"    Source : CSV  ({cfg['X_path']}, {cfg['Y_path']})")
        X_df = pd.read_csv(cfg["X_path"])
        Y_df = pd.read_csv(cfg["Y_path"])
        storm_ids = None
        if cfg.get("storm_id_column") and cfg["storm_id_column"] in X_df.columns:
            storm_ids = X_df[cfg["storm_id_column"]].astype(str).tolist()
            X_df = X_df.drop(columns=[cfg["storm_id_column"]])
        if cfg.get("X_columns"):
            X_df = X_df[cfg["X_columns"]]
        if cfg.get("Y_columns"):
            Y_df = Y_df[cfg["Y_columns"]]
        x_cols   = list(X_df.columns)
        X        = X_df.values.astype(float)
        Y        = Y_df.values.astype(float)
        HC_bench = None
        if cfg.get("HC_path"):
            HC_bench = pd.read_csv(cfg["HC_path"]).values.astype(float)

    assert X.shape[0] == Y.shape[0], (
        f"Row mismatch: X has {X.shape[0]} storms, Y has {Y.shape[0]}")
    if HC_bench is not None:
        assert HC_bench.shape[0] == Y.shape[1], (
            f"HC_bench has {HC_bench.shape[0]} rows but Y has {Y.shape[1]} columns")
        assert HC_bench.shape[1] == len(cfg["TBL_AER"]), (
            f"HC_bench has {HC_bench.shape[1]} columns but TBL_AER has "
            f"{len(cfg['TBL_AER'])} levels")
        print(f"    HC_bench : {HC_bench.shape}  (nodes x AER levels)")
    print(f"    X        : {X.shape}  (storms x parameters)  {x_cols}")
    print(f"    Y        : {Y.shape}  (storms x nodes)")

    return X, Y, HC_bench, storm_ids, x_cols


# ---------------------------------------------------------------------------
# Growth loop  (verbatim from Section 7)
# ---------------------------------------------------------------------------

def grow_subset(Z_full, X_scaled, Y_r_full, Y_full, HC_bench, cfg):
    """
    Grow subset from k_initial to k_max until all active thresholds are met.
    Returns  final_indices, history
    """
    print("\n[5] Iterative growth loop ...")
    print(f"    k_initial={cfg['k_initial']},  k_max={cfg['k_max']},  "
          f"step={cfg['k_step']}")
    if HC_bench is not None:
        thr = cfg.get("rmse_threshold")
        print(f"    HC threshold : mean_rmse <= {thr}"
              if thr is not None else
              "    HC metrics logged (no RMSE stopping threshold)")

    tbl_aer = cfg["TBL_AER"]
    dry_thr = cfg["dry_threshold"]
    do_hc   = HC_bench is not None
    history = []
    k       = cfg["k_initial"]

    while True:
        indices = select_kmedoids(Z_full, k, cfg["random_seed"])
        sf  = evaluate_sf_metrics(Z_full, X_scaled, Y_r_full, indices,
                                  cfg["n_coverage_clusters"], cfg["random_seed"])
        row = dict(sf)
        if do_hc:
            row.update(evaluate_hc_metrics(
                Y_full[indices, :], HC_bench, tbl_aer, dry_thr,
                cfg.get("min_wet_storms", 2)))
        history.append(row)

        msg = (f"    k={k:4d} | cov={sf['coverage']:.3f} | "
               f"disc={sf['discrepancy']:.4f} | maximin={sf['maximin']:.4f}")
        if do_hc:
            msg += (f" | bias={row['mean_bias']:+.4f} | "
                    f"unc={row['mean_uncertainty']:.4f} | "
                    f"rmse={row['mean_rmse']:.4f}")
        print(msg)

        cov_ok  = sf["coverage"]    >= cfg["coverage_threshold"]
        disc_ok = sf["discrepancy"] <= cfg["discrepancy_threshold"]
        rmse_ok = True
        if do_hc and cfg.get("rmse_threshold") is not None:
            rmse_ok = row["mean_rmse"] <= cfg["rmse_threshold"]

        if cov_ok and disc_ok and rmse_ok:
            print(f"\n    All thresholds met at k={k}. Stopping.")
            break
        if k >= cfg["k_max"]:
            print(f"\n    k_max={cfg['k_max']} reached. Returning current subset.")
            break
        k = min(k + cfg["k_step"], cfg["k_max"])

    return indices, history


# ---------------------------------------------------------------------------
# Sensitivity analysis  (verbatim from Section 8)
# ---------------------------------------------------------------------------

def sensitivity_analysis(X, Y_r, Y_full, HC_bench, cfg):
    print(f"\n[6] Sensitivity analysis — sweeping alpha  "
          f"(k={cfg['k_sensitivity']}, beta=1 fixed) ...")
    tbl_aer = cfg["TBL_AER"]
    dry_thr = cfg["dry_threshold"]
    do_hc   = HC_bench is not None
    k       = cfg["k_sensitivity"]
    results = []

    for alpha in cfg["alpha_sweep"]:
        Z, scaler_X, _ = build_joint_matrix(X, Y_r, alpha, 1.0)
        X_sc            = scaler_X.transform(X)
        indices = select_kmedoids(Z, k, cfg["random_seed"])
        sf      = evaluate_sf_metrics(Z, X_sc, Y_r, indices,
                                      cfg["n_coverage_clusters"], cfg["random_seed"])
        row = {"alpha": alpha, "beta": 1.0, **sf}
        if do_hc:
            row.update(evaluate_hc_metrics(
                Y_full[indices, :], HC_bench, tbl_aer, dry_thr,
                cfg.get("min_wet_storms", 2)))
        results.append(row)
        msg = (f"    alpha={alpha:.2f} | cov={sf['coverage']:.3f} | "
               f"disc={sf['discrepancy']:.4f}")
        if do_hc:
            msg += (f" | bias={row['mean_bias']:+.4f} | "
                    f"unc={row['mean_uncertainty']:.4f} | "
                    f"rmse={row['mean_rmse']:.4f}")
        print(msg)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Save results  (verbatim from Section 10)
# ---------------------------------------------------------------------------

def save_results(X, indices, storm_ids, x_cols, final_metrics, history, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sel = pd.DataFrame(X[indices], columns=x_cols)
    if storm_ids is not None:
        df_sel.insert(0, "storm_id", [storm_ids[i] for i in indices])
    df_sel.insert(0, "original_index", indices)
    df_sel.to_csv(out_dir / "selected_storms.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "growth_history.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(out_dir / "final_metrics.csv", index=False)
    print(f"\n    selected_storms.csv -> {out_dir}")
    print(f"    growth_history.csv  -> {out_dir}")
    print(f"    final_metrics.csv   -> {out_dir}")


# ---------------------------------------------------------------------------
# Top-level pipeline  (verbatim from Section 11)
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Optional[dict] = None):
    """
    Execute the full subset-selection pipeline.

    Returns  final_indices, final_metrics, history, sens_df
    """
    base = copy.deepcopy(SUBSET_SELECTION_DEFAULTS)
    if cfg:
        base.update(cfg)
    cfg = base

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["random_seed"])

    # 1. Load
    print("\n[1] Loading data ...")
    X, Y, HC_bench, storm_ids, x_cols = _load_pipeline_data(cfg)

    # 2. PCA
    print(f"\n[2] PCA on Y  (retaining {cfg['pca_variance_threshold']*100:.0f}% variance) ...")
    Y_r, pca = reduce_output(Y, cfg["pca_variance_threshold"])
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"    Components retained : {Y_r.shape[1]}")
    print(f"    Variance explained  : {cumvar[-1]*100:.2f}%")

    # 3. Joint matrix
    print(f"\n[3] Building joint matrix  "
          f"(alpha={cfg['alpha_default']}, beta={cfg['beta_default']}) ...")
    Z, scaler_X, _ = build_joint_matrix(
        X, Y_r, cfg["alpha_default"], cfg["beta_default"])
    X_scaled = scaler_X.transform(X)

    # 4. Growth loop
    final_indices, history = grow_subset(Z, X_scaled, Y_r, Y, HC_bench, cfg)

    # 5. Final metrics
    final_metrics = dict(evaluate_sf_metrics(
        Z, X_scaled, Y_r, final_indices,
        cfg["n_coverage_clusters"], cfg["random_seed"]))
    if HC_bench is not None:
        final_metrics.update(evaluate_hc_metrics(
            Y[final_indices, :], HC_bench, cfg["TBL_AER"], cfg["dry_threshold"],
            cfg.get("min_wet_storms", 2)))

    print(f"\n    Final subset  k = {final_metrics['k']}")
    print(f"    Coverage          = {final_metrics['coverage']:.4f}  "
          f"(threshold >= {cfg['coverage_threshold']})")
    print(f"    Discrepancy       = {final_metrics['discrepancy']:.4f}  "
          f"(threshold <= {cfg['discrepancy_threshold']})")
    print(f"    Maximin           = {final_metrics['maximin']:.4f}")
    if HC_bench is not None:
        print(f"    Mean Bias         = {final_metrics['mean_bias']:+.4f}  m")
        print(f"    Mean Uncertainty  = {final_metrics['mean_uncertainty']:.4f}  m")
        print(f"    Mean RMSE         = {final_metrics['mean_rmse']:.4f}  m")

    # 6. Sensitivity
    sens_df = sensitivity_analysis(X, Y_r, Y, HC_bench, cfg)
    sens_df.to_csv(out_dir / "sensitivity_alpha.csv", index=False)

    # 7. Plots
    print("\n[7] Generating plots ...")
    plot_growth_history(history, cfg, out_dir)
    plot_sensitivity(sens_df, cfg, out_dir)
    plot_pca_coverage(Y_r, Y_r[final_indices], out_dir)

    # 8. Save
    save_results(X, final_indices, storm_ids, x_cols,
                 final_metrics, history, out_dir)

    print("\n=== Pipeline complete ===")
    return final_indices, final_metrics, history, sens_df
