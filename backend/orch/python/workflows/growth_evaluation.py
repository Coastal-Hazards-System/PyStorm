"""
backend/orch/python/workflows/growth_evaluation.py
====================================================
RTCS Selection (optimal k) — Iterative growth loop with DSW and hazard-curve evaluation.

Sweeps subset size from k_initial to k_max (step k_step).  At each k:
  - selects a k-medoid subset from the joint parameter/response space
  - back-computes global DSW weights
  - reconstructs hazard curves at every node via JPM-OS
  - compares to benchmark HCs: logs global bias/uncertainty/RMSE and
    mean nodal bias at user-specified return period levels

Public API
----------
  run_growth_evaluation(cfg)  ->  (final_indices, history_df)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.engines.sampling.python.pca import reduce_output
from backend.engines.sampling.python.joint_matrix import build_joint_matrix
from backend.engines.sampling.python.kmedoids import select_kmedoids
from backend.engines.sampling.metrics import evaluate_sf_metrics
from backend.engines.weights.dsw import (
    compute_global_dsw,
    reconstruct_hc_global_dsw,
    evaluate_hc_metrics,
)
from backend.orch.python.workflows.rtcs_selection import (
    _load_pipeline_data, _load_forced_indices, _remap_forced_indices,
)
from backend.orch.python.postproc.plots import plot_growth_evaluation
from config.defaults import RTCS_SELECTION_DEFAULTS


def run_growth_evaluation(cfg: Optional[dict] = None):
    """
    RTCS Selection (optimal k) — Iterative growth loop with HC evaluation.

    Returns
    -------
    final_indices : ndarray  [k_final]
    history_df    : DataFrame  one row per k step
    """
    base = copy.deepcopy(RTCS_SELECTION_DEFAULTS)
    if cfg:
        base.update(cfg)
    cfg = base

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["random_seed"])

    print(f"\n{'='*60}")
    print(f"  RTCS Selection (optimal k) — Growth Loop  "
          f"(k {cfg['k_initial']} -> {cfg['k_max']}, step {cfg['k_step']})")
    print(f"{'='*60}")

    # 1. Load  (HC_bench required for DSW evaluation)
    print("\n[1] Loading data ...")
    X_full, Y, HC_bench, storm_ids, x_cols_full = _load_pipeline_data(cfg)
    forced = _remap_forced_indices(cfg, _load_forced_indices(cfg))

    # Column filter: keep only physically meaningful TC parameters
    sel_cols = cfg.get("x_select_columns")
    if sel_cols is not None:
        X = X_full[:, sel_cols]
        x_cols = [x_cols_full[i] for i in sel_cols]
        print(f"    X columns   : {sel_cols} -> {x_cols}")
    else:
        X = X_full
        x_cols = x_cols_full
    if forced is not None:
        print(f"    Forced storms : {len(forced)}  "
              f"(growth loop selects {cfg['k_initial'] - len(forced)} "
              f"-> {cfg['k_max'] - len(forced)} additional)")
    if HC_bench is None:
        raise ValueError(
            "HC_bench is required for RTCS Selection (optimal k).  "
            "Set HC_source in the preprocessor config and re-run preprocess.py.")

    tbl_aer   = cfg["TBL_AER"]
    dry_thr   = cfg["dry_threshold"]
    report_rp = cfg["bias_report_rp"]

    # 2. PCA
    print(f"\n[2] PCA on Y  (retaining {cfg['pca_variance_threshold']*100:.0f}% variance) ...")
    Y_r, pca = reduce_output(Y, cfg["pca_variance_threshold"])
    print(f"    Components retained : {Y_r.shape[1]}")
    print(f"    Variance explained  : {np.cumsum(pca.explained_variance_ratio_)[-1]*100:.2f}%")

    # 3. Joint matrix
    print(f"\n[3] Building joint matrix  "
          f"(alpha={cfg['alpha_default']}, beta={cfg['beta_default']}) ...")
    Z, scaler_X, _ = build_joint_matrix(X, Y_r, cfg["alpha_default"], cfg["beta_default"])
    X_scaled = scaler_X.transform(X)

    # 4. Growth loop
    rp_hdr = "  ".join(f"bias_rp{rp:>4d}" for rp in report_rp)
    print(f"\n[4] Growth loop ...")
    print(f"    {'k':>4s} | {'cov':>5s} | {'disc':>6s} | {'rmse':>6s} | {rp_hdr}")

    history: list = []
    indices:  Optional[np.ndarray] = None
    k = cfg["k_initial"]

    while True:
        indices  = select_kmedoids(Z, k, cfg["random_seed"], forced_indices=forced)
        sf       = evaluate_sf_metrics(Z, X_scaled, Y_r, indices,
                                       cfg["n_coverage_clusters"], cfg["random_seed"])
        dsw_method = cfg.get("dsw_method", 1)
        Y_sel = Y[indices, :]
        DSW_g = compute_global_dsw(Y_sel, HC_bench, tbl_aer,
                                   dry_thr=dry_thr, method=dsw_method)
        HC_rec = reconstruct_hc_global_dsw(Y_sel, DSW_g, tbl_aer, dry_thr=dry_thr)
        hc_m = evaluate_hc_metrics(Y_sel, HC_bench, tbl_aer,
                                   dry_thr=dry_thr, dsw_method=dsw_method)
        for rp in report_rp:
            col = int(np.argmin(np.abs(tbl_aer - 1.0 / rp)))
            hc_m[f"bias_rp{rp}"] = float(np.nanmean(
                HC_rec[:, col] - HC_bench[:, col]))

        history.append({"k": k, **sf, **hc_m})

        rp_vals = "  ".join(f"{hc_m[f'bias_rp{rp}']:>+10.4f}" for rp in report_rp)
        print(f"    {k:>4d} | {sf['coverage']:>5.3f} | {sf['discrepancy']:>6.4f} | "
              f"{hc_m['mean_rmse']:>6.4f} | {rp_vals}")

        rmse_ok = (cfg.get("rmse_threshold") is None
                   or hc_m["mean_rmse"] <= cfg["rmse_threshold"])
        if (sf["coverage"]    >= cfg["coverage_threshold"] and
                sf["discrepancy"] <= cfg["discrepancy_threshold"] and rmse_ok):
            print(f"\n    All thresholds met at k={k}. Stopping.")
            break
        if k >= cfg["k_max"]:
            print(f"\n    k_max={cfg['k_max']} reached.")
            break
        k = min(k + cfg["k_step"], cfg["k_max"])

    history_df = pd.DataFrame(history)

    # 5. Save
    print("\n[5] Saving outputs ...")
    df_sel = pd.DataFrame(X[indices], columns=x_cols)
    if storm_ids is not None:
        df_sel.insert(0, "storm_id", [storm_ids[i] for i in indices])
    # Map indices back to the original (pre-bbox-filter) ordering when applicable
    bbox_storm_map = cfg.get("bbox_storm_indices")
    if bbox_storm_map is not None:
        orig_indices = np.asarray(bbox_storm_map, dtype=int)[indices]
    else:
        orig_indices = indices
    df_sel.insert(0, "original_index", orig_indices)
    df_sel.to_csv(out_dir / "selected_storms.csv", index=False)
    history_df.to_csv(out_dir / "growth_history.csv", index=False)
    print(f"    selected_storms.csv -> {out_dir}")
    print(f"    growth_history.csv  -> {out_dir}")

    # 6. Plot
    print("\n[6] Generating plots ...")
    plot_growth_evaluation(history_df, cfg, out_dir)

    print("\n=== Growth evaluation complete ===")
    return indices, history_df
