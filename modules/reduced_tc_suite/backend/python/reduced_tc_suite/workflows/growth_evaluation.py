"""RTCS Selection (optimal k) — RMSE-driven growth sweep.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Sweeps subset size from k_min to k_max in steps of k_step.  At each k:

  1.  PAM selection on the joint matrix Z (parameter + response space).
  2.  Discrete Storm Weights back-computed at every node from the
      benchmark HC.
  3.  Global DSWs aggregated across active nodes.
  4.  Hazard curves reconstructed at every node via JPM integration.
  5.  Per-node RMSE against the benchmark, and the mean across nodes
      (global RMSE).

After the full sweep, the smallest k whose global RMSE is at or below the
user-supplied tolerance is selected.  If no k meets the tolerance, the k
with the lowest observed RMSE is selected and a warning is emitted.

The selected RTCS is then carried through the same diagnostic suite as
the fixed-k workflow (PCA / SPLOM plots, HC verification, QBM
post-correction).

Public API
----------
  run_growth_evaluation(cfg)  ->  (selected_indices, history_df)
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from reduced_tc_suite.sampling.pca import reduce_output
from reduced_tc_suite.sampling.joint_matrix import build_joint_matrix
from reduced_tc_suite.sampling.kmedoids import select_subset
from reduced_tc_suite.sampling.metrics import evaluate_sf_metrics
from reduced_tc_suite.weights.dsw import (
    compute_global_dsw,
    reconstruct_hc_global_dsw,
    evaluate_hc_reconstruction,
)
from reduced_tc_suite.weights.qbm import compute_qbm_bias
from reduced_tc_suite.workflows.rtcs_selection import (
    _load_pipeline_data,
    _load_forced_indices,
    _remap_forced_indices,
    _splom,
)
from reduced_tc_suite.postproc.plots import (
    plot_rmse_vs_k,
    plot_pca_yspace,
    plot_hc_comparison,
    plot_hc_qbm,
)
from reduced_tc_suite.config.defaults import RTCS_SELECTION_DEFAULTS


def _select_optimal_k(history_df: pd.DataFrame,
                      tolerance: Optional[float]) -> tuple[int, int, bool]:
    """Pick the smallest k whose global RMSE meets the tolerance.

    Falls back to argmin(mean_rmse) if no k meets it.
    Returns (k_selected, history_row_index, met_tolerance).
    """
    rmse   = history_df["mean_rmse"].values
    k_vals = history_df["k"].values

    if tolerance is not None:
        meets = np.where(np.isfinite(rmse) & (rmse <= tolerance))[0]
        if len(meets) > 0:
            i = meets[np.argmin(k_vals[meets])]
            return int(k_vals[i]), int(i), True

    finite = np.where(np.isfinite(rmse))[0]
    if len(finite) == 0:
        raise RuntimeError("All sweep iterations produced non-finite RMSE.")
    i = int(finite[np.argmin(rmse[finite])])
    return int(k_vals[i]), i, False


def run_growth_evaluation(cfg: Optional[dict] = None):
    """RTCS Selection (optimal k) — full RMSE sweep + tolerance-driven pick.

    Returns
    -------
    selected_indices : ndarray  [k_selected]
    history_df       : DataFrame  one row per k step
    """
    base = copy.deepcopy(RTCS_SELECTION_DEFAULTS)
    if cfg:
        base.update(cfg)
    cfg = base

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["random_seed"])

    # Accept legacy k_initial as an alias for k_min.
    k_min     = cfg.get("k_min", cfg.get("k_initial", 20))
    k_max     = cfg["k_max"]
    k_step    = cfg["k_step"]
    tolerance = cfg.get("rmse_threshold")

    print(f"\n{'='*60}")
    print(f"  RTCS Selection (optimal k) - RMSE-Driven Growth Sweep")
    print(f"  k sweep         : {k_min} -> {k_max}  (step {k_step})")
    if tolerance is not None:
        print(f"  RMSE tolerance  : {tolerance:g} m")
    else:
        print(f"  RMSE tolerance  : not set (will report argmin RMSE)")
    print(f"{'='*60}")

    print("\n[1] Loading data ...")
    X_full, Y, HC_bench, storm_ids, x_cols_full = _load_pipeline_data(cfg)
    forced = _remap_forced_indices(cfg, _load_forced_indices(cfg))

    sel_cols = cfg.get("x_select_columns")
    if sel_cols is not None:
        X      = X_full[:, sel_cols]
        x_cols = [x_cols_full[i] for i in sel_cols]
        print(f"    X columns   : {sel_cols} -> {x_cols}")
    else:
        X      = X_full
        x_cols = x_cols_full

    if HC_bench is None:
        raise ValueError(
            "HC_bench is required for RTCS Selection (optimal k). "
            "Set HC_source in the preprocessor config and re-run preprocess.py.")

    n_forced = 0 if forced is None else len(forced)
    if n_forced > 0:
        print(f"    Forced storms : {n_forced}")
        if k_min < n_forced:
            print(f"    k_min ({k_min}) < |forced| ({n_forced}). "
                  f"Raising k_min to {n_forced}.")
            k_min = n_forced
        if k_max < n_forced:
            raise ValueError(
                f"k_max ({k_max}) is smaller than the number of forced storms "
                f"({n_forced}).")

    tbl_aer        = cfg["TBL_AER"]
    dry_thr        = cfg["dry_threshold"]
    report_aer     = cfg["bias_report_aer"]
    min_wet_storms = cfg.get("min_wet_storms", 2)
    dsw_method     = cfg.get("dsw_method", 1)

    print(f"\n[2] PCA on Y  (retaining {cfg['pca_variance_threshold']*100:.0f}% "
          f"variance) ...")
    Y_r, pca = reduce_output(
        Y, cfg["pca_variance_threshold"],
        dry_strategy=cfg.get("pca_dry_strategy", "drop_always_dry"),
        min_wet_fraction=cfg.get("pca_min_wet_fraction", 0.2),
    )
    print(f"    Components retained : {Y_r.shape[1]}")
    print(f"    Variance explained  : "
          f"{np.cumsum(pca.explained_variance_ratio_)[-1]*100:.2f}%")

    print(f"\n[3] Building joint matrix  "
          f"(alpha={cfg['alpha_default']}, beta={cfg['beta_default']}) ...")
    Z, scaler_X, _ = build_joint_matrix(
        X, Y_r, cfg["alpha_default"], cfg["beta_default"])
    X_scaled = scaler_X.transform(X)

    aer_hdr = "  ".join(f"bias_aer{n:>4d}" for n in report_aer)
    print(f"\n[4] Growth sweep  (full range, no early stop) ...")
    print(f"    {'k':>4s} | {'cov':>5s} | {'disc':>6s} | "
          f"{'rmse':>6s} | {aer_hdr}")

    history: list = []
    sel_method = cfg.get("selection_method", "kmedoids")
    k = k_min
    while True:
        indices = select_subset(Z, k, cfg["random_seed"],
                                forced_indices=forced, method=sel_method)
        sf = evaluate_sf_metrics(Z, X_scaled, Y_r, indices,
                                 cfg["n_coverage_clusters"],
                                 cfg["random_seed"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, hc_m = evaluate_hc_reconstruction(
                Y[indices, :], HC_bench, tbl_aer, dry_thr, report_aer,
                dsw_method=dsw_method, min_wet_storms=min_wet_storms)

        history.append({"k": k, **sf, **hc_m})

        aer_vals = "  ".join(
            f"{hc_m[f'bias_aer{n}']:>+10.4f}" for n in report_aer)
        print(f"    {k:>4d} | {sf['coverage']:>5.3f} | "
              f"{sf['discrepancy']:>6.4f} | {hc_m['mean_rmse']:>6.4f} | "
              f"{aer_vals}")

        if k >= k_max:
            break
        k = min(k + k_step, k_max)

    history_df = pd.DataFrame(history)

    print(f"\n[5] Selecting optimal k ...")
    k_selected, sel_idx, met = _select_optimal_k(history_df, tolerance)
    rmse_selected = float(history_df.loc[sel_idx, "mean_rmse"])
    if met:
        print(f"    Smallest k meeting tolerance "
              f"({tolerance:g} m): k = {k_selected}  "
              f"(global RMSE = {rmse_selected:.4f} m)")
    else:
        msg = (f"No k in [{k_min}, {k_max}] met the RMSE tolerance "
               f"({tolerance!r} m).  Falling back to argmin RMSE: "
               f"k = {k_selected}  (global RMSE = {rmse_selected:.4f} m)")
        warnings.warn(msg, RuntimeWarning)
        print(f"    {msg}")

    # Re-select at k_selected so we can run the diagnostic suite below.
    indices = select_subset(
        Z, k_selected, cfg["random_seed"], forced_indices=forced,
        method=sel_method)

    print(f"\n[6] Saving outputs ...")
    df_sel = pd.DataFrame(X_full[indices], columns=x_cols_full)
    if storm_ids is not None:
        df_sel.insert(0, "storm_id", [storm_ids[i] for i in indices])
    bbox_storm_map = cfg.get("bbox_storm_indices")
    if bbox_storm_map is not None:
        orig_indices = np.asarray(bbox_storm_map, dtype=int)[indices]
    else:
        orig_indices = indices
    df_sel.insert(0, "original_index", orig_indices)
    df_sel.to_csv(out_dir / "selected_storms.csv", index=False)
    history_df.to_csv(out_dir / "growth_history.csv", index=False)
    print(f"    selected_storms.csv -> {out_dir}  (k = {k_selected})")
    print(f"    growth_history.csv  -> {out_dir}")

    print(f"\n[7] Per-node RMSE at k = {k_selected} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        DSW_global = compute_global_dsw(
            Y[indices, :], HC_bench, tbl_aer, dry_thr,
            min_wet_storms, method=dsw_method)
        HC_recon = reconstruct_hc_global_dsw(
            Y[indices, :], DSW_global, tbl_aer, dry_thr)
        resid = HC_recon - HC_bench
        node_rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        node_bias = np.nanmean(resid, axis=1)
    pd.DataFrame({"node_index": np.arange(len(node_rmse)),
                  "rmse":       node_rmse,
                  "bias":       node_bias}).to_csv(
        out_dir / "node_rmse.csv", index=False)
    n_valid = int(np.isfinite(node_rmse).sum())
    print(f"    node_rmse.csv -> {out_dir}  "
          f"({n_valid} valid nodes, mean = {np.nanmean(node_rmse):.4f} m, "
          f"max = {np.nanmax(node_rmse):.4f} m)")

    print(f"\n[8] Plot: global RMSE vs k ...")
    plot_rmse_vs_k(history_df, tolerance, k_selected, rmse_selected,
                   met_tolerance=met, out_dir=out_dir)

    print(f"\n[9] Diagnostic plots at k = {k_selected} ...")
    new_indices = (np.setdiff1d(indices, forced)
                   if forced is not None else indices)
    plot_pca_yspace(
        Y_r, forced, new_indices, out_dir, "pca_yspace_final.png",
        title=f"Y-space PCA - Selected RTCS (k = {k_selected})")
    _splom(X_full, x_cols_full, forced, new_indices, cfg, out_dir,
           "tc_splom_final.png")

    print(f"\n[10] HC verification plot (9 sampled nodes) ...")
    plot_hc_comparison(
        Y[indices, :], DSW_global, HC_bench, tbl_aer,
        out_dir, dry_thr=dry_thr,
        n_nodes=9, seed=cfg["random_seed"])

    print(f"\n[11] Quantile Bias Mapping (QBM) post-correction ...")
    qbm_aer_mode = cfg.get("qbm_aer_mode", "631")
    qbm_mode     = cfg.get("qbm_mode", "aer")
    qbm_win      = cfg.get("qbm_win_frac",  0.10)
    qbm_ramp     = cfg.get("qbm_ramp_frac", 0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bias_tbl = compute_qbm_bias(
            Y[indices, :], DSW_global, HC_bench, tbl_aer,
            dry_thr=dry_thr, win_frac=qbm_win, ramp_frac=qbm_ramp,
            aer_mode=qbm_aer_mode, qbm_mode=qbm_mode)

    qbm_path = out_dir / "qbm_bias.h5"
    with h5py.File(str(qbm_path), "w") as hf:
        hf.create_dataset("bias", data=bias_tbl)
        hf.create_dataset("tbl_aer", data=tbl_aer)
        hf.attrs["qbm_mode"] = qbm_mode
    print(f"    QBM bias saved: {qbm_path}")

    plot_hc_qbm(
        Y[indices, :], DSW_global, bias_tbl, HC_bench,
        tbl_aer, out_dir, dry_thr=dry_thr,
        n_nodes=9, seed=cfg["random_seed"],
        aer_mode=qbm_aer_mode, qbm_mode=qbm_mode,
        win_frac=qbm_win, ramp_frac=qbm_ramp)

    print(f"\n=== Growth evaluation complete  "
          f"(k_selected = {k_selected}) ===")
    return indices, history_df
