"""
backend/orch/python/workflows/rtcs_selection.py
=================================================
RTCS Selection (fixed k) — Representative TC Subset selection.

Selects a fixed number of storms that best covers both the TC parameter
space (X) and the hydrodynamic response space (Y) using PCA + k-medoids
on a weighted joint matrix.

Public API
----------
  run_rtcs_selection(cfg)  ->  (indices, metrics)
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.io.store import read_store
from backend.engines.sampling.python.pca import reduce_output
from backend.engines.sampling.python.joint_matrix import build_joint_matrix
from backend.engines.sampling.python.kmedoids import select_kmedoids
from backend.engines.sampling.metrics import evaluate_sf_metrics
from backend.engines.weights.dsw import (
    evaluate_hc_metrics, compute_global_dsw,
)
from backend.engines.weights.qbm import compute_qbm_bias
from backend.orch.python.postproc.plots import (
    plot_pca_yspace, plot_tc_splom, plot_hc_comparison, plot_hc_qbm,
)
from config.defaults import RTCS_SELECTION_DEFAULTS


# ---------------------------------------------------------------------------
# Data loading  (also imported by growth_evaluation)
# ---------------------------------------------------------------------------

def _load_pipeline_data(cfg: dict):
    """
    Load X, Y, HC_bench from HDF5 store or CSV fallback.
    Applies node subsampling when node_stride is set.
    Applies bbox-based storm/node filtering when bbox_storm_indices /
    bbox_node_col_indices are present in cfg.
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

    # ── Bbox storm filter ──────────────────────────────────────────────
    storm_idx = cfg.get("bbox_storm_indices")
    if storm_idx is not None:
        storm_idx = np.asarray(storm_idx, dtype=int)
        # Store mapping: original 0-based index -> new 0-based position
        cfg["_bbox_orig_to_new"] = {int(orig): new for new, orig in enumerate(storm_idx)}
        X = X[storm_idx, :]
        Y = Y[storm_idx, :]
        if storm_ids is not None:
            storm_ids = [storm_ids[i] for i in storm_idx]
        print(f"    Bbox storm filter -> {len(storm_idx)} storms retained")

    # ── Bbox node filter ───────────────────────────────────────────────
    node_idx = cfg.get("bbox_node_col_indices")
    if node_idx is not None:
        node_idx = np.asarray(node_idx, dtype=int)
        Y = Y[:, node_idx]
        if HC_bench is not None:
            HC_bench = HC_bench[node_idx, :]
        print(f"    Bbox node filter -> {len(node_idx)} nodes retained")

    stride = cfg.get("node_stride")
    if stride:
        idx      = np.arange(0, Y.shape[1], stride)
        Y        = Y[:, idx]
        if HC_bench is not None:
            HC_bench = HC_bench[idx, :]
        print(f"    Node stride {stride} -> {len(idx)} nodes retained")

    assert X.shape[0] == Y.shape[0], (
        f"Row mismatch: X has {X.shape[0]} storms, Y has {Y.shape[0]}")
    if HC_bench is not None:
        assert HC_bench.shape[0] == Y.shape[1], (
            f"HC_bench has {HC_bench.shape[0]} rows but Y has {Y.shape[1]} columns")
        assert HC_bench.shape[1] == len(cfg["TBL_AER"]), (
            f"HC_bench has {HC_bench.shape[1]} AER columns but TBL_AER has "
            f"{len(cfg['TBL_AER'])} levels")
        print(f"    HC_bench : {HC_bench.shape}  (nodes x AER levels)")
    print(f"    X        : {X.shape}  (storms x parameters)  {x_cols}")
    print(f"    Y        : {Y.shape}  (storms x nodes)")

    return X, Y, HC_bench, storm_ids, x_cols


# ---------------------------------------------------------------------------
# Pre-selected storm loader  (shared with growth_evaluation)
# ---------------------------------------------------------------------------

def _load_forced_indices(cfg: dict) -> Optional[np.ndarray]:
    """
    Return an array of storm indices that must be included in any selection.

    Checks, in order:
      1. cfg["pre_selected_indices"]  — explicit list/array (0-based)
      2. cfg["pre_selected_csv"]      — CSV file, two supported formats:
           a. Has an 'original_index' column (output of a previous run)
           b. Single column, no header (e.g. MATLAB export) — values are
              treated as 1-based and converted to 0-based automatically

    Returns None when neither is set.
    """
    direct = cfg.get("pre_selected_indices")
    if direct is not None and len(direct) > 0:
        return np.asarray(direct, dtype=int)

    csv_path = cfg.get("pre_selected_csv")
    if csv_path:
        df = pd.read_csv(csv_path)

        if "original_index" in df.columns:
            indices = df["original_index"].values.astype(int)
        elif df.shape[1] == 1:
            # Headerless single-column file (e.g. MATLAB export — 1-based)
            df = pd.read_csv(csv_path, header=None)
            indices = df.iloc[:, 0].values.astype(int) - 1  # 1-based -> 0-based
        else:
            raise ValueError(
                f"pre_selected_csv '{csv_path}': expected either an "
                "'original_index' column or a single-column headerless file.")

        print(f"    Pre-selected : {len(indices)} storms loaded from {csv_path}")
        return indices

    return None


def _remap_forced_indices(cfg: dict, forced: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    When a bbox storm filter is active, remap forced (pre-selected) storm
    indices from the original 0-based ordering to the filtered ordering.
    Indices that fall outside the bbox filter are dropped with a warning.
    """
    if forced is None:
        return None
    mapping = cfg.get("_bbox_orig_to_new")
    if mapping is None:
        return forced

    remapped = []
    dropped = 0
    for orig in forced:
        new = mapping.get(int(orig))
        if new is not None:
            remapped.append(new)
        else:
            dropped += 1

    if dropped > 0:
        print(f"    Pre-selected : {dropped} storms outside bbox filter (dropped)")
    print(f"    Pre-selected : {len(remapped)} storms after bbox remapping")
    return np.array(remapped, dtype=int) if remapped else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _splom(X, x_cols, forced, new_indices, cfg, out_dir, filename):
    """Thin wrapper that unpacks SPLOM config keys and calls plot_tc_splom."""
    plot_tc_splom(
        X=X, x_cols=x_cols,
        forced_indices=forced,
        new_indices=new_indices,
        param_spec=cfg.get("splom_params"),
        param_labels=cfg.get("splom_labels"),
        title=cfg.get("splom_title", "JPM-OS-Q"),
        out_dir=out_dir,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rtcs_selection(cfg: Optional[dict] = None):
    """
    RTCS Selection (fixed k) — Select a fixed-size Representative TC Subset.

    Returns
    -------
    indices : ndarray [k_total]
    metrics : dict  (k, coverage, discrepancy, maximin)
    """
    base = copy.deepcopy(RTCS_SELECTION_DEFAULTS)
    if cfg:
        base.update(cfg)
    cfg = base

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["random_seed"])

    # 1. Load
    print("\n[1] Loading data ...")

    # Load pre-selected indices first (original 0-based space) so we can
    # guarantee they survive any bbox/radius filter.
    forced_orig = _load_forced_indices(cfg)
    bbox_storm_idx = cfg.get("bbox_storm_indices")
    if bbox_storm_idx is not None and forced_orig is not None:
        # Pre-selected storms are always included — the radius filter
        # only restricts the candidate pool for new selections.
        bbox_arr = np.asarray(bbox_storm_idx, dtype=int)
        n_before = len(bbox_arr)
        cfg["bbox_storm_indices"] = np.union1d(bbox_arr, forced_orig)
        n_added = len(cfg["bbox_storm_indices"]) - n_before
        if n_added > 0:
            print(f"    Pre-selected : {n_added} storms outside radius "
                  f"added back (always included)")

    X_full, Y, HC_bench, storm_ids, x_cols_full = _load_pipeline_data(cfg)
    forced = _remap_forced_indices(cfg, forced_orig)

    # Column filter: keep only physically meaningful TC parameters
    sel_cols = cfg.get("x_select_columns")
    if sel_cols is not None:
        X      = X_full[:, sel_cols]
        x_cols = [x_cols_full[i] for i in sel_cols]
        print(f"    X columns   : {sel_cols} -> {x_cols}")
    else:
        X      = X_full
        x_cols = x_cols_full

    n_additional = cfg["k_additional"]
    k = (len(forced) if forced is not None else 0) + n_additional

    print(f"\n{'='*60}")
    if forced is not None:
        print(f"  RTCS Selection (fixed k)")
        print(f"  Pre-selected : {len(forced)}  +  additional : {n_additional}"
              f"  =  total k : {k}")
    else:
        print(f"  RTCS Selection (fixed k = {k})")
    print(f"{'='*60}")

    # 1b. Initial SPLOM  (all storms + pre-selected; no newly selected yet)
    _splom(X_full, x_cols_full, forced, None, cfg, out_dir, "tc_splom_initial.png")


    # 2. PCA
    print(f"\n[2] PCA on Y  (retaining {cfg['pca_variance_threshold']*100:.0f}% variance) ...")
    Y_r, pca = reduce_output(Y, cfg["pca_variance_threshold"])
    print(f"    Components retained : {Y_r.shape[1]}")
    print(f"    Variance explained  : {np.cumsum(pca.explained_variance_ratio_)[-1]*100:.2f}%")

    # 2b. Initial Y-space PCA plot  (pre-selected only)
    plot_pca_yspace(Y_r, forced, None, out_dir, "pca_yspace_initial.png",
                    title="Y-space PCA — Initial (Pre-selected)")

    # 2c. Alpha/beta optimization via DSW HC evaluation
    ab_grid = cfg.get("alpha_beta_grid")
    if ab_grid is not None and HC_bench is not None:
        print("\n[3] Optimizing alpha/beta via DSW-HC evaluation ...")
        best_alpha, best_beta = cfg["alpha_default"], cfg["beta_default"]
        best_score = np.inf
        sweep_rows = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for alpha, beta in ab_grid:
                Z_trial, _, _ = build_joint_matrix(X, Y_r, alpha, beta)
                idx_trial = select_kmedoids(Z_trial, k, cfg["random_seed"],
                                             forced_indices=forced)
                hc_m = evaluate_hc_metrics(
                    Y[idx_trial, :], HC_bench, cfg["TBL_AER"],
                    cfg["dry_threshold"], cfg.get("min_wet_storms", 2),
                    dsw_method=cfg.get("dsw_method", 1))
                score = abs(hc_m["mean_bias"]) + hc_m["mean_rmse"]
                sweep_rows.append({"alpha": alpha, "beta": beta, **hc_m, "score": score})
                tag = " ***" if score < best_score else ""
                print(f"    a={alpha:>5.1f}  b={beta:>4.1f} | "
                      f"bias={hc_m['mean_bias']:+.4f} | rmse={hc_m['mean_rmse']:.4f} | "
                      f"score={score:.4f}{tag}")
                if score < best_score:
                    best_score = score
                    best_alpha, best_beta = alpha, beta

        cfg["alpha_default"] = best_alpha
        cfg["beta_default"]  = best_beta
        pd.DataFrame(sweep_rows).to_csv(out_dir / "alpha_beta_sweep.csv", index=False)
        print(f"\n    Optimal: alpha={best_alpha}, beta={best_beta}  "
              f"(score={best_score:.4f})")
        step = 4
    else:
        if ab_grid is not None and HC_bench is None:
            print("\n    [alpha/beta optimization skipped — no HC_bench available]")
        step = 3

    # Joint matrix
    print(f"\n[{step}] Building joint matrix  "
          f"(alpha={cfg['alpha_default']}, beta={cfg['beta_default']}) ...")
    Z, scaler_X, _ = build_joint_matrix(X, Y_r, cfg["alpha_default"], cfg["beta_default"])
    X_scaled = scaler_X.transform(X)

    # Select
    step += 1
    print(f"\n[{step}] Selecting k={k} medoids ...")
    indices = select_kmedoids(Z, k, cfg["random_seed"], forced_indices=forced)

    # Metrics
    step += 1
    metrics = dict(evaluate_sf_metrics(Z, X_scaled, Y_r, indices,
                                       cfg["n_coverage_clusters"], cfg["random_seed"]))
    print(f"    Coverage    = {metrics['coverage']:.4f}  "
          f"(threshold >= {cfg['coverage_threshold']})")
    print(f"    Discrepancy = {metrics['discrepancy']:.4f}  "
          f"(threshold <= {cfg['discrepancy_threshold']})")
    print(f"    Maximin     = {metrics['maximin']:.4f}")

    # Save
    step += 1
    print(f"\n[{step}] Saving outputs ...")
    df_sel = pd.DataFrame(X_full[indices], columns=x_cols_full)
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
    pd.DataFrame([metrics]).to_csv(out_dir / "selection_metrics.csv", index=False)
    print(f"    selected_storms.csv   -> {out_dir}")
    print(f"    selection_metrics.csv -> {out_dir}")

    # Plots
    step += 1
    print(f"\n[{step}] Generating plots ...")
    new_indices = np.setdiff1d(indices, forced) if forced is not None else indices
    plot_pca_yspace(Y_r, forced, new_indices, out_dir, "pca_yspace_final.png",
                    title="Y-space PCA — Final (Pre-selected + New)")
    _splom(X_full, x_cols_full, forced, new_indices, cfg, out_dir, "tc_splom_final.png")

    # HC verification plot (visual-only, does not affect selection)
    if HC_bench is not None:
        step += 1
        dsw_m = cfg.get("dsw_method", 1)
        print(f"\n[{step}] HC verification plot (9 sampled nodes, DSW method {dsw_m}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            DSW_global = compute_global_dsw(
                Y[indices, :], HC_bench, cfg["TBL_AER"],
                cfg["dry_threshold"], cfg.get("min_wet_storms", 2),
                method=dsw_m)
        plot_hc_comparison(Y[indices, :], DSW_global, HC_bench, cfg["TBL_AER"],
                           out_dir, dry_thr=cfg["dry_threshold"],
                           n_nodes=9, seed=cfg["random_seed"])

        # ── QBM bias correction ──────────────────────────────────────
        step += 1
        qbm_m = cfg.get("qbm_mode", "aer")
        qbm_aer = cfg.get("qbm_aer_mode", "631")
        qbm_win = cfg.get("qbm_win_frac", 0.10)
        qbm_ramp = cfg.get("qbm_ramp_frac", 0.03)
        print(f"\n[{step}] Quantile Bias Mapping (QBM) post-correction "
              f"(qbm_mode={qbm_m}, aer_mode={qbm_aer}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bias_tbl = compute_qbm_bias(
                Y[indices, :], DSW_global, HC_bench, cfg["TBL_AER"],
                dry_thr=cfg["dry_threshold"],
                win_frac=qbm_win, ramp_frac=qbm_ramp,
                aer_mode=qbm_aer, qbm_mode=qbm_m)

        # Save QBM bias (standard AER levels per node) to HDF5
        import h5py
        qbm_path = out_dir / "qbm_bias.h5"
        bias_label = "log_aer_delta" if qbm_m == "aer" else "response_bias"
        with h5py.File(str(qbm_path), "w") as hf:
            hf.create_dataset("bias", data=bias_tbl)
            hf.create_dataset("tbl_aer", data=cfg["TBL_AER"])
            hf.attrs["qbm_mode"] = qbm_m
            hf.attrs["bias_type"] = bias_label
        print(f"    QBM bias saved: {qbm_path}  "
              f"({bias_tbl.shape[0]} nodes x {bias_tbl.shape[1]} AER levels, "
              f"type={bias_label})")

        plot_hc_qbm(Y[indices, :], DSW_global, bias_tbl, HC_bench,
                     cfg["TBL_AER"], out_dir, dry_thr=cfg["dry_threshold"],
                     n_nodes=9, seed=cfg["random_seed"],
                     aer_mode=qbm_aer, qbm_mode=qbm_m,
                     win_frac=qbm_win, ramp_frac=qbm_ramp)

    print("\n=== Subset selection complete ===")
    return indices, metrics
