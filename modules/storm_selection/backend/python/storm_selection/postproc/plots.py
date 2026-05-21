"""Diagnostic plots for the storm_selection pipeline.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Functions consume structured pipeline outputs (DataFrames, indices), not raw
engine arrays — this module is part of the orchestration role's post-processing
responsibility per CyHAN v1.1 §4.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_growth_history(history, cfg, out_dir):
    df    = pd.DataFrame(history)
    do_hc = "mean_rmse" in df.columns
    k_v   = df["k"].values
    nrows = 2 if do_hc else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5 * nrows))
    fig.suptitle("Growth Loop — All Metrics vs Subset Size k",
                 fontsize=13, fontweight="bold")
    if nrows == 1:
        axes = axes[np.newaxis, :]

    BLUE, GREEN, ORANGE, RED, PURPLE = (
        "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0")

    ax = axes[0, 0]
    ax.plot(k_v, df["coverage"], "o-", color=BLUE, lw=2)
    ax.axhline(cfg["coverage_threshold"], color=RED, ls="--",
               label=f"Threshold = {cfg['coverage_threshold']}")
    ax.set(xlabel="k", ylabel="Y-space Coverage", title="Coverage of Y-space")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(k_v, df["discrepancy"], "s-", color=GREEN, lw=2)
    ax.axhline(cfg["discrepancy_threshold"], color=RED, ls="--",
               label=f"Threshold = {cfg['discrepancy_threshold']}")
    ax.set(xlabel="k", ylabel="Centered L2 Discrepancy",
           title="Input-Space Discrepancy\n(lower = more uniform)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(k_v, df["maximin"], "^-", color=ORANGE, lw=2)
    ax.set(xlabel="k", ylabel="Maximin Distance",
           title="Maximin Distance in Z-space")
    ax.grid(alpha=0.3)

    if do_hc:
        ax = axes[1, 0]
        ax.plot(k_v, df["mean_bias"], "o-", color=PURPLE, lw=2)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="k", ylabel="Mean Bias  (m)", title="HC Mean Bias vs Benchmark")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(k_v, df["mean_uncertainty"], "s-", color=PURPLE, lw=2)
        ax.set(xlabel="k", ylabel="Mean Uncertainty  (m)",
               title="HC Mean Uncertainty vs Benchmark")
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(k_v, df["mean_rmse"], "^-", color=PURPLE, lw=2)
        thr = cfg.get("rmse_threshold")
        if thr is not None:
            ax.axhline(thr, color=RED, ls="--", label=f"Threshold = {thr}")
            ax.legend(fontsize=8)
        ax.set(xlabel="k", ylabel="Mean RMSE  (m)", title="HC Mean RMSE vs Benchmark")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = out_dir / "growth_history.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


def plot_rmse_vs_k(history_df, tolerance, k_selected, rmse_selected,
                   met_tolerance=True, out_dir=None):
    """Plot global HC RMSE vs subset size k.

    Overlays the user-supplied tolerance (horizontal line) and the
    selected k (vertical line + marker).  This is the primary diagnostic
    for the optimal-k workflow.
    """
    df     = history_df
    k_v    = df["k"].values
    rmse_v = df["mean_rmse"].values

    BLUE, RED, GREEN, AMBER = "#2196F3", "#F44336", "#4CAF50", "#FF9800"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(k_v, rmse_v, "o-", color=BLUE, lw=2, ms=6,
            label="Global mean RMSE")

    if tolerance is not None:
        ax.axhline(tolerance, color=RED, ls="--", lw=1.5,
                   label=f"Tolerance = {tolerance:g} m")

    pick_color = GREEN if met_tolerance else AMBER
    if met_tolerance:
        sel_label = (f"Selected k = {k_selected}  "
                     f"(RMSE = {rmse_selected:.3f} m)")
    else:
        sel_label = (f"Selected k = {k_selected}  "
                     f"(argmin RMSE = {rmse_selected:.3f} m, "
                     f"tolerance not met)")
    ax.axvline(k_selected, color=pick_color, ls=":", lw=1.5, label=sel_label)
    ax.plot([k_selected], [rmse_selected], "D", color=pick_color, ms=11,
            markeredgecolor="black", zorder=5)
    ax.annotate(
        f"k = {k_selected}\nRMSE = {rmse_selected:.3f} m",
        xy=(k_selected, rmse_selected),
        xytext=(10, 12), textcoords="offset points",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=pick_color,
                  alpha=0.9))

    ax.set_xlabel("Number of selected storms, k (RTCS size)", fontsize=11)
    ax.set_ylabel("Global mean hazard-curve RMSE  (m)", fontsize=11)
    ax.set_title(
        "RTCS Selection (optimal k): Global HC RMSE vs Subset Size",
        fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fpath = Path(out_dir) / "rmse_vs_k.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {fpath}")


def plot_pca_yspace(
    Y_r_full:       np.ndarray,
    forced_indices: Optional[np.ndarray],
    new_indices:    Optional[np.ndarray],
    out_dir:        Path,
    filename:       str,
    title:          Optional[str] = None,
    sub_indices:    Optional[np.ndarray] = None,
    sub_label:      str = "Sub-RTCS",
) -> None:
    """Y-space PCA scatter (PC 1 vs PC 2)."""
    n_total = len(Y_r_full)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(Y_r_full[:, 0], Y_r_full[:, 1],
               s=30, facecolors="none", edgecolors="gray", linewidths=0.8,
               zorder=1)

    if forced_indices is not None and len(forced_indices) > 0:
        Y_forced = Y_r_full[forced_indices]
        ax.scatter(Y_forced[:, 0], Y_forced[:, 1],
                   s=30, facecolors="none", edgecolors="#1565C0", linewidths=0.8,
                   zorder=2)

    if new_indices is not None and len(new_indices) > 0:
        Y_new = Y_r_full[new_indices]
        ax.scatter(Y_new[:, 0], Y_new[:, 1],
                   s=30, facecolors="none", edgecolors="#C62828", linewidths=0.8,
                   zorder=3)

    if sub_indices is not None and len(sub_indices) > 0:
        Y_sub = Y_r_full[sub_indices]
        ax.scatter(Y_sub[:, 0], Y_sub[:, 1],
                   s=40, facecolors="#2E7D32", edgecolors="#2E7D32",
                   linewidths=0.8, zorder=4)

    ax.set(xlabel="PC 1", ylabel="PC 2")
    ax.set_title(title or "Y-space PCA — PC 1 vs PC 2",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    handles = [
        plt.scatter([], [], s=60, facecolors="none",
                    edgecolors="gray", linewidths=0.8,
                    label=f"All storms  (n={n_total})"),
    ]
    if forced_indices is not None and len(forced_indices) > 0:
        handles.append(
            plt.scatter([], [], s=60, facecolors="none",
                        edgecolors="#1565C0", linewidths=0.8,
                        label=f"Pre-selected  (n={len(forced_indices)})"))
    if new_indices is not None and len(new_indices) > 0:
        handles.append(
            plt.scatter([], [], s=60, facecolors="none",
                        edgecolors="#C62828", linewidths=0.8,
                        label=f"Newly selected  (n={len(new_indices)})"))
    if sub_indices is not None and len(sub_indices) > 0:
        handles.append(
            plt.scatter([], [], s=60, facecolors="#2E7D32",
                        edgecolors="#2E7D32", linewidths=0.8,
                        label=f"{sub_label}  (n={len(sub_indices)})"))
    ax.legend(handles=handles, fontsize=9)

    fpath = out_dir / filename
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


# ---------------------------------------------------------------------------
# TC parameter SPLOM
# ---------------------------------------------------------------------------

def _bubble_scatter(ax, x, y, color, base_area=30, filled=False):
    """Scatter x/y as bubbles sized by count at each unique location."""
    if len(x) == 0:
        return
    coords = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    uniq, counts = np.unique(coords, axis=0, return_counts=True)
    sizes = base_area * np.sqrt(counts)
    ax.scatter(uniq[:, 0], uniq[:, 1], s=sizes,
               facecolors=color if filled else "none",
               edgecolors=color, linewidths=0.8, zorder=2)


def plot_tc_splom(
    X:              np.ndarray,
    x_cols:         list,
    forced_indices: Optional[np.ndarray],
    new_indices:    Optional[np.ndarray],
    param_spec:     Optional[list],
    param_labels:   Optional[list],
    title:          str,
    out_dir:        Path,
    filename:       str,
    sub_indices:    Optional[np.ndarray] = None,
    sub_label:      str = "Sub-RTCS",
) -> None:
    """Scatter-plot matrix (SPLOM / pairs plot) for TC parameters."""
    if param_spec is None:
        col_idx = list(range(min(4, len(x_cols))))
    else:
        col_idx = [
            x_cols.index(p) if isinstance(p, str) else int(p)
            for p in param_spec
        ]
    labels = param_labels if param_labels else [x_cols[c] for c in col_idx]
    n      = len(col_idx)

    D_all    = X[:, col_idx]
    D_forced = (X[forced_indices][:, col_idx]
                if forced_indices is not None and len(forced_indices) > 0
                else None)
    D_new    = (X[new_indices][:, col_idx]
                if new_indices is not None and len(new_indices) > 0
                else None)
    D_sub    = (X[sub_indices][:, col_idx]
                if sub_indices is not None and len(sub_indices) > 0
                else None)

    ranges = []
    for i in range(n):
        lo, hi = D_all[:, i].min(), D_all[:, i].max()
        pad = (hi - lo) * 0.05 or 1.0
        ranges.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if row == col:
                ax.hist(D_all[:, row], bins=20, color="gray",
                        edgecolor="white", linewidth=0.5)
                ax.set_xlim(ranges[col])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.set_tick_params(length=0)

            else:
                _bubble_scatter(ax, D_all[:, col],    D_all[:, row],    "gray")
                if D_forced is not None:
                    _bubble_scatter(ax, D_forced[:, col], D_forced[:, row], "#1565C0")
                if D_new is not None:
                    _bubble_scatter(ax, D_new[:, col],    D_new[:, row],    "#C62828")
                if D_sub is not None:
                    _bubble_scatter(ax, D_sub[:, col], D_sub[:, row],
                                    "#2E7D32", filled=True)
                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])

            if row == 0:
                ax.xaxis.set_label_position("top")
                ax.xaxis.tick_top()
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.xaxis.set_tick_params(length=0)

            if col == n - 1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.set_tick_params(length=0)

    for i, label in enumerate(labels):
        axes[i, i].text(0.5, 0.5, label,
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        transform=axes[i, i].transAxes)

    legend_handles = [
        plt.scatter([], [], s=60, facecolors="none",
                    edgecolors="gray", linewidths=0.8,
                    label=f"All storms  (n={len(X)})"),
    ]
    if D_forced is not None:
        legend_handles.append(
            plt.scatter([], [], s=60, facecolors="none",
                        edgecolors="#1565C0", linewidths=0.8,
                        label=f"Pre-selected  (n={len(forced_indices)})"))
    if D_new is not None:
        legend_handles.append(
            plt.scatter([], [], s=60, facecolors="none",
                        edgecolors="#C62828", linewidths=0.8,
                        label=f"Newly selected  (n={len(new_indices)})"))
    if D_sub is not None:
        legend_handles.append(
            plt.scatter([], [], s=60, facecolors="#2E7D32",
                        edgecolors="#2E7D32", linewidths=0.8,
                        label=f"{sub_label}  (n={len(sub_indices)})"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fpath = out_dir / filename
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


# ---------------------------------------------------------------------------
# HC comparison: benchmark vs RTCS reconstruction
# ---------------------------------------------------------------------------

def _nodal_dsw_at_node(resp, hc_node, tbl_aer, dry_thr):
    """Back-compute nodal DSWs for a single node via DRM."""
    from storm_selection.weights.dsw import _surge_to_aer

    valid = (~np.isnan(resp)) & (resp > dry_thr)
    if valid.sum() < 2:
        return None, None
    desc      = np.argsort(resp[valid])[::-1]
    surge     = resp[valid][desc]
    aer_q     = _surge_to_aer(hc_node, tbl_aer, surge)
    if np.all(np.isnan(aer_q)):
        return None, None
    dsw       = np.empty_like(aer_q)
    dsw[0]    = np.where(np.isnan(aer_q[0]), 0.0, aer_q[0])
    dsw[1:]   = np.where(
        np.isnan(aer_q[1:]) | np.isnan(aer_q[:-1]), 0.0, np.diff(aer_q))
    dsw       = np.clip(dsw, 0.0, None)
    return surge, dsw


def _plot_hc_grid(
    Y_sub, DSW_global, HC_bench, tbl_aer, nodes, out_dir,
    dry_thr, filename, title, show_nodal,
):
    """Shared 3x3 HC comparison grid (internal)."""
    CLR_BENCH = "gray"
    CLR_NODAL = "#1565C0"
    CLR_GLOBAL = "#C62828"

    n_nodes = len(nodes)
    ncols = 3
    nrows = (n_nodes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax_idx, node in enumerate(nodes):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]

        ax.plot(tbl_aer, HC_bench[node, :], "-", color=CLR_BENCH,
                lw=1.5, label="Benchmark")

        if show_nodal:
            surge_n, dsw_n = _nodal_dsw_at_node(
                Y_sub[:, node], HC_bench[node, :], tbl_aer, dry_thr)
            if surge_n is not None:
                cum_aer_n = np.cumsum(dsw_n)
                ax.plot(cum_aer_n, surge_n, "o", color=CLR_NODAL,
                        ms=3, label="RTCS (Nodal DSW)")

        resp = Y_sub[:, node]
        valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
        if valid.sum() >= 2:
            desc      = np.argsort(resp[valid])[::-1]
            surge_g   = resp[valid][desc]
            cum_aer_g = np.cumsum(DSW_global[valid][desc])
            ax.plot(cum_aer_g, surge_g, "o", color=CLR_GLOBAL,
                    ms=3, label="RTCS (Global DSW)")

        ax.set_xscale("log")
        ax.set_xlim(1e1, 1e-6)
        ax.set_xlabel("AER (year$^{-1}$)")
        ax.set_ylabel("TC Response")
        ax.set_title(f"Node {node}", fontsize=10)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    for ax_idx in range(n_nodes, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fpath = Path(out_dir) / filename
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


def plot_hc_comparison(
    Y_sub:       np.ndarray,
    DSW_global:  np.ndarray,
    HC_bench:    np.ndarray,
    tbl_aer:     np.ndarray,
    out_dir,
    dry_thr:     float = 0.0,
    n_nodes:     int = 9,
    seed:        int = 42,
):
    """Generate two HC comparison plots (3x3 grid)."""
    out_dir = Path(out_dir)
    m = HC_bench.shape[0]
    n_nodes = min(n_nodes, m)

    rng   = np.random.default_rng(seed)
    nodes = sorted(rng.choice(m, size=n_nodes, replace=False).tolist())

    _plot_hc_grid(Y_sub, DSW_global, HC_bench, tbl_aer, nodes, out_dir,
                  dry_thr, "hc_comparison.png",
                  "HC Comparison — Benchmark vs RTCS (Global DSW)",
                  show_nodal=False)

    _plot_hc_grid(Y_sub, DSW_global, HC_bench, tbl_aer, nodes, out_dir,
                  dry_thr, "hc_comparison_nodal.png",
                  "HC Comparison — Benchmark vs RTCS (Nodal + Global DSW)",
                  show_nodal=True)


def plot_hc_qbm(
    Y_sub:        np.ndarray,
    DSW_global:   np.ndarray,
    bias_array:   np.ndarray,
    HC_bench:     np.ndarray,
    tbl_aer:      np.ndarray,
    out_dir,
    dry_thr:      float = 0.0,
    n_nodes:      int = 9,
    seed:         int = 42,
    aer_mode:     str = "631",
    qbm_mode:     str = "aer",
    win_frac:     float = 0.10,
    ramp_frac:    float = 0.03,
):
    """Plot HC comparison: Benchmark vs DSW-only vs QBM-corrected (3x3 grid)."""
    from storm_selection.weights.qbm import correct_node_qbm

    CLR_BENCH = "gray"
    CLR_DSW   = "#C62828"
    CLR_QBM   = "#2E7D32"

    if qbm_mode == "aer":
        qbm_label = "RTCS (QBM-AER corrected)"
        title_suffix = "QBM-AER"
    else:
        qbm_label = "RTCS (QBM-response corrected)"
        title_suffix = "QBM-Response"

    out_dir = Path(out_dir)
    m = HC_bench.shape[0]
    n_nodes = min(n_nodes, m)

    rng   = np.random.default_rng(seed)
    nodes = sorted(rng.choice(m, size=n_nodes, replace=False).tolist())

    ncols = 3
    nrows = (n_nodes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"HC Comparison -- Benchmark vs DSW vs {title_suffix}",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax_idx, node in enumerate(nodes):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]

        ax.plot(tbl_aer, HC_bench[node, :], "-", color=CLR_BENCH,
                lw=1.5, label="Benchmark")

        resp = Y_sub[:, node]
        valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
        if valid.sum() >= 2:
            desc      = np.argsort(resp[valid])[::-1]
            surge_g   = resp[valid][desc]
            cum_aer_g = np.cumsum(DSW_global[valid][desc])
            ax.plot(cum_aer_g, surge_g, "o", color=CLR_DSW,
                    ms=3, label="RTCS (Global DSW)")

            aer_out, surge_out = correct_node_qbm(
                resp, DSW_global, HC_bench[node, :],
                bias_array[node, :], tbl_aer,
                dry_thr=dry_thr, aer_mode=aer_mode,
                win_frac=win_frac, ramp_frac=ramp_frac,
                qbm_mode=qbm_mode)

            if aer_out is not None:
                if qbm_mode == "aer":
                    changed = not np.allclose(aer_out, cum_aer_g)
                else:
                    changed = not np.allclose(surge_out, surge_g)
                if changed:
                    ax.plot(aer_out, surge_out, "o",
                            color=CLR_QBM, ms=3, label=qbm_label)

        ax.set_xscale("log")
        ax.set_xlim(1e1, 1e-6)
        ax.set_xlabel("AER (year$^{-1}$)")
        ax.set_ylabel("TC Response")
        ax.set_title(f"Node {node}", fontsize=10)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    for ax_idx in range(n_nodes, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fpath = out_dir / "hc_comparison_qbm.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")
