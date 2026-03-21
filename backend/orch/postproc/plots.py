"""
backend/orch/diagnostics/plots.py
===================================
Diagnostic plots for the subset selection pipeline.

Verbatim from tc_subset_selection_v3_hdf5.py Section 9.
Lives in orch/diagnostics/ because these functions consume structured
pipeline outputs (history list, sensitivity DataFrame), not raw arrays.

Public API
----------
  plot_growth_history(history, cfg, out_dir)     -> None
  plot_sensitivity(sens_df, cfg, out_dir)        -> None
  plot_pca_coverage(Y_r_full, Y_r_sub, out_dir)  -> None
"""

from __future__ import annotations

from pathlib import Path

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


def plot_sensitivity(sens_df, cfg, out_dir):
    do_hc = "mean_rmse" in sens_df.columns
    ncols = 3 if do_hc else 2
    nrows = 2 if do_hc else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"Alpha Sensitivity  (k = {cfg['k_sensitivity']}, beta = 1)",
                 fontsize=13, fontweight="bold")
    if nrows == 1:
        axes = axes[np.newaxis, :]

    a = sens_df["w"].values if "w" in sens_df.columns else sens_df["alpha"].values
    BLUE, GREEN, ORANGE, RED, PURPLE = (
        "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0")

    ax1 = axes[0, 0]
    ln1 = ax1.plot(a, sens_df["coverage"], "o-", color=BLUE, lw=2,
                   label="Coverage (Y-space)")
    ax1.axhline(cfg["coverage_threshold"], color=BLUE, ls=":", alpha=0.5)
    ax1.set_xlabel("w"); ax1.set_ylabel("Y-space Coverage", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0, 1.05)
    ax1.set_title("Coverage & Discrepancy vs w")

    ax1r = ax1.twinx()
    ln2  = ax1r.plot(a, sens_df["discrepancy"], "s--", color=GREEN, lw=2,
                     label="Discrepancy (X)")
    ax1r.axhline(cfg["discrepancy_threshold"], color=GREEN, ls=":", alpha=0.5)
    ax1r.set_ylabel("Centered L2 Discrepancy", color=GREEN)
    ax1r.tick_params(axis="y", labelcolor=GREEN)
    ax1.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], fontsize=8)
    ax1.grid(alpha=0.3)

    sf_ok = ((sens_df["coverage"]    >= cfg["coverage_threshold"]) &
             (sens_df["discrepancy"] <= cfg["discrepancy_threshold"]))
    if sf_ok.any():
        for row_ax in axes:
            for col_ax in row_ax:
                col_ax.axvspan(a[sf_ok].min(), a[sf_ok].max(),
                               alpha=0.08, color="green")

    ax = axes[0, 1]
    ax.plot(a, sens_df["maximin"], "^-", color=ORANGE, lw=2)
    ax.set(xlabel="w", ylabel="Maximin Distance",
           title="Maximin Distance vs w"); ax.grid(alpha=0.3)

    if not do_hc and ncols == 3:
        axes[0, 2].set_visible(False)

    if do_hc:
        ax = axes[1, 0]
        ax.plot(a, sens_df["mean_rmse"], "^-", color=PURPLE, lw=2)
        thr = cfg.get("rmse_threshold")
        if thr is not None:
            ax.axhline(thr, color=RED, ls="--", label=f"Threshold = {thr}")
            ax.legend(fontsize=8)
        ax.set(xlabel="w", ylabel="Mean RMSE  (m)",
               title="HC Mean RMSE vs w"); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(a, sens_df["mean_bias"], "o-", color=PURPLE, lw=2)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="w", ylabel="Mean Bias  (m)",
               title="HC Mean Bias vs w"); ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(a, sens_df["mean_uncertainty"], "s-", color=PURPLE, lw=2)
        ax.set(xlabel="w", ylabel="Mean Uncertainty  (m)",
               title="HC Mean Uncertainty vs w"); ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = out_dir / "sensitivity_w.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


def plot_pca_coverage(Y_r_full, Y_r_sub, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y_r_full[:, 0], Y_r_full[:, 1],
               c="lightgrey", s=15, label="Full dataset", zorder=1)
    ax.scatter(Y_r_sub[:, 0], Y_r_sub[:, 1],
               c="#E91E63", s=45, edgecolors="k", lw=0.4,
               label=f"Selected  (k = {len(Y_r_sub)})", zorder=2)
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title="Y-space PCA — Full Dataset vs Selected Subset")
    ax.legend(); ax.grid(alpha=0.3)
    fpath = out_dir / "pca_coverage.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")
