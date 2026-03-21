"""
cli/run_dsw.py
===============
DSW + HC Reconstruction — compute Discrete Storm Weights (DSW) via
back-computation from benchmark hazard curves, and reconstruct hazard
curves at every node via JPM integration for a selected storm subset.

Usage
-----
  1. Run cli/run_rtcs_selection.py first to produce selected_storms.csv
     and the subset indices.
  2. Set CONFIG below.
  3. Run:  python cli/run_dsw.py

Input:
  data/processed/tc_data.h5              — full storm suite (Y and HC)
  data/processed/outputs/selected_storms.csv  — subset selection output

Output:
  data/processed/outputs/dsw_weights.csv      — global DSW per selected storm
  data/processed/outputs/hc_reconstructed.csv — reconstructed HC  [m x N_AER]
  data/processed/outputs/dsw_metrics.csv      — mean_bias, mean_uncertainty, mean_rmse
  data/processed/outputs/hc_comparison.png    — reconstructed vs benchmark overlay

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import os
os.chdir(_PROJECT_ROOT)

from backend.io.store import read_store
from backend.engines.weights.dsw import (
    compute_global_dsw,
    reconstruct_hc_global_dsw,
    evaluate_hc_metrics,
)

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":        "data/processed/tc_data.h5",
    "selected_csv":   "data/processed/outputs/selected_storms.csv",
    "output_dir":     "data/processed/outputs",

    # dry-node threshold (m) — surges at or below this are treated as dry
    "dry_threshold": 0.0,

    # AER levels for HC reconstruction output table.
    # Set to None to use the levels stored in /HC attrs of the HDF5 store.
    "tbl_aer": None,

    # Nodes to include in the HC comparison plot (None = all, or a list of
    # integer indices, e.g. [0, 10, 50, 100]).
    "plot_nodes": None,

    # Number of nodes to plot if plot_nodes is None (randomly sampled)
    "n_plot_nodes": 12,
}
# ---------------------------------------------------------------------------

def _load_inputs(cfg):
    """Load Y_sub, HC_bench, tbl_aer, and storm indices from store + CSV."""
    store = read_store(Path(cfg["h5_path"]))

    sel = pd.read_csv(cfg["selected_csv"])
    if "original_index" not in sel.columns:
        raise ValueError(
            "selected_storms.csv must contain an 'original_index' column.\n"
            "Re-run cli/run_rtcs_selection.py to regenerate it."
        )
    indices = sel["original_index"].values.astype(int)

    Y_sub    = store.Y[indices, :]
    HC_bench = store.HC

    if HC_bench is None:
        raise ValueError(
            "The HDF5 store does not contain a /HC group.\n"
            "Re-run cli/preprocess.py with HC_source set."
        )

    tbl_aer = cfg.get("tbl_aer")
    if tbl_aer is None:
        if store.aer_levels is not None:
            tbl_aer = store.aer_levels
        else:
            raise ValueError(
                "No AER levels found in store and tbl_aer not set in CONFIG."
            )
    tbl_aer = np.asarray(tbl_aer, dtype=np.float64)

    return Y_sub, HC_bench, tbl_aer, indices, store


def _plot_hc_comparison(HC_recon, HC_bench, tbl_aer, node_indices, out_dir):
    """Overlay reconstructed vs benchmark HCs for a sample of nodes."""
    n = len(node_indices)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Reconstructed HC vs Benchmark", fontsize=13, fontweight="bold")

    rp = 1.0 / tbl_aer   # return periods (years)

    for ax_idx, node in enumerate(node_indices):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        ax.plot(rp, HC_bench[node, :], "k-",  lw=1.5, label="Benchmark")
        ax.plot(rp, HC_recon[node, :], "r--", lw=1.5, label="Reconstructed")
        ax.set_xscale("log")
        ax.set_xlabel("Return Period (yr)"); ax.set_ylabel("Surge (m)")
        ax.set_title(f"Node {node}"); ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fpath = Path(out_dir) / "hc_comparison.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


def main():
    cfg     = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DSW + HC Reconstruction")
    print(f"  Store  : {cfg['h5_path']}")
    print(f"  Subset : {cfg['selected_csv']}")
    print(f"{'='*60}")

    # Load
    Y_sub, HC_bench, tbl_aer, indices, store = _load_inputs(cfg)
    k, m = Y_sub.shape
    print(f"\n  Selected storms : k = {k}")
    print(f"  Mesh nodes      : m = {m}")
    print(f"  AER levels      : {len(tbl_aer)}")

    # Step 1 + 2 — Global DSWs
    print("\n[1] Computing global DSWs ...")
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer)
    nan_count  = int(np.sum(np.isnan(DSW_global)))
    print(f"    DSW range : [{np.nanmin(DSW_global):.4e}, {np.nanmax(DSW_global):.4e}]")
    if nan_count:
        print(f"    Warning   : {nan_count} storms have NaN DSW (no valid nodal coverage)")

    # Step 3 — HC reconstruction
    print("\n[2] Reconstructing hazard curves ...")
    HC_recon = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, cfg["dry_threshold"])

    # Step 4 — Metrics
    from backend.engines.weights.dsw import _hc_residual_metrics
    metrics = _hc_residual_metrics(HC_recon, HC_bench)
    print(f"\n    Mean Bias        : {metrics['mean_bias']:+.4f}  m")
    print(f"    Mean Uncertainty : {metrics['mean_uncertainty']:.4f}  m")
    print(f"    Mean RMSE        : {metrics['mean_rmse']:.4f}  m")

    # Save outputs
    print("\n[3] Saving outputs ...")

    # DSW weights
    dsw_df = pd.DataFrame({
        "original_index": indices,
        "dsw_global":     DSW_global,
    })
    dsw_df.to_csv(out_dir / "dsw_weights.csv", index=False)
    print(f"    dsw_weights.csv       -> {out_dir}")

    # Reconstructed HC  [m x N_AER]
    aer_hdrs = [f"AER_{a:.3e}" for a in tbl_aer]
    pd.DataFrame(HC_recon, columns=aer_hdrs).to_csv(
        out_dir / "hc_reconstructed.csv", index=False)
    print(f"    hc_reconstructed.csv  -> {out_dir}")

    # Metrics
    pd.DataFrame([metrics]).to_csv(out_dir / "dsw_metrics.csv", index=False)
    print(f"    dsw_metrics.csv       -> {out_dir}")

    # Plot
    print("\n[4] Generating HC comparison plot ...")
    plot_nodes = cfg.get("plot_nodes")
    if plot_nodes is None:
        rng        = np.random.default_rng(42)
        n_plot     = min(cfg.get("n_plot_nodes", 12), m)
        plot_nodes = sorted(rng.choice(m, size=n_plot, replace=False).tolist())
    _plot_hc_comparison(HC_recon, HC_bench, tbl_aer, plot_nodes, out_dir)

    print("\n=== DSW run complete ===")


if __name__ == "__main__":
    main()
