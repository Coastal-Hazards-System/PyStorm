"""reduced_tc_suite - post-selection DSW + HC reconstruction.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Compute Discrete Storm Weights and reconstruct hazard curves for a previously
selected storm subset (the selected_storms.csv emitted by run_reduced_tc_suite.py).

Usage
-----
  1. Run scripts/run_reduced_tc_suite.py first to produce selected_storms.csv.
  2. Edit the USER OPTIONS block below.
  3. Run:  python scripts/dsw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ---------------------------------------------------------------------------
# CONFIG  - paths and DSW parameters
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":        str(_MODULE_ROOT / "data/inputs/processed/tc_data.h5"),
    # DSW post-processing follows fixed-mode selection by default.  Point this
    # at data/outputs/optimal/selected_storms.csv to post-process
    # the optimal-mode growth-loop final iteration instead.
    "selected_csv":   str(_MODULE_ROOT / "data/outputs/fixed/selected_storms.csv"),
    "output_dir":     str(_MODULE_ROOT / "data/outputs/fixed"),

    "dry_threshold": 0.0,
    "tbl_aer":       None,   # None = use levels stored in /HC attrs
    "plot_nodes":    None,
    "n_plot_nodes":  12,
}

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================


def _load_inputs(cfg):
    from reduced_tc_suite.io.store import read_store

    store = read_store(Path(cfg["h5_path"]))

    sel = pd.read_csv(cfg["selected_csv"])
    if "original_index" not in sel.columns:
        raise ValueError(
            "selected_storms.csv must contain an 'original_index' column.\n"
            "Re-run scripts/run_reduced_tc_suite.py to regenerate it."
        )
    indices = sel["original_index"].values.astype(int)

    Y_sub    = store.Y[indices, :]
    HC_bench = store.HC

    if HC_bench is None:
        raise ValueError(
            "The HDF5 store does not contain a /HC group.\n"
            "Re-run scripts/preprocess.py with HC_source set."
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

    mri = 1.0 / tbl_aer        # mean return interval (yr)

    for ax_idx, node in enumerate(node_indices):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        ax.plot(mri, HC_bench[node, :], "k-",  lw=1.5, label="Benchmark")
        ax.plot(mri, HC_recon[node, :], "r--", lw=1.5, label="Reconstructed")
        ax.set_xscale("log")
        ax.set_xlabel("Mean Return Interval, MRI (yr)"); ax.set_ylabel("Surge (m)")
        ax.set_title(f"Node {node}"); ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fpath = Path(out_dir) / "hc_comparison.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {fpath}")


def main():
    from reduced_tc_suite.weights.dsw import (
        compute_global_dsw, reconstruct_hc_global_dsw, _hc_residual_metrics,
    )

    cfg     = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DSW + HC Reconstruction")
    print(f"  Store  : {cfg['h5_path']}")
    print(f"  Subset : {cfg['selected_csv']}")
    print(f"{'='*60}")

    Y_sub, HC_bench, tbl_aer, indices, store = _load_inputs(cfg)
    k, m = Y_sub.shape
    print(f"\n  Selected storms : k = {k}")
    print(f"  Mesh nodes      : m = {m}")
    print(f"  AER levels      : {len(tbl_aer)}")

    print("\n[1] Computing global DSWs ...")
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer)
    nan_count  = int(np.sum(np.isnan(DSW_global)))
    print(f"    DSW range : [{np.nanmin(DSW_global):.4e}, {np.nanmax(DSW_global):.4e}]")
    if nan_count:
        print(f"    Warning   : {nan_count} storms have NaN DSW (no valid nodal coverage)")

    print("\n[2] Reconstructing hazard curves ...")
    HC_recon = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, cfg["dry_threshold"])

    metrics = _hc_residual_metrics(HC_recon, HC_bench)
    print(f"\n    Mean Bias        : {metrics['mean_bias']:+.4f}  m")
    print(f"    Mean Uncertainty : {metrics['mean_uncertainty']:.4f}  m")
    print(f"    Mean RMSE        : {metrics['mean_rmse']:.4f}  m")

    print("\n[3] Saving outputs ...")

    dsw_df = pd.DataFrame({
        "original_index": indices,
        "dsw_global":     DSW_global,
    })
    dsw_df.to_csv(out_dir / "dsw_weights.csv", index=False)
    print(f"    dsw_weights.csv       -> {out_dir}")

    aer_hdrs = [f"AER_{a:.3e}" for a in tbl_aer]
    pd.DataFrame(HC_recon, columns=aer_hdrs).to_csv(
        out_dir / "hc_reconstructed.csv", index=False)
    print(f"    hc_reconstructed.csv  -> {out_dir}")

    pd.DataFrame([metrics]).to_csv(out_dir / "dsw_metrics.csv", index=False)
    print(f"    dsw_metrics.csv       -> {out_dir}")

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
