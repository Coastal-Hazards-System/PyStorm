"""method_testbed - compare the GPD-location selection methods across stations.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Runs every GPD-location SELECTION method (wmse, stability, mrl, gof) on each
station's POT series through the full PST pipeline and stacks the resulting
hazard curves - ONE subplot per station - so the operator can compare, at a
glance, how each method's threshold choice propagates to the hazard curve.
Each subplot overlays the empirical POT points and the four best-estimate
hazard curves, annotated with per-method metrics (selected μ, λ_μ, and the
best-estimate magnitude at MRI = 100 and 500 yr).

Headless by design. Writes one figure per (series × fit) to
data/outputs/plots/testbed/method_testbed_<series>_<fit>.png - so the mle and
mom results coexist and never overwrite one another.

Run
---
    python scripts/method_testbed.py                 # dwl+ntr × mle+mom (default)
    python scripts/method_testbed.py --fit mom       # MoM only (both series)
    python scripts/method_testbed.py --fit mle       # MLE only
    python scripts/method_testbed.py --series dwl    # one series, both fits
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]          # PST module root
_BACKEND = ROOT / "backend" / "python"
_COMMON  = ROOT.parents[1] / "common" / "python"       # shared CyHAN common library (§5.2)
for _p in (_BACKEND, _COMMON):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from probabilistic_simulation_technique.config import PSTConfig
from probabilistic_simulation_technique.orchestrator import PSTOrchestrator
from pystorm_common import (
    WAVE_MAKER, EMPHASIS, C, EMPIRICAL, GRID,
)

# ── Testbed options ─────────────────────────────────────────────────────────
POT_OUTPUTS     = ROOT.parent / "peaks_over_threshold" / "data" / "outputs"
STATIONS        = ["8518750", "8651370", "8724580", "8761724", "8771450"]
METHODS         = ["wmse", "stability", "mrl", "gof"]
EVENTS_PER_YEAR = 10.0
MIN_EXCEEDANCES = 20
NUM_SIM         = 200
SEED            = 628
AER_MARKS       = (1e-2, 1e-3)            # AER levels (1/yr) reported in the box

_COLOR = {"wmse": WAVE_MAKER, "stability": EMPHASIS,
          "mrl": C["sea_green"], "gof": C["amber"]}


def _hc_at_aer(res, aer):
    """Best-estimate HC magnitude at a given AER (1/yr) off the 22-AER table."""
    k = int(np.nanargmin(np.abs(res.aer_table - aer)))
    return float(res.hc_table_be[k])


def _run(station, series, method, fit, tmp, ylabel):
    csv = POT_OUTPUTS / station / f"{series}_{station}_pot.csv"
    cfg = PSTConfig(
        input_csv=csv, output_dir=tmp, plots_dir=tmp,
        record_length_years=None, events_per_year=EVENTS_PER_YEAR,
        min_exceedances=MIN_EXCEEDANCES, gpd_selection=method, gpd_fit_method=fit,
        num_simulations=NUM_SIM, random_seed=SEED,
        plot_threshold_diagnostics=False, y_axis_label=ylabel)
    return PSTOrchestrator(cfg).run()


def main(series="dwl", fit="mle"):
    ylabel = {"dwl": "Detrended Water Level (m)",
              "ntr": "Non-Tidal Residual (m)"}.get(series, "Response (m)")
    tmp = Path(tempfile.mkdtemp())
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(9.5, 3.0 * len(STATIONS)),
                             sharex=True)
    fig.patch.set_facecolor("white")

    for ax, st in zip(np.atleast_1d(axes), STATIONS):
        csv = POT_OUTPUTS / st / f"{series}_{st}_pot.csv"
        if not csv.is_file():
            ax.text(0.5, 0.5, f"{st}: {csv.name} missing", transform=ax.transAxes,
                    ha="center", va="center"); continue
        v   = np.sort(pd.read_csv(csv)["value"].to_numpy())[::-1]
        n   = v.size; lam = n / (n / EVENTS_PER_YEAR)
        aer = (np.arange(1, n + 1) / (n + 1)) * lam        # empirical AER (1/yr)
        ax.scatter(aer, v, s=9, color=EMPIRICAL, alpha=0.7, zorder=1,
                   label="empirical POT")

        notes = []
        for method in METHODS:
            try:
                res = _run(st, series, method, fit, tmp, ylabel)
            except Exception as exc:                      # noqa: BLE001
                notes.append(f"{method:<9} FAILED: {str(exc)[:40]}"); continue
            ax.plot(res.hc_plot_aer, res.hc_plot_be, lw=1.7, color=_COLOR[method],
                    label=method, zorder=3)
            hc = "  ".join(f"HC@{m:g}={_hc_at_aer(res, m):.2f}" for m in AER_MARKS)
            notes.append(f"{method:<9} mu={res.gpd_threshold:5.2f} "
                         f"lam_mu={res.lambda_mu:4.2f}  {hc}")

        ax.set_xscale("log")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Station {st}", fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, which="both", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.015, 0.97, "\n".join(notes), transform=ax.transAxes, va="top",
                ha="left", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec=GRID, alpha=0.85))
        ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor=GRID,
                  loc="lower right", ncol=1)

    # AER decreases left→right (rarer events on the right), like a return plot.
    np.atleast_1d(axes)[0].invert_xaxis()
    np.atleast_1d(axes)[-1].set_xlabel(
        "Annual Exceedance Rate, AER (1/yr)   [MRI = 1 / AER]", fontsize=11)
    fig.suptitle(f"PyStorm-PST GPD-location selection comparison "
                 f"({series}, fit={fit})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out = (ROOT / "data" / "outputs" / "plots" / "testbed"
           / f"method_testbed_{series}_{fit}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[testbed] wrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare GPD-location selection "
                                "methods across stations (stacked hazard curves).")
    p.add_argument("--series", choices=["dwl", "ntr", "both"], default="both",
                   help="POT series to compare (default: both → one figure each).")
    p.add_argument("--fit", choices=["mle", "mom", "both"], default="both",
                   help="GPD fit estimator (default: both → one figure each). "
                        "Output files are named per series AND fit, so mle/mom "
                        "results never overwrite each other.")
    a = p.parse_args()
    series_list = ["dwl", "ntr"] if a.series == "both" else [a.series]
    fit_list    = ["mle", "mom"] if a.fit    == "both" else [a.fit]
    for _s in series_list:
        for _f in fit_list:
            main(series=_s, fit=_f)
