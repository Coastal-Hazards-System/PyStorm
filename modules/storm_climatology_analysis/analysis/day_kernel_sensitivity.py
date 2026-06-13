"""Sensitivity analysis for the daily-SRR temporal kernel bandwidth (DAY_KERNEL).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The daily SRR smooths each CRL's closest-approach day-of-year (doy) values, weighted
by the same distance kernel the SRR uses (Wi), with a circular (period-365) Gaussian
temporal kernel of bandwidth h = DAY_KERNEL days. Choosing h is a kernel-density
bandwidth-selection problem on circular data.

This script answers "is 15 days a good bandwidth?" two ways, from a selection table
(selection_<basin>_<v>.csv, which carries the per-pair doy and closest-approach dist):

  1. Optimal bandwidth by weighted leave-one-out (LOO) likelihood cross-validation,
     per CRL, then aggregated over the basin. For each CRL with enough storms and a
     bandwidth h, the LOO log-likelihood is
         sum_j w_j * log( f_{-j}(doy_j) ),
         f_{-j}(d) = ( conv_h(d) - w_j*K_h(0) ) / ( W - w_j ),
         conv_h(d) = sum_{d'} c[d'] * K_h(circ_dist(d, d')),
     with c[d] the weight mass on day d, W = sum c, and K_h the circular Gaussian.
     The h that maximizes this is the data-driven optimal bandwidth for that CRL.

  2. Practical sensitivity: how the seasonal curve's peak day and roughness change
     across candidate bandwidths (so we can see how forgiving the choice is).

Run:  python analysis/day_kernel_sensitivity.py [selection.csv ...]
      (no args -> newest selection_*_*.csv under data/outputs/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "python"))
from storm_climatology_analysis.selection import gaussian_weights  # noqa: E402
from storm_climatology_analysis.gkf import DOYS, doy_diff           # noqa: E402

K_SIZE = 200.0                       # distance kernel (must match the run that built the selection)
H_GRID = np.arange(3.0, 45.1, 1.0)   # candidate bandwidths (days) to score
MIN_STORMS = 15                      # skip CRLs with fewer selected storms (LOO unstable)
DEFAULT_H = 14.0                     # the default we are testing


def _circ_kernel_matrix(h: float) -> np.ndarray:
    """365x365 circular-Gaussian kernel K_h(circ_dist(d, d')), normalized per column-day."""
    dd = doy_diff(DOYS, DOYS)                       # (365, 365) circular |day - day'|
    k = 1.0 / (np.sqrt(2.0 * np.pi) * h) * np.exp(-0.5 * (dd / h) ** 2)
    return k


def loo_scores(sel: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Per-CRL LOO log-likelihood over H_GRID; returns (mean score per h, summary)."""
    wi = gaussian_weights(K_SIZE, sel["dist"].to_numpy(float))
    doy = sel["doy"].to_numpy(int)
    crl = sel["crl_id"].to_numpy()
    k0 = 1.0 / (np.sqrt(2.0 * np.pi) * H_GRID)      # K_h(0) per bandwidth

    # Precompute the kernel matrices once (shared across CRLs).
    kmats = {h: _circ_kernel_matrix(h) for h in H_GRID}

    order = np.argsort(crl, kind="stable")
    crl_s, doy_s, wi_s = crl[order], doy[order], wi[order]
    bounds = np.flatnonzero(np.diff(crl_s)) + 1
    groups = np.split(np.arange(len(crl_s)), bounds)

    per_crl_best = []                                # optimal h for each usable CRL
    per_crl_n = []                                   # storm count (for weighting)
    total = np.zeros_like(H_GRID)                    # storm-weighted sum of LOO loglik
    n_used = 0
    for idx in groups:
        if len(idx) < MIN_STORMS:
            continue
        d = doy_s[idx]                               # 1..365
        w = wi_s[idx]
        W = w.sum()
        # Weight mass per day (1..365).
        c = np.bincount(d, weights=w, minlength=366)[1:366]   # (365,)
        crl_score = np.empty_like(H_GRID)
        ok = True
        for hi, h in enumerate(H_GRID):
            conv = kmats[h] @ c                       # (365,) full density incl. self
            fj = (conv[d - 1] - w * k0[hi]) / (W - w)  # leave-one-out density at each storm
            if np.any(fj <= 0) or not np.all(np.isfinite(fj)):
                ok = False
                break
            crl_score[hi] = float(np.sum(w * np.log(fj)))
        if not ok:
            continue
        per_crl_best.append(H_GRID[int(np.argmax(crl_score))])
        per_crl_n.append(len(idx))
        # Add the per-CRL score normalized by its weight mass so large/small CRLs compare.
        total += crl_score / W
        n_used += 1

    per_crl_best = np.array(per_crl_best)
    per_crl_n = np.array(per_crl_n)
    mean_score = total / max(n_used, 1)
    summary = {
        "n_crls_used": n_used,
        "best_h_global": float(H_GRID[int(np.argmax(mean_score))]),
        "best_h_median": float(np.median(per_crl_best)) if n_used else np.nan,
        "best_h_p25": float(np.percentile(per_crl_best, 25)) if n_used else np.nan,
        "best_h_p75": float(np.percentile(per_crl_best, 75)) if n_used else np.nan,
        "best_h_mean": float(np.average(per_crl_best, weights=per_crl_n)) if n_used else np.nan,
    }
    return mean_score, summary


def practical_sensitivity(sel: pd.DataFrame, hs=(7.0, 10.0, 15.0, 21.0, 30.0)) -> pd.DataFrame:
    """Peak day and a roughness metric of the basin-aggregate seasonal curve vs h."""
    wi = gaussian_weights(K_SIZE, sel["dist"].to_numpy(float))
    d = sel["doy"].to_numpy(int)
    c = np.bincount(d, weights=wi, minlength=366)[1:366]      # basin weight mass per day
    rows = []
    for h in hs:
        curve = _circ_kernel_matrix(h) @ c                    # (365,)
        curve = curve / curve.sum()
        peak = int(DOYS[int(np.argmax(curve))])
        # Roughness: mean |2nd difference| (circular), normalized by the peak height.
        d2 = np.abs(np.diff(curve, 2, append=curve[:2]))
        rows.append({"h_days": h, "peak_doy": peak,
                     "roughness": float(d2.mean() / curve.max())})
    return pd.DataFrame(rows)


def run_one(path: Path) -> None:
    header = pd.read_csv(path, nrows=0).columns
    if "doy" not in header:
        print(f"\n=== {path.name}: skipped (no 'doy' column; pre-daily-SRR output) ===")
        return
    sel = pd.read_csv(path, usecols=["crl_id", "doy", "dist"])
    print(f"\n=== {path.name}  ({len(sel):,} CRL-TC pairs) ===")
    mean_score, summ = loo_scores(sel)
    print(f"LOO likelihood CV over h = [{H_GRID[0]:.0f}, {H_GRID[-1]:.0f}] days, "
          f"{summ['n_crls_used']:,} CRLs (>= {MIN_STORMS} storms):")
    print(f"  optimal bandwidth (aggregate LOO peak) : {summ['best_h_global']:.0f} days")
    print(f"  per-CRL optimal h  median [p25, p75]   : "
          f"{summ['best_h_median']:.0f} [{summ['best_h_p25']:.0f}, {summ['best_h_p75']:.0f}] days "
          f"(storm-wt mean {summ['best_h_mean']:.1f})")
    # Relative LOO loss of the default vs the optimum.
    i_def = int(np.argmin(np.abs(H_GRID - DEFAULT_H)))
    best = mean_score.max()
    print(f"  LOO score at h={DEFAULT_H:.0f} vs optimum            : "
          f"{mean_score[i_def]:.4f} vs {best:.4f} "
          f"(deficit {best - mean_score[i_def]:.4f} nats/storm)")
    print("Practical sensitivity (basin-aggregate seasonal curve):")
    tbl = practical_sensitivity(sel)
    print(tbl.to_string(index=False))


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        out = ROOT / "data" / "outputs"
        args = sorted(out.glob("selection_*_*.csv"))
        if not args:
            raise SystemExit("No selection_*_*.csv found under data/outputs/.")
    for p in args:
        run_one(p)


if __name__ == "__main__":
    main()
