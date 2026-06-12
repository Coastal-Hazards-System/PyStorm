"""Apples-to-apples comparison of time-normalization scales for the SSH module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Double normalization scales each storm's time axis by a per-storm timescale T so the
shapes collapse. The timescale D = area/peak used in the shape-model study is one
estimator of T, but it is not a literal on-axis width. This script compares three
estimators of the time-normalization scale on identical footing:

  W_eq   equivalent width  = (integral of a) / peak           [= integral of n]
  FWHM   full width at half maximum = time with n >= 0.5      [literal on-axis width]
  W_rms  second-moment width = 2 * sqrt( integral tau^2 n / integral n )

Each is scored with the same metrics (all on the frequently wet save points):

  coverage      fraction of storms for which the scale is finite and positive
  CV            across-storm coefficient of variation of the scale
  collapse      mean across-storm STD of the doubly-normalized shape (smaller = tighter)
  loo_rmse      leave-one-out reconstruction RMSE using the storm's own scale
  fit_rmse      RMSE of the rising/falling parametric fit to the canonical shape
  corr_W_eq     correlation of the scale with W_eq (how interchangeable they are)

Run:  python analysis/timescale_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "python"))
from storm_surge_hydrograph import io, hydrograph as H   # noqa: E402

DT = 0.25
MIN_WET = 5
MIN_STORMS = 40
PHYS = np.arange(-30.0, 30.0 + DT / 2, DT)
SGRID = np.arange(-8.0, 8.0 + 0.05 / 2, 0.05)


def w_eq(tau: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(n) * DT)                              # area / peak


def fwhm(tau: np.ndarray, n: np.ndarray) -> float:
    # Width where n >= 0.5; interpolate the two half-max crossings around the peak.
    k = int(np.argmax(n))
    half = 0.5
    # rising crossing (search left of peak)
    li = k
    while li > 0 and n[li] >= half:
        li -= 1
    if n[li] >= half:                                        # never drops below on the left
        t_left = tau[0]
    else:
        t_left = np.interp(half, [n[li], n[li + 1]], [tau[li], tau[li + 1]])
    ri = k
    while ri < len(n) - 1 and n[ri] >= half:
        ri += 1
    if n[ri] >= half:
        t_right = tau[-1]
    else:
        t_right = np.interp(half, [n[ri], n[ri - 1]], [tau[ri], tau[ri - 1]])
    return float(t_right - t_left)


def w_rms(tau: np.ndarray, n: np.ndarray) -> float:
    m0 = np.sum(n)
    if m0 <= 0:
        return 0.0
    return float(2.0 * np.sqrt(np.sum(tau ** 2 * n) / m0))


ESTIMATORS: dict[str, Callable] = {"W_eq": w_eq, "FWHM": fwhm, "W_rms": w_rms}


def _load(sp_id: int):
    raw = ROOT / "data" / "inputs" / "raw"
    staid = io.load_staid(raw / "CTXCS_staID.csv")
    pts = io.discover_save_points(raw, staid, "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv",
                                  only=[sp_id])
    surge = io.load_surge_matrix(pts[0].surge_path)
    g = pts[0].ground_elev
    storms = []
    for c in range(surge.shape[1]):
        ns = H.normalize_storm(surge[:, c], g, dt_hours=DT, dry_value=-99999.0,
                               min_wet_samples=MIN_WET)
        if ns is not None:
            storms.append((ns.tau, ns.n))
    return storms


def _metrics_for(storms, scale_fn) -> dict:
    scales = np.array([scale_fn(t, n) for t, n in storms])
    ok = np.isfinite(scales) & (scales > 0)
    coverage = float(ok.mean())
    s_ok = scales[ok]
    storms_ok = [storms[i] for i in np.flatnonzero(ok)]
    cv = float(s_ok.std() / s_ok.mean())

    # Collapse spread on dimensionless grid.
    dn = np.array([np.interp(SGRID, t / sc, n, left=0, right=0)
                   for (t, n), sc in zip(storms_ok, s_ok)])
    collapse = float(dn.std(axis=0).mean())

    # Parametric-fit RMSE of the canonical shape.
    cano = dn.mean(axis=0)
    cano = cano / cano[len(SGRID) // 2] if cano[len(SGRID) // 2] > 0 else cano
    fit = H.fit_limbs(SGRID, np.clip(cano, 0, 1))

    # LOO reconstruction RMSE (storm reconstructed with its own scale).
    rmse = []
    n_ok = len(storms_ok)
    for i, ((t, n), sc) in enumerate(zip(storms_ok, s_ok)):
        acc = np.zeros_like(SGRID); k = 0
        for j, ((tj, nj), scj) in enumerate(zip(storms_ok, s_ok)):
            if j == i:
                continue
            acc += np.interp(SGRID, tj / scj, nj, left=0, right=0); k += 1
        C = acc / k
        n_phys = np.interp(PHYS, t, n, left=0, right=0)
        rec = np.interp(PHYS, SGRID * sc, C, left=0, right=0)
        rmse.append(np.sqrt(np.mean((n_phys - rec) ** 2)))
    return {"coverage": coverage, "cv": cv, "collapse": collapse,
            "loo_rmse": float(np.mean(rmse)), "fit_rmse": fit.rmse,
            "_scales": scales}


def main() -> None:
    raw = ROOT / "data" / "inputs" / "raw"
    staid = io.load_staid(raw / "CTXCS_staID.csv")
    pts = io.discover_save_points(raw, staid, "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv")
    rows = []
    for p in pts:
        storms = _load(p.sp_id)
        if len(storms) < MIN_STORMS:
            continue
        per = {name: _metrics_for(storms, fn) for name, fn in ESTIMATORS.items()}
        weq = per["W_eq"]["_scales"]
        for name in ESTIMATORS:
            m = per[name]
            sc = m["_scales"]
            both = np.isfinite(sc) & np.isfinite(weq) & (sc > 0) & (weq > 0)
            corr = float(np.corrcoef(sc[both], weq[both])[0, 1]) if both.sum() > 2 else np.nan
            rows.append({"sp_id": p.sp_id, "estimator": name, "n_storms": len(storms),
                         "coverage": round(m["coverage"], 3), "cv": round(m["cv"], 2),
                         "collapse": round(m["collapse"], 4), "loo_rmse": round(m["loo_rmse"], 4),
                         "fit_rmse": round(m["fit_rmse"], 4), "corr_W_eq": round(corr, 3)})
        print(f"SP{p.sp_id:05d}: scored")
    df = pd.DataFrame(rows)
    outdir = ROOT / "data" / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "timescale_metrics.csv", index=False)

    print("\n=== mean over scored save points (apples to apples) ===")
    agg = df.groupby("estimator")[["coverage", "cv", "collapse", "loo_rmse", "fit_rmse", "corr_W_eq"]].mean()
    agg = agg.reindex(["W_eq", "FWHM", "W_rms"])
    print(agg.round(4).to_string())
    print(f"\nwrote {outdir / 'timescale_metrics.csv'}")


if __name__ == "__main__":
    main()
