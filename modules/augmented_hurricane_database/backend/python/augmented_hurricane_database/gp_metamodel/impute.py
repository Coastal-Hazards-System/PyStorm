"""GP-metamodel data imputation for central pressure and radius of max wind.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of the CHS DI scripts (CHS_TC_HURDAT_Atlantic_DI_Cp.m /
CHS_TC_HURDATv2_Atlantic_DI_Rm.m), with three optional upgrades over the
original (all on by default):

  * Vecchia / NNGP — predictions condition on all training fixes, not a subset.
  * physical mean — a wind–pressure (for Cp) / lat·deficit (for Rmax) kriging
    trend instead of a constant mean.
  * parallel training — the two models per target are fit concurrently.

The metamodels are SELF-TRAINED on rows where the target is observed, then used
to predict the missing rows:
  1. Central pressure. Cp6 (known motion) and Cp3 (fallback) on observed-pmin
     rows; predict the missing rows — Cp6 where a fix has Vf and Hd, Cp3 else.
  2. Radius of max wind. Using pressure-completed data, Rm7 / Rm4 on observed-
     rmax rows; predict the missing rows and clamp to [8, 600] km.

Observed values are kept; only missing rows are filled (matching the mains).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .features import (
    CP_BASE, RMAX_MAX, RMAX_MIN, cp_features_full, cp_features_small,
    finite_rows, motion_known, rm_features_full, rm_features_small,
)
from .gp import GPModel, fit_gp

# Physical-mean trend column indices (into each model's feature matrix).
# Cp deficit ~ wind–pressure: linear [vmax, lat] + quadratic [vmax].
_CP_TREND_FULL = ([2, 0], [2])     # [lat,lon,vmax,Vf,sinHd,cosHd]
_CP_TREND_SMALL = ([2, 0], [2])    # [lat,lon,vmax]
# Rmax ~ latitude + intensity: linear [lat, Cp-deficit, vmax].
_RM_TREND_FULL = ([0, 3, 2], [])   # [lat,lon,vmax,Cpdef,Vf,sinHd,cosHd]
_RM_TREND_SMALL = ([0, 3, 2], [])  # [lat,lon,vmax,Cpdef]


@dataclass
class ImputeReport:
    target: str
    n_missing: int = 0
    n_filled: int = 0
    n_train_full: int = 0
    n_train_small: int = 0
    loocv: Dict[str, dict] = field(default_factory=dict)


def _run_two(f_full, f_small, parallel: bool):
    """Run the two model fits, concurrently when ``parallel`` (GIL released in
    the C++ kernel / LAPACK, so the fits overlap)."""
    if not parallel:
        return f_full(), f_small()
    with ThreadPoolExecutor(max_workers=2) as ex:
        a, b = ex.submit(f_full), ex.submit(f_small)
        return a.result(), b.result()


def _common_kwargs(gp_kwargs, vecchia, n_neighbors):
    base = dict(gp_kwargs or {})
    base.setdefault("vecchia", vecchia)
    base.setdefault("n_neighbors", n_neighbors)
    return base


def impute_central_pressure(
    df: pd.DataFrame, *, gp_kwargs: Optional[dict] = None,
    vecchia: bool = True, physical_mean: bool = True, n_neighbors: int = 30,
    parallel: bool = True, verbose: bool = True,
):
    """Fill missing ``pmin_hpa`` with GP-metamodel predictions. Returns (df, report)."""
    df = df.copy()
    pmin = df["pmin_hpa"].to_numpy(float)
    known = np.isfinite(pmin)
    y = CP_BASE - pmin
    mk = motion_known(df)
    Xfull, Xsmall = cp_features_full(df), cp_features_small(df)
    ok_full, ok_small = finite_rows(Xfull), finite_rows(Xsmall)

    rep = ImputeReport(target="pmin_hpa", n_missing=int((~known).sum()))
    tr6, tr3 = known & ok_full, known & ok_small
    rep.n_train_full, rep.n_train_small = int(tr6.sum()), int(tr3.sum())

    base = _common_kwargs(gp_kwargs, vecchia, n_neighbors)
    lin6, q6 = _CP_TREND_FULL if physical_mean else ([], [])
    lin3, q3 = _CP_TREND_SMALL if physical_mean else ([], [])
    gp6, gp3 = _run_two(
        lambda: fit_gp(Xfull[tr6], y[tr6], trend_linear=lin6, trend_quad=q6, **base),
        lambda: fit_gp(Xsmall[tr3], y[tr3], trend_linear=lin3, trend_quad=q3, **base),
        parallel)
    rep.loocv = {"Cp6": dict(r2=gp6.loocv_r2, rmse=gp6.loocv_rmse),
                 "Cp3": dict(r2=gp3.loocv_r2, rmse=gp3.loocv_rmse)}
    if verbose:
        print(f"[gpm] Cp6 train={rep.n_train_full:,} LOOCV R2={gp6.loocv_r2:.3f} "
              f"RMSE={gp6.loocv_rmse:.2f} hPa | Cp3 train={rep.n_train_small:,} "
              f"LOOCV R2={gp3.loocv_r2:.3f} RMSE={gp3.loocv_rmse:.2f} hPa")

    use_full = mk & ok_full
    pred = np.full(len(df), np.nan)
    if use_full.any():
        pred[use_full] = CP_BASE - gp6.predict(Xfull[use_full])
    use_small = (~use_full) & ok_small
    if use_small.any():
        pred[use_small] = CP_BASE - gp3.predict(Xsmall[use_small])
    pred = np.round(pred)

    df, rep.n_filled = _apply(df, "pmin_hpa", known, pred)
    if verbose:
        print(f"[gpm] central pressure: filled {rep.n_filled:,} of "
              f"{rep.n_missing:,} missing rows")
    return df, rep


def impute_rmax(
    df: pd.DataFrame, *, gp_kwargs: Optional[dict] = None,
    vecchia: bool = True, physical_mean: bool = True, log_rmax: bool = True,
    n_neighbors: int = 30, parallel: bool = True,
    clamp=(RMAX_MIN, RMAX_MAX), verbose: bool = True,
):
    """Fill missing ``rmax_km`` with GP-metamodel predictions. Returns (df, report).

    With ``log_rmax`` the GP is fit in log space — Rmax is positive and roughly
    lognormal, so a log transform both matches the log-linear size physics and
    stabilizes the strongly intensity-dependent scatter.
    """
    df = df.copy()
    pmin = df["pmin_hpa"].to_numpy(float)
    rmax = df["rmax_km"].to_numpy(float)
    known = np.isfinite(rmax)
    mk = motion_known(df)
    Xfull, Xsmall = rm_features_full(df, pmin), rm_features_small(df, pmin)
    ok_full, ok_small = finite_rows(Xfull), finite_rows(Xsmall)

    rep = ImputeReport(target="rmax_km", n_missing=int((~known).sum()))
    tr7, tr4 = known & ok_full, known & ok_small
    rep.n_train_full, rep.n_train_small = int(tr7.sum()), int(tr4.sum())

    base = _common_kwargs(gp_kwargs, vecchia, n_neighbors)
    tfm = "log" if log_rmax else "none"
    lin7, q7 = _RM_TREND_FULL if physical_mean else ([], [])
    lin4, q4 = _RM_TREND_SMALL if physical_mean else ([], [])
    gp7, gp4 = _run_two(
        lambda: fit_gp(Xfull[tr7], rmax[tr7], trend_linear=lin7, trend_quad=q7,
                       transform=tfm, **base),
        lambda: fit_gp(Xsmall[tr4], rmax[tr4], trend_linear=lin4, trend_quad=q4,
                       transform=tfm, **base),
        parallel)
    rep.loocv = {"Rm7": dict(r2=gp7.loocv_r2, rmse=gp7.loocv_rmse),
                 "Rm4": dict(r2=gp4.loocv_r2, rmse=gp4.loocv_rmse)}
    if verbose:
        print(f"[gpm] Rm7 train={rep.n_train_full:,} LOOCV R2={gp7.loocv_r2:.3f} "
              f"RMSE={gp7.loocv_rmse:.2f} km | Rm4 train={rep.n_train_small:,} "
              f"LOOCV R2={gp4.loocv_r2:.3f} RMSE={gp4.loocv_rmse:.2f} km")

    use_full = mk & ok_full
    pred = np.full(len(df), np.nan)
    if use_full.any():
        pred[use_full] = gp7.predict(Xfull[use_full])
    use_small = (~use_full) & ok_small
    if use_small.any():
        pred[use_small] = gp4.predict(Xsmall[use_small])
    pred = np.round(np.clip(pred, *clamp))

    df, rep.n_filled = _apply(df, "rmax_km", known, pred)
    if verbose:
        print(f"[gpm] radius of max wind: filled {rep.n_filled:,} of "
              f"{rep.n_missing:,} missing rows")
    return df, rep


def impute_all(
    df: pd.DataFrame, *, targets=("pmin", "rmax"),
    gp_kwargs: Optional[dict] = None, vecchia: bool = True,
    physical_mean: bool = True, log_rmax: bool = True,
    parallel: bool = True, verbose: bool = True,
    cp_max_support: int = 6000, cp_neighbors: int = 30,
    rmax_max_support: int = 3000, rmax_neighbors: int = 10,
):
    """Run central-pressure then Rmax imputation (Rmax uses completed pressure).

    Cp and Rmax use independently tuned settings: Cp (smooth, long-range) takes a
    larger calibration support; Rmax (short-range, noisy) takes a small NNGP
    conditioning set. ``gp_kwargs`` supplies any shared overrides.
    """
    shared = dict(vecchia=vecchia, physical_mean=physical_mean, parallel=parallel,
                  verbose=verbose)
    reports: Dict[str, ImputeReport] = {}
    if "pmin" in targets:
        df, reports["pmin"] = impute_central_pressure(
            df, gp_kwargs={**(gp_kwargs or {}), "max_support": cp_max_support},
            n_neighbors=cp_neighbors, **shared)
    if "rmax" in targets:
        df, reports["rmax"] = impute_rmax(
            df, gp_kwargs={**(gp_kwargs or {}), "max_support": rmax_max_support},
            n_neighbors=rmax_neighbors, log_rmax=log_rmax, **shared)
    return df, reports


def _apply(df, col, known, pred):
    """Fill the missing rows of ``col`` from ``pred``. Returns (df, n_filled)."""
    fill = (~known) & np.isfinite(pred)
    df.loc[fill, col] = pred[fill]
    return df, int(fill.sum())
