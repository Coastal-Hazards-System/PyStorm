"""GP-metamodel data imputation for central pressure and radius of max wind.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of the CHS DI scripts (CHS_TC_HURDAT_Atlantic_DI_Cp.m /
CHS_TC_HURDATv2_Atlantic_DI_Rm.m), with three optional upgrades over the
original (all on by default):

  * Vecchia / NNGP - predictions condition on all training fixes, not a subset.
  * physical mean - a wind-pressure (for Cp) / lat·deficit (for Rmax) kriging
    trend instead of a constant mean.
  * parallel training - the two models per target are fit concurrently.

The metamodels are SELF-TRAINED on rows where the target is observed, then used
to predict the missing rows:
  1. Central pressure. Cp6 (known motion) and Cp3 (fallback) on observed-pmin
     rows; predict the missing rows - Cp6 where a fix has Vf and Hd, Cp3 else.
  2. Radius of max wind. Using pressure-completed data, Rm7 / Rm4 on observed-
     rmax rows; predict the missing rows and clamp to [8, 600] km.

Observed values are kept; only missing rows are filled (matching the mains).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .features import (
    CP_BASE, RMAX_MAX, RMAX_MIN, cp_features_full, cp_features_small,
    finite_rows, motion_known, rm_features_full, rm_features_small,
)
from .gp import GPModel, fit_gp

# Physical-mean trend column indices (into each model's feature matrix).
# Cp deficit ~ wind-pressure: linear [vmax, lat] + quadratic [vmax].
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


def _common_kwargs(gp_kwargs, n_neighbors, n_cal, n_lhs):
    base = dict(gp_kwargs or {})
    base.setdefault("n_neighbors", n_neighbors)
    base.setdefault("n_cal", n_cal)
    base.setdefault("n_lhs", n_lhs)
    return base


def _cache_path(model_dir, basin, name, y, settings):
    """Cache file for one model, keyed by basin, model, and a signature of the
    settings plus a fingerprint of the training data. None disables caching."""
    if model_dir is None:
        return None
    sig_src = dict(settings, n=int(np.size(y)), ysum=round(float(np.nansum(y)), 2))
    sig = hashlib.md5(json.dumps(sig_src, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return Path(model_dir) / f"{basin}_{name}_{sig}.npz"


def _fit_cached(path, retrain, fit_fn):
    """Load a matching cached model when present and not retraining; otherwise fit
    and (if a path is given) save. Returns the model."""
    if path is not None and not retrain and path.exists():
        return GPModel.load(path)
    model = fit_fn()
    if path is not None:
        model.save(path)
    return model


def impute_central_pressure(
    df: pd.DataFrame, *, gp_kwargs: Optional[dict] = None,
    vecchia_full: bool = True, vecchia_small: bool = True,
    physical_mean: bool = True, n_neighbors: int = 30,
    n_cal: int = 1200, n_lhs: int = 120, parallel: bool = True, verbose: bool = True,
    basin: str = "", model_dir=None, retrain: bool = False,
):
    """Fill missing ``pmin_hpa`` with GP-metamodel predictions. Returns (df, report).

    ``vecchia_full`` / ``vecchia_small`` select the solver per model (Cp6 / Cp3):
    True is the nearest-neighbor GP, False the exact full GP over the support.
    When ``model_dir`` is set, trained models are cached there and reused unless
    ``retrain`` is True.
    """
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

    base = _common_kwargs(gp_kwargs, n_neighbors, n_cal, n_lhs)
    lin6, q6 = _CP_TREND_FULL if physical_mean else ([], [])
    lin3, q3 = _CP_TREND_SMALL if physical_mean else ([], [])
    s6 = dict(base, vecchia=vecchia_full, lin=list(lin6), q=list(q6))
    s3 = dict(base, vecchia=vecchia_small, lin=list(lin3), q=list(q3))
    p6 = _cache_path(model_dir, basin, "Cp6", y[tr6], s6)
    p3 = _cache_path(model_dir, basin, "Cp3", y[tr3], s3)
    gp6, gp3 = _run_two(
        lambda: _fit_cached(p6, retrain, lambda: fit_gp(
            Xfull[tr6], y[tr6], trend_linear=lin6, trend_quad=q6, vecchia=vecchia_full, **base)),
        lambda: _fit_cached(p3, retrain, lambda: fit_gp(
            Xsmall[tr3], y[tr3], trend_linear=lin3, trend_quad=q3, vecchia=vecchia_small, **base)),
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
    vecchia_full: bool = True, vecchia_small: bool = True,
    physical_mean: bool = True, log_rmax: bool = True,
    n_neighbors: int = 30, n_cal: int = 1200, n_lhs: int = 120,
    parallel: bool = True, clamp=(RMAX_MIN, RMAX_MAX), verbose: bool = True,
    basin: str = "", model_dir=None, retrain: bool = False, train_pmin=None,
):
    """Fill missing ``rmax_km`` with GP-metamodel predictions. Returns (df, report).

    Follows the MATLAB workflow: the models are TRAINED on observed pressure
    (``train_pmin``, dropping rows with missing pressure) and used to PREDICT the
    missing Rmax with the completed pressure (``df['pmin_hpa']``). When
    ``train_pmin`` is None the completed pressure is used for both.

    With ``log_rmax`` the GP is fit in log space - Rmax is positive and roughly
    lognormal, so a log transform both matches the log-linear size physics and
    stabilizes the strongly intensity-dependent scatter. ``vecchia_full`` /
    ``vecchia_small`` select the solver per model (Rm7 / Rm4). When ``model_dir``
    is set, trained models are cached there and reused unless ``retrain`` is True.
    """
    df = df.copy()
    pmin = df["pmin_hpa"].to_numpy(float)                 # completed (for prediction)
    pmin_tr = pmin if train_pmin is None else np.asarray(train_pmin, float)  # observed (for training)
    rmax = df["rmax_km"].to_numpy(float)
    known = np.isfinite(rmax)
    mk = motion_known(df)
    # Training features use observed pressure; prediction features use completed.
    Xtr_full, Xtr_small = rm_features_full(df, pmin_tr), rm_features_small(df, pmin_tr)
    Xpr_full, Xpr_small = rm_features_full(df, pmin), rm_features_small(df, pmin)
    ok_tr_full, ok_tr_small = finite_rows(Xtr_full), finite_rows(Xtr_small)

    rep = ImputeReport(target="rmax_km", n_missing=int((~known).sum()))
    tr7, tr4 = known & ok_tr_full, known & ok_tr_small
    rep.n_train_full, rep.n_train_small = int(tr7.sum()), int(tr4.sum())

    base = _common_kwargs(gp_kwargs, n_neighbors, n_cal, n_lhs)
    tfm = "log" if log_rmax else "none"
    lin7, q7 = _RM_TREND_FULL if physical_mean else ([], [])
    lin4, q4 = _RM_TREND_SMALL if physical_mean else ([], [])
    s7 = dict(base, vecchia=vecchia_full, transform=tfm, lin=list(lin7), q=list(q7))
    s4 = dict(base, vecchia=vecchia_small, transform=tfm, lin=list(lin4), q=list(q4))
    p7 = _cache_path(model_dir, basin, "Rm7", rmax[tr7], s7)
    p4 = _cache_path(model_dir, basin, "Rm4", rmax[tr4], s4)
    gp7, gp4 = _run_two(
        lambda: _fit_cached(p7, retrain, lambda: fit_gp(
            Xtr_full[tr7], rmax[tr7], trend_linear=lin7, trend_quad=q7,
            transform=tfm, vecchia=vecchia_full, **base)),
        lambda: _fit_cached(p4, retrain, lambda: fit_gp(
            Xtr_small[tr4], rmax[tr4], trend_linear=lin4, trend_quad=q4,
            transform=tfm, vecchia=vecchia_small, **base)),
        parallel)
    rep.loocv = {"Rm7": dict(r2=gp7.loocv_r2, rmse=gp7.loocv_rmse),
                 "Rm4": dict(r2=gp4.loocv_r2, rmse=gp4.loocv_rmse)}
    if verbose:
        print(f"[gpm] Rm7 train={rep.n_train_full:,} LOOCV R2={gp7.loocv_r2:.3f} "
              f"RMSE={gp7.loocv_rmse:.2f} km | Rm4 train={rep.n_train_small:,} "
              f"LOOCV R2={gp4.loocv_r2:.3f} RMSE={gp4.loocv_rmse:.2f} km")

    ok_pr_full, ok_pr_small = finite_rows(Xpr_full), finite_rows(Xpr_small)
    use_full = mk & ok_pr_full
    pred = np.full(len(df), np.nan)
    if use_full.any():
        pred[use_full] = gp7.predict(Xpr_full[use_full])
    use_small = (~use_full) & ok_pr_small
    if use_small.any():
        pred[use_small] = gp4.predict(Xpr_small[use_small])
    pred = np.round(np.clip(pred, *clamp))

    df, rep.n_filled = _apply(df, "rmax_km", known, pred)
    if verbose:
        print(f"[gpm] radius of max wind: filled {rep.n_filled:,} of "
              f"{rep.n_missing:,} missing rows")
    return df, rep


def impute_all(
    df: pd.DataFrame, *, targets=("pmin", "rmax"),
    gp_kwargs: Optional[dict] = None,
    cp6_vecchia: bool = True, cp3_vecchia: bool = True,
    rm7_vecchia: bool = True, rm4_vecchia: bool = True,
    physical_mean: bool = True, log_rmax: bool = True,
    parallel: bool = True, verbose: bool = True,
    cp_max_support: int = 6000, cp_neighbors: int = 30,
    cp_n_cal: int = 1200, cp_n_lhs: int = 120,
    rmax_max_support: int = 3000, rmax_neighbors: int = 10,
    rmax_n_cal: int = 1200, rmax_n_lhs: int = 120,
    basin: str = "", model_dir=None, retrain: bool = False,
):
    """Run central-pressure then Rmax imputation (Rmax uses completed pressure).

    Cp and Rmax use independently tuned settings: Cp (smooth, long-range) takes a
    larger calibration support; Rmax (short-range, noisy) takes a small NNGP
    conditioning set. The solver is selectable per model via the ``*_vecchia``
    flags (True = NNGP, False = exact full GP). ``n_cal`` / ``n_lhs`` set the
    calibration-subset size and the Latin-hypercube budget per target; the full GP
    needs deeper values than the NNGP. ``gp_kwargs`` supplies any shared overrides.
    """
    reports: Dict[str, ImputeReport] = {}
    # Observed pressure, captured before Cp imputation. Following the MATLAB
    # workflow, the Rmax models are trained on this (observed) pressure, then used
    # to predict the missing Rmax with the completed pressure.
    pmin_observed = df["pmin_hpa"].to_numpy(float).copy()
    if "pmin" in targets:
        df, reports["pmin"] = impute_central_pressure(
            df, gp_kwargs={**(gp_kwargs or {}), "max_support": cp_max_support},
            vecchia_full=cp6_vecchia, vecchia_small=cp3_vecchia,
            physical_mean=physical_mean, n_neighbors=cp_neighbors,
            n_cal=cp_n_cal, n_lhs=cp_n_lhs, parallel=parallel, verbose=verbose,
            basin=basin, model_dir=model_dir, retrain=retrain)
    if "rmax" in targets:
        df, reports["rmax"] = impute_rmax(
            df, gp_kwargs={**(gp_kwargs or {}), "max_support": rmax_max_support},
            vecchia_full=rm7_vecchia, vecchia_small=rm4_vecchia,
            physical_mean=physical_mean, log_rmax=log_rmax, n_neighbors=rmax_neighbors,
            n_cal=rmax_n_cal, n_lhs=rmax_n_lhs, parallel=parallel, verbose=verbose,
            basin=basin, model_dir=model_dir, retrain=retrain, train_pmin=pmin_observed)
    return df, reports


def _apply(df, col, known, pred):
    """Fill the missing rows of ``col`` from ``pred``. Returns (df, n_filled)."""
    fill = (~known) & np.isfinite(pred)
    df.loc[fill, col] = pred[fill]
    return df, int(fill.sum())
