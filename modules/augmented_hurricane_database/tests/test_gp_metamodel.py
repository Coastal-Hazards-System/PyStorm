"""test_gp_metamodel - tests for the GP-metamodel imputation (gp_metamodel subpackage).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) GP recovery of a smooth function; (2) Vecchia/NNGP with a physical trend; (3) the log transform on positive data; (4) end-to-end imputation of missing values; (5) first-fix routing to the small model; (6) model-cache reuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from augmented_hurricane_database.gp_metamodel import (
    fit_gp, impute_all, impute_central_pressure,
)


def test_gp_recovers_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform([10, -80], [40, -40], size=(300, 2))
    # smooth target
    f = lambda Z: 30 + 5 * np.sin(Z[:, 0] / 5) + 0.2 * (Z[:, 1] + 60)
    y = f(X)
    # global (support) mode - also exercises the predictive variance path
    gp = fit_gp(X, y, vecchia=False, max_support=300, n_lhs=60, n_polish=2,
                seed=1, store_variance=True)

    Xt = rng.uniform([10, -80], [40, -40], size=(120, 2))
    yt = f(Xt)
    ss = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - np.sum((yt - gp.predict(Xt)) ** 2) / ss
    assert r2 > 0.95
    assert gp.loocv_r2 > 0.9
    _, std = gp.predict(Xt[:5], return_std=True)
    assert np.all(np.isfinite(std)) and np.all(std >= 0)


def test_vecchia_nngp_and_physical_trend():
    rng = np.random.default_rng(3)
    X = rng.uniform([10, -80, 40], [40, -40, 250], size=(800, 3))   # lat,lon,vmax
    # wind-pressure-like target: quadratic in vmax + smooth spatial
    f = lambda Z: 0.0007 * Z[:, 2] ** 2 + 3 * np.sin(Z[:, 0] / 6) + 0.1 * (Z[:, 1] + 60)
    y = f(X)
    gp = fit_gp(X, y, vecchia=True, n_neighbors=25, trend_linear=[2, 0],
                trend_quad=[2], max_support=400, n_lhs=50, seed=2)
    assert gp.mode == "nngp" and gp.Xn_all.shape[0] == 800
    Xt = rng.uniform([10, -80, 40], [40, -40, 250], size=(200, 3))
    yt = f(Xt)
    ss = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - np.sum((yt - gp.predict(Xt)) ** 2) / ss
    assert r2 > 0.95


def test_log_transform_positive_and_recovers():
    rng = np.random.default_rng(7)
    X = rng.uniform([10, -80], [40, -40], size=(500, 2))
    # positive, right-skewed (lognormal-ish) target
    y = np.exp(2.5 + 0.03 * (X[:, 0] - 25) + 0.2 * rng.standard_normal(500))
    gp = fit_gp(X, y, transform="log", vecchia=True, max_support=400,
                n_lhs=50, seed=1)
    assert gp.transform == "log"
    Xt = rng.uniform([10, -80], [40, -40], size=(150, 2))
    pred = gp.predict(Xt)
    assert np.all(pred > 0)                      # back-transform stays positive
    # correlation with the noiseless trend
    truth = np.exp(2.5 + 0.03 * (Xt[:, 0] - 25))
    assert np.corrcoef(pred, truth)[0, 1] > 0.8


def _synthetic_storms(rng, n_storms=40, per=12):
    rows = []
    for tc in range(1, n_storms + 1):
        lat0 = rng.uniform(12, 30)
        for s in range(per):
            lat = lat0 + 0.4 * s
            lon = rng.uniform(-85, -50)
            vmax = np.clip(rng.normal(120, 40), 40, 280)
            trans = np.nan if s == 0 else rng.uniform(10, 40)   # first fix: no motion
            head = np.nan if s == 0 else rng.uniform(-180, 180)
            # smooth ground-truth relationships
            pmin = 1010 - 0.5 * (vmax - 40) + 0.3 * (lat - 20)
            rmax = 60 - 0.1 * (vmax - 40) + 0.5 * abs(lat - 25)
            rows.append((tc, s + 1, lat, lon, vmax, trans, head, pmin, rmax))
    df = pd.DataFrame(rows, columns=[
        "tc_no", "snap_no", "lat", "lon", "vmax_kmh", "trans_kmh",
        "heading_deg", "pmin_hpa", "rmax_km"])
    return df


def test_impute_fills_missing():
    rng = np.random.default_rng(2)
    df = _synthetic_storms(rng)
    truth = df.copy()

    # knock out ~35% of pmin and rmax
    miss_p = rng.random(len(df)) < 0.35
    miss_r = rng.random(len(df)) < 0.35
    df.loc[miss_p, "pmin_hpa"] = np.nan
    df.loc[miss_r, "rmax_km"] = np.nan

    out, reports = impute_all(df, gp_kwargs={"max_support": 400, "n_lhs": 60},
                              verbose=False)

    # every missing row with usable features is filled
    assert out["pmin_hpa"].isna().sum() == 0
    assert out["rmax_km"].isna().sum() == 0
    assert reports["pmin"].n_filled == int(miss_p.sum())
    assert reports["rmax"].n_filled == int(miss_r.sum())

    # Rmax clamped to the physical band
    assert out["rmax_km"].between(8, 600).all()

    # imputed pmin tracks truth (loose) - recovered within a few hPa on average
    err = (out.loc[miss_p, "pmin_hpa"].to_numpy()
           - truth.loc[miss_p, "pmin_hpa"].to_numpy())
    assert np.sqrt(np.mean(err ** 2)) < 8.0
    # observed rows are untouched
    keep = ~miss_p
    assert np.allclose(out.loc[keep, "pmin_hpa"], truth.loc[keep, "pmin_hpa"])


def test_first_fix_routed_to_small_model():
    # single-point storm (no motion) is imputed via the 3-feature model path
    rng = np.random.default_rng(5)
    df = _synthetic_storms(rng, n_storms=30, per=10)
    # add a one-point storm with missing pmin and no motion
    one = pd.DataFrame([[999, 1, 22.0, -70.0, 150.0, np.nan, np.nan, np.nan, 50.0]],
                       columns=df.columns)
    df = pd.concat([df, one], ignore_index=True)
    out, rep = impute_central_pressure(df, gp_kwargs={"max_support": 400, "n_lhs": 50},
                                       verbose=False)
    val = out.loc[out.tc_no == 999, "pmin_hpa"].iloc[0]
    assert np.isfinite(val) and 850 < val < 1015


def test_model_cache_reuse(tmp_path):
    # With model_dir set, models are saved and a second run reuses them, yielding
    # identical output; retrain=True overwrites. Different settings -> new cache.
    rng = np.random.default_rng(7)
    df = _synthetic_storms(rng, n_storms=40, per=10)
    df.loc[rng.random(len(df)) < 0.3, "pmin_hpa"] = np.nan
    df.loc[rng.random(len(df)) < 0.3, "rmax_km"] = np.nan
    kw = dict(basin="test", model_dir=tmp_path,
              gp_kwargs={"max_support": 300, "n_lhs": 40}, verbose=False)

    out1, _ = impute_all(df, **kw)
    saved = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert {n.split("_")[1] for n in saved} == {"Cp6", "Cp3", "Rm7", "Rm4"}

    # reuse: identical results, and no error loading the cached models
    out2, _ = impute_all(df, retrain=False, **kw)
    assert np.allclose(out1["pmin_hpa"], out2["pmin_hpa"], equal_nan=True)
    assert np.allclose(out1["rmax_km"], out2["rmax_km"], equal_nan=True)
