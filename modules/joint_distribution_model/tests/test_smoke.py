"""test_smoke - smoke tests for the joint_distribution_model module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config defaults + validators (storm_type, cp_source, basins);
(2) distance_weighted_adj (w=1 identity; adjusted mean/std = weighted mean/std);
(3) heading recenter wrapping; (4) adjust_crl intensity-bin thresholds; (5) ecdf_boot
shape / >= th / sorted; (6) Weibull/lognormal/normal fits recover known params and
truncated-Weibull bounds; (7) copula tau -> rho = sin(pi*tau/2), symmetric unit
diagonal; (8) sca_source version tag + DSRR accessor; (9) end-to-end run(config) on a
synthetic SCA selection CSV + DSRR npz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from joint_distribution_model.config import JDMConfig, INTENSITY_BINS
from joint_distribution_model import adjust, bootstrap, marginals, copula, sca_source


# ---------------------------------------------------------------------------
# (1) config
# ---------------------------------------------------------------------------

def test_config_defaults_and_validators():
    cfg = JDMConfig()
    assert cfg.storm_type == "tc" and cfg.basins == ["atlantic"]
    assert cfg.cp_source == "cp_gauss"
    assert (cfg.min_dp, cfg.dp_low, cfg.dp_med) == (8.0, 28.0, 48.0)
    assert JDMConfig(basins="both").basins == ["atlantic", "pacific"]
    for bad in (dict(storm_type="x"), dict(cp_source="x"), dict(basins="mars")):
        with pytest.raises(Exception):
            JDMConfig(**bad)


# ---------------------------------------------------------------------------
# (2)-(3) distance-weighted adjustment + heading recenter
# ---------------------------------------------------------------------------

def test_distance_weighted_adj_unit_weights_identity():
    x = np.array([10.0, 20.0, 35.0, 50.0, 12.0])
    adj, stat = adjust.distance_weighted_adj(x, np.ones_like(x))
    assert np.allclose(adj, x)                              # w=1 -> identity
    assert stat["mu"] == pytest.approx(stat["mu_dw"])


def test_distance_weighted_adj_recovers_weighted_moments():
    rng = np.random.default_rng(0)
    x = rng.normal(30, 10, 200)
    w = rng.uniform(0.1, 1.0, 200)
    adj, stat = adjust.distance_weighted_adj(x, w)
    # Adjusted sample mean/std equal the distance-weighted mean/std.
    assert adj.mean() == pytest.approx(stat["mu_dw"], rel=1e-9)
    assert adj.std(ddof=1) == pytest.approx(stat["sigma_dw"], rel=1e-9)


def test_heading_zero_degree_adj_wraps():
    hd = np.array([170.0, -170.0, 10.0])
    out = adjust.heading_zero_degree_adj(hd, hd_mean=-20.0)
    # 170-(-20)=190 -> -170; -170-(-20)=-150; 10-(-20)=30.
    assert out == pytest.approx([-170.0, -150.0, 30.0])


# ---------------------------------------------------------------------------
# (4) intensity binning
# ---------------------------------------------------------------------------

def test_adjust_crl_intensity_bins():
    # Five storms with deficits spanning LI/MI/HI; unit weights -> Dp unchanged.
    dp = np.array([10.0, 30.0, 50.0, 60.0, 20.0])
    cp = 1013.0 - dp
    n = dp.size
    bins = adjust.adjust_crl(
        heading=np.zeros(n), cp=cp, rmax=np.full(n, 40.0), vt=np.full(n, 20.0),
        gaussW=np.ones(n), year=np.full(n, 2000), dsrr_mean_all=0.0, dsrr_stdv_all=1.0,
        ref_pressure=1013.0, start_year=1938, min_dp=8.0, dp_low=28.0, dp_med=48.0)
    assert bins["all"].shape[0] == 5                       # all >= 8
    assert bins["high"].shape[0] == 2                      # 50, 60 (>=48)
    assert bins["med"].shape[0] == 1                       # 30
    assert bins["low"].shape[0] == 2                       # 10, 20
    # year filter drops pre-start_year storms.
    yb = adjust.adjust_crl(
        heading=np.zeros(n), cp=cp, rmax=np.full(n, 40.0), vt=np.full(n, 20.0),
        gaussW=np.ones(n), year=np.array([1900, 2000, 2000, 2000, 2000]),
        dsrr_mean_all=0.0, dsrr_stdv_all=1.0, ref_pressure=1013.0, start_year=1938,
        min_dp=8.0, dp_low=28.0, dp_med=48.0)
    assert yb["all"].shape[0] == 4


# ---------------------------------------------------------------------------
# (5) bootstrap
# ---------------------------------------------------------------------------

def test_ecdf_boot_shape_threshold_sorted():
    rng = np.random.default_rng(1)
    pot = rng.uniform(30, 100, 40)
    boot = bootstrap.ecdf_boot(pot, n_sim=200, th=28.0, rng=rng)
    assert boot.shape == (200, 40)
    assert (boot >= 28.0).all()
    assert np.all(np.diff(boot, axis=1) <= 1e-9)           # each row descending


# ---------------------------------------------------------------------------
# (6) marginal fits
# ---------------------------------------------------------------------------

def test_fit_weibull_recovers_params():
    rng = np.random.default_rng(2)
    A, k = 30.0, 1.6
    x = A * rng.weibull(k, 20000)
    Ah, kh = marginals.fit_weibull(x)
    assert Ah == pytest.approx(A, rel=0.05)
    assert kh == pytest.approx(k, rel=0.05)


def test_fit_lognorm_and_norm():
    rng = np.random.default_rng(3)
    mu, sig = np.log(40.0), 0.4
    x = np.exp(rng.normal(mu, sig, 20000))
    mh, sh = marginals.fit_lognorm(x)
    assert mh == pytest.approx(mu, abs=0.02) and sh == pytest.approx(sig, abs=0.02)
    y = rng.normal(20.0, 5.0, 20000)
    m2, s2 = marginals.fit_norm(y)
    assert m2 == pytest.approx(20.0, abs=0.1) and s2 == pytest.approx(5.0, abs=0.1)


def test_trunc_weibull_ppf_bounds():
    lo, hi = marginals.trunc_weibull_ppf([0.0, 1.0], 30.0, 1.6, 28.0, 48.0)
    assert lo == pytest.approx(28.0) and hi == pytest.approx(48.0)
    mid = float(marginals.trunc_weibull_ppf(0.5, 30.0, 1.6, 28.0, 48.0))
    assert 28.0 < mid < 48.0


# ---------------------------------------------------------------------------
# (7) copula
# ---------------------------------------------------------------------------

def test_copula_tau_to_rho():
    rng = np.random.default_rng(4)
    a = rng.normal(size=500)
    data = np.column_stack([a, a + 0.3 * rng.normal(size=500),
                            rng.normal(size=500), rng.normal(size=500)])
    tau, rho = copula.fit_copula(data)
    assert np.allclose(np.diag(rho), 1.0)
    assert np.allclose(rho, rho.T, equal_nan=True)
    assert rho[0, 1] == pytest.approx(np.sin(np.pi * tau[0, 1] / 2.0))
    assert rho[0, 1] > rho[0, 2]                           # correlated pair stronger


# ---------------------------------------------------------------------------
# (8) sca_source
# ---------------------------------------------------------------------------

def test_version_tag_and_dsrr_accessor(tmp_path):
    assert sca_source.version_tag(
        tmp_path / "selection_atlantic_1938-2025_20260227.csv",
        "atlantic") == "atlantic_1938-2025_20260227"
    # Build a tiny DSRR npz and round-trip the accessor.
    headings = np.arange(-179, 181, dtype=float)
    arrays = {"crl_id": np.array([1, 2]), "lat": np.array([25.0, 26.0]),
              "lon": np.array([-90.0, -91.0]), "headings": headings}
    for b in INTENSITY_BINS:
        arrays[f"dsrr_mean_{b}"] = np.array([10.0, 20.0])
        arrays[f"dsrr_stdv_{b}"] = np.array([30.0, 31.0])
        arrays[f"dsrr_cdf_{b}"] = np.tile(np.linspace(0, 1, headings.size + 1), (2, 1))
    p = tmp_path / "dsrr_atlantic_x.npz"
    np.savez(p, **arrays)
    d = sca_source.load_dsrr(p, bins=INTENSITY_BINS)
    assert d.coord(2) == (26.0, -91.0)
    assert d.heading_stats(1, "high") == (10.0, 30.0)
    assert d.heading_cdf(2, "low").shape[0] == headings.size + 1


# ---------------------------------------------------------------------------
# (9) end-to-end
# ---------------------------------------------------------------------------

def _write_synthetic_sca(tmp_path, n_crl=3, n_storm=80):
    """A synthetic SCA selection CSV + DSRR npz covering all intensity bins."""
    rng = np.random.default_rng(7)
    rows = []
    for cid in range(1, n_crl + 1):
        dp = 8.0 + rng.exponential(22.0, n_storm)          # spans LI/MI/HI
        rows.append(pd.DataFrame({
            "crl_id": cid, "year": rng.integers(1940, 2024, n_storm),
            "heading_deg": rng.uniform(-180, 180, n_storm),
            "trans_kmh": np.clip(rng.normal(25, 8, n_storm), 2, 80),
            "rmax_km": np.exp(rng.normal(np.log(45), 0.4, n_storm)),
            "cp_mindist": 1013.0 - dp, "cp_gauss": 1013.0 - dp,
            "gaussW": rng.uniform(0.1, 1.0, n_storm)}))
    sel = pd.concat(rows, ignore_index=True)
    sel_path = tmp_path / "selection_atlantic_1938-2025_20260227.csv"
    sel.to_csv(sel_path, index=False)

    headings = np.arange(-179, 181, dtype=float)
    arrays = {"crl_id": np.arange(1, n_crl + 1),
              "lat": 25.0 + np.arange(n_crl), "lon": -90.0 - np.arange(n_crl),
              "headings": headings}
    for b in INTENSITY_BINS:
        arrays[f"dsrr_mean_{b}"] = np.full(n_crl, 5.0)
        arrays[f"dsrr_stdv_{b}"] = np.full(n_crl, 35.0)
        arrays[f"dsrr_cdf_{b}"] = np.tile(np.linspace(0, 1, headings.size + 1), (n_crl, 1))
    np.savez(tmp_path / "dsrr_atlantic_1938-2025_20260227.npz", **arrays)
    return sel_path


def test_end_to_end_run(tmp_path):
    import api_joint_distribution_model as api
    _write_synthetic_sca(tmp_path)
    result = api.run({
        "basins": "atlantic", "sca_outputs_dir": tmp_path,
        "output_dir": tmp_path / "out", "n_boot": 40, "n_jobs": 1, "seed": 1,
    })
    br = result.results["atlantic"]
    assert br.n_crls == 3 and br.n_records > 0
    assert br.marginals_path.is_file() and br.copula_path.is_file()
    assert br.adjusted_path.is_file()

    marg = pd.read_csv(br.marginals_path)
    # Every CRL x intensity x param row present; Weibull Dp params finite for HI/MI.
    assert set(marg["intensity"]) >= {"all", "high", "med", "low"}
    hi_dp = marg[(marg.intensity == "high") & (marg.param == "Dp")]
    assert np.isfinite(hi_dp["p1"]).all() and np.isfinite(hi_dp["p2"]).all()

    z = np.load(br.copula_path)
    rho = z["rho_high"]
    assert rho.shape == (3, 4, 4)
    diag = np.diagonal(rho, axis1=1, axis2=2)
    assert np.allclose(diag[np.isfinite(diag)], 1.0)       # unit diagonal
