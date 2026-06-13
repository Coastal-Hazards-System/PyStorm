"""test_smoke - smoke tests for the storm_surge_hydrograph (SSH) module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config construction and validation; (2) staID sign handling and storm normalization; (3) unit-hydrograph build, scaling, and double-normalization collapse; (4) overland/overwater threshold and duration<->equiv-width conversion; (5) parametric limb fit; (6) writer round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storm_surge_hydrograph.config import SSHConfig
from storm_surge_hydrograph import io
from storm_surge_hydrograph.hydrograph import (
    normalize_storm, build_unit_hydrograph, fit_limbs, scale_to_peak, _gen_gauss,
    width_at_level, threshold_depth, actual_durations,
    equiv_width_from_actual_duration, actual_duration_from_equiv_width,
)

DRY = -99999.0
DT = 0.25


def _gauss_storm(n=400, peak_idx=200, peak_elev=3.0, ground=1.0, width=20.0,
                 wet_only=True):
    """A synthetic storm column: Gaussian surge-above-ground on a NAVD88 baseline.

    Below-ground samples become the dry sentinel; trailing samples are NaN padding.
    """
    k = np.arange(n)
    surge_ag = (peak_elev - ground) * np.exp(-0.5 * ((k - peak_idx) / width) ** 2)
    elev = ground + surge_ag
    if wet_only:
        elev = np.where(surge_ag > 1e-3, elev, DRY)     # below-ground -> dry
    col = np.full(n + 50, np.nan)                        # trailing NaN padding
    col[:n] = elev
    return col


def test_config_defaults_and_validation():
    cfg = SSHConfig()
    assert cfg.dt_hours == 0.25
    assert cfg.depth_is_positive_down is True
    assert cfg.raw_dir.name == "raw" and cfg.processed_dir.name == "processed"
    with pytest.raises(ValueError):
        SSHConfig(aggregate="rms")


def test_load_staid_sign(tmp_path):
    p = tmp_path / "sta.csv"
    p.write_text("3911,29.36,-94.91,-0.43\n4149,29.36,-94.91,-2.35\n", encoding="utf-8")
    df = io.load_staid(p, depth_is_positive_down=True)
    # ground elevation above NAVD88 = -depth.
    assert df.loc[df.sp_id == 3911, "ground_elev"].iloc[0] == pytest.approx(0.43)
    assert df.loc[df.sp_id == 4149, "ground_elev"].iloc[0] == pytest.approx(2.35)


def test_normalize_storm_peak_aligned():
    col = _gauss_storm(peak_idx=200, peak_elev=3.0, ground=1.0)
    ns = normalize_storm(col, ground_elev=1.0, dt_hours=DT, dry_value=DRY,
                         min_wet_samples=5)
    assert ns is not None
    assert ns.n.max() == pytest.approx(1.0)               # normalized peak = 1
    assert ns.tau[np.argmax(ns.n)] == pytest.approx(0.0)  # peak at tau = 0
    assert ns.peak_elev == pytest.approx(3.0, abs=1e-3)
    assert ns.peak_surge == pytest.approx(2.0, abs=1e-3)
    # equivalent width (area/peak) of a Gaussian ~ sigma*sqrt(2pi); here sigma=20*dt h.
    assert ns.equiv_width == pytest.approx(20 * DT * np.sqrt(2 * np.pi), rel=0.05)


def test_normalize_storm_too_dry_returns_none():
    col = np.full(100, DRY)
    assert normalize_storm(col, 1.0, dt_hours=DT, dry_value=DRY, min_wet_samples=5) is None


def test_build_unit_hydrograph_and_scaling():
    # 12 storms, same shape, different peaks -> a clean unit hydrograph (double_norm).
    rng_peaks = np.linspace(2.5, 5.0, 12)
    cols = [_gauss_storm(peak_idx=180 + 3 * i, peak_elev=p, ground=1.0, width=18.0)
            for i, p in enumerate(rng_peaks)]
    surge = np.full((max(c.size for c in cols), len(cols)), np.nan)
    for j, c in enumerate(cols):
        surge[:c.size, j] = c
    uh = build_unit_hydrograph(surge, sp_id=1, ground_elev=1.0, dt_hours=DT,
                               dry_value=DRY, min_wet_samples=5, window_hours=None,
                               max_window_hours=72.0, aggregate="mean")
    assert uh is not None and uh.n_storms == 12
    assert uh.dimensionless is True                       # double_norm by default
    mid = uh.grid.size // 2
    assert uh.grid[mid] == pytest.approx(0.0)
    assert uh.u[mid] == pytest.approx(1.0)                # peak = 1
    assert uh.u.max() == pytest.approx(1.0)
    assert np.all(uh.u >= -1e-9) and np.all(uh.u <= 1.0 + 1e-9)

    # Scaling: E = ground + C(tau/D)*(peak - ground); max == target peak, baseline ~ ground.
    tau, elev = scale_to_peak(uh, peak_elev=4.0)
    assert elev.max() == pytest.approx(4.0, abs=1e-6)
    assert elev.min() >= uh.ground_elev - 1e-9
    assert elev.min() == pytest.approx(uh.ground_elev, abs=0.05)
    # Equivalent width sets the physical time span: doubling W doubles the width.
    tau_a, _ = scale_to_peak(uh, 4.0, equiv_width=5.0)
    tau_b, _ = scale_to_peak(uh, 4.0, equiv_width=10.0)
    assert tau_b.max() == pytest.approx(2.0 * tau_a.max(), rel=1e-9)


def test_threshold_depth_overland_vs_overwater():
    # Overland (ground >= MHHW or MHHW None): depth = offset above ground.
    assert threshold_depth(1.0, None, 0.30) == pytest.approx(0.30)
    assert threshold_depth(2.0, 0.3, 0.30) == pytest.approx(0.30)        # ground above MHHW
    # Overwater (ground < MHHW): depth = (MHHW - ground) + offset above ground.
    assert threshold_depth(-1.5, 0.3, 0.30) == pytest.approx(0.3 - (-1.5) + 0.30)


def test_actual_duration_equiv_width_conversion():
    # Clean ensemble; actual duration above 0.30 m converts to equivalent width.
    cols = [_gauss_storm(peak_elev=p, ground=1.0, width=16.0) for p in np.linspace(2.5, 5, 10)]
    surge = np.full((max(c.size for c in cols), len(cols)), np.nan)
    for j, c in enumerate(cols):
        surge[:c.size, j] = c
    uh = build_unit_hydrograph(surge, sp_id=3, ground_elev=1.0, dt_hours=DT, dry_value=DRY,
                               min_wet_samples=5, window_hours=None, max_window_hours=72.0,
                               aggregate="mean", method="double_norm")
    W = float(np.median(uh.equiv_widths))
    peak_surge = 3.0                                     # peak above ground (m)
    # Round trip W -> actual duration -> W (offset 0.30 m above ground).
    T = actual_duration_from_equiv_width(uh, W, peak_surge, offset_m=0.30)
    assert T > 0
    assert equiv_width_from_actual_duration(uh, T, peak_surge, offset_m=0.30) == pytest.approx(W, rel=1e-6)
    # Scaling by an actual duration matches scaling by the equivalent width it implies.
    ta, ea = scale_to_peak(uh, 1.0 + peak_surge, actual_duration=T, offset_m=0.30)
    tb, eb = scale_to_peak(uh, 1.0 + peak_surge, equiv_width=W)
    assert np.allclose(ta, tb) and np.allclose(ea, eb)
    # Per-storm actual durations are positive and finite for storms exceeding the threshold.
    ad = actual_durations(uh, offset_m=0.30)
    assert np.all(ad > 0) and np.all(np.isfinite(ad))


def test_double_norm_collapses_duration():
    # Two families with very different WIDTHS but same dimensionless shape collapse.
    narrow = [_gauss_storm(peak_elev=3.0, ground=1.0, width=8.0) for _ in range(6)]
    broad = [_gauss_storm(peak_elev=3.0, ground=1.0, width=24.0) for _ in range(6)]
    cols = narrow + broad
    surge = np.full((max(c.size for c in cols), len(cols)), np.nan)
    for j, c in enumerate(cols):
        surge[:c.size, j] = c
    uh = build_unit_hydrograph(surge, sp_id=2, ground_elev=1.0, dt_hours=DT, dry_value=DRY,
                               min_wet_samples=5, window_hours=None, max_window_hours=72.0,
                               aggregate="mean", method="double_norm")
    # Widths differ ~3x (8 vs 24 sigma) but the doubly-normalized stack collapses tightly.
    assert uh.equiv_widths.max() / uh.equiv_widths.min() > 2.0
    assert uh.stack.std(axis=0).mean() < 0.03            # tight collapse on s-grid
    # Amplitude-only on the same data leaves a much larger spread.
    uh_amp = build_unit_hydrograph(surge, sp_id=2, ground_elev=1.0, dt_hours=DT, dry_value=DRY,
                                   min_wet_samples=5, window_hours=None, max_window_hours=72.0,
                                   aggregate="mean", method="amplitude")
    assert uh_amp.dimensionless is False
    assert uh_amp.stack.std(axis=0).mean() > uh.stack.std(axis=0).mean()


def test_fit_limbs_recovers_gaussian():
    # A symmetric Gaussian unit hydrograph -> p ~ 2, sigma ~ true width.
    tau = np.arange(-40, 40.01, 0.25)
    u = _gen_gauss(tau, sigma=6.0, p=2.0)
    fit = fit_limbs(tau, u)
    assert fit.sigma_rise == pytest.approx(6.0, rel=0.1)
    assert fit.p_rise == pytest.approx(2.0, rel=0.2)
    assert fit.sigma_fall == pytest.approx(6.0, rel=0.1)
    assert fit.rmse < 1e-3


def test_writer_roundtrip(tmp_path):
    from storm_surge_hydrograph import writer
    cols = [_gauss_storm(peak_elev=p, ground=1.0) for p in (3.0, 4.0, 5.0)]
    surge = np.full((max(c.size for c in cols), len(cols)), np.nan)
    for j, c in enumerate(cols):
        surge[:c.size, j] = c
    uh = build_unit_hydrograph(surge, sp_id=7, ground_elev=1.0, dt_hours=DT,
                               dry_value=DRY, min_wet_samples=5, window_hours=None,
                               max_window_hours=72.0, aggregate="mean")
    uh.fit = fit_limbs(uh.tau, uh.u)
    up = writer.write_unit_hydrograph(uh, tmp_path / "u.csv")
    d = pd.read_csv(up)
    xcol = "s_dimensionless" if uh.dimensionless else "tau_hours"
    assert {xcol, "u_empirical", "u_parametric"} <= set(d.columns)
    assert d["u_empirical"].max() == pytest.approx(1.0, abs=1e-9)
    sp = writer.write_scaled_hydrograph(uh, 6.0, tmp_path / "s.csv")
    ds = pd.read_csv(sp)
    assert ds["elevation_m_navd88"].max() == pytest.approx(6.0, abs=1e-6)
