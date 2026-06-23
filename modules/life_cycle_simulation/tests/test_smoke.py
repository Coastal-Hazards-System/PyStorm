"""test_smoke - smoke tests for the life_cycle_simulation module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config defaults, crl_ids coercion, and validators; (2) the 365-day
calendar doy <-> (month, day) maps; (3) SRR-table prefix detection and daily-companion
location; (4) Poisson rate and stratum-probability formulas; (5) the simulator's
counts -> per-TC expansion, stratum/day draws, and reproducibility; (6) the catalog /
summary writers and additivity; (7) storm_type tc/etc dispatch (etc placeholder raises);
(8) an end-to-end run(config) over a synthetic SRR table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from life_cycle_simulation.config import LCSConfig, STRATA
from life_cycle_simulation import calendar365 as cal
from life_cycle_simulation import srr_source, simulator, writer


# ---------------------------------------------------------------------------
# Synthetic SRR fixtures (a tiny two-CRL table, Sep-peaked seasonality)
# ---------------------------------------------------------------------------

MONTHS = cal.MONTHS


def _srr_table():
    """A 2-CRL SRR table: all = low+med+high, mass concentrated in Aug/Sep."""
    rows = []
    for cid, scale in ((1, 1.0), (2, 0.5)):
        rec = {"crl_id": cid, "lat": 25.0 + cid, "lon": -90.0}
        # Per-stratum annual rates (TC/km/yr) and a simple monthly split (Aug, Sep).
        annual = {"low": 0.0010 * scale, "med": 0.0006 * scale, "high": 0.0004 * scale}
        rec["srr_all"] = sum(annual.values())
        for s in STRATA:
            rec[f"srr_{s}"] = annual[s]
            for m in MONTHS:
                frac = 0.6 if m == "Sep" else (0.4 if m == "Aug" else 0.0)
                rec[f"srr_{s}_{m}"] = annual[s] * frac
        # all monthly = sum of stratum monthly
        for m in MONTHS:
            rec[f"srr_all_{m}"] = sum(rec[f"srr_{s}_{m}"] for s in STRATA)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("crl_id", drop=False)


def _daily_table():
    """Companion daily SRR for CRL 1 + 2: a single spike on doy 245 (Sep 2)."""
    frames = []
    for cid in (1, 2):
        d = pd.DataFrame({"crl_id": cid, "doy": cal.DOYS})
        for s in STRATA:
            v = np.zeros(cal.NDOY)
            v[244] = 1.0                       # doy 245 -> index 244
            d[f"srr_daily_{s}"] = v
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# (1) config
# ---------------------------------------------------------------------------

def test_config_defaults_and_coercion():
    cfg = LCSConfig(input_csv="x.csv")
    assert cfg.storm_type == "tc"
    assert cfg.crl_ids == [844] and cfg.radius_km == 200.0
    assert cfg.sim_years == 100 and cfg.n_realizations == 1000
    assert cfg.day_method == "daily"
    # crl_ids accepts a scalar or a list, always -> list[int].
    assert LCSConfig(input_csv="x", crl_ids=7).crl_ids == [7]
    assert LCSConfig(input_csv="x", crl_ids=[3, 4]).crl_ids == [3, 4]


def test_config_validators_reject_bad_input():
    with pytest.raises(Exception):
        LCSConfig(input_csv="x", storm_type="bogus")
    with pytest.raises(Exception):
        LCSConfig(input_csv="x", day_method="weekly")
    with pytest.raises(Exception):
        LCSConfig(input_csv="x", radius_km=-5)
    with pytest.raises(Exception):
        LCSConfig(input_csv="x", sim_years=0)


# ---------------------------------------------------------------------------
# (2) calendar
# ---------------------------------------------------------------------------

def test_calendar_doy_maps():
    assert cal.doy_to_month(np.array([1]))[0] == 1     # Jan 1
    assert cal.doy_to_day(np.array([1]))[0] == 1
    assert cal.doy_to_month(np.array([365]))[0] == 12  # Dec 31
    assert cal.doy_to_day(np.array([365]))[0] == 31
    # Sep 2 is day-of-year 245 (243 days through Aug + 2).
    assert cal.doy_to_month(np.array([245]))[0] == 9
    assert cal.doy_to_day(np.array([245]))[0] == 2
    assert cal.DAYS_IN_MONTH.sum() == 365


# ---------------------------------------------------------------------------
# (3) SRR source
# ---------------------------------------------------------------------------

def test_detect_prefix_and_companion(tmp_path):
    df = _srr_table()
    assert srr_source._detect_prefix(df.columns) == "srr"
    p = tmp_path / "srr_atlantic_1938-2025_20260227.csv"
    p.write_text("crl_id\n1\n", encoding="utf-8")
    daily = tmp_path / "srr_daily_atlantic_1938-2025_20260227.csv"
    daily.write_text("crl_id\n1\n", encoding="utf-8")
    assert srr_source.locate_daily_companion(p) == daily
    assert srr_source.locate_daily_companion(tmp_path / "srr_only.csv") is None


def test_build_crl_srr_daily_vs_monthly():
    srr_df, daily_df = _srr_table(), _daily_table()
    # Daily method: the day pmf is a single spike on doy 245.
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    assert s.daily_used is True
    assert s.doy_pmf.shape == (3, 365)
    assert int(np.argmax(s.doy_pmf.sum(axis=0))) + 1 == 245
    assert s.annual["all"] == pytest.approx(0.0020)
    # Monthly method: mass split Aug/Sep, each spread uniformly within the month.
    m = srr_source.build_crl_srr(srr_df, None, 1, day_method="monthly")
    assert m.daily_used is False
    for row in m.doy_pmf:
        assert row.sum() == pytest.approx(1.0)
    # September (doy 244..273) carries 0.6 of the mass for the low stratum.
    sep = m.doy_pmf[0, 243:273].sum()
    assert sep == pytest.approx(0.6, abs=1e-9)


def test_build_crl_srr_unknown_id_raises():
    with pytest.raises(KeyError):
        srr_source.build_crl_srr(_srr_table(), None, 99, day_method="monthly")


# ---------------------------------------------------------------------------
# (4) formulas
# ---------------------------------------------------------------------------

def test_poisson_rate_and_stratum_probs():
    # lambda = SRR * 2R: 0.002 TC/km/yr over a 200 km radius -> 0.8 TC/yr.
    assert simulator.poisson_rate(0.002, 200.0) == pytest.approx(0.8)
    p = simulator.stratum_probs({"all": 0.002, "low": 0.001, "med": 0.0006, "high": 0.0004})
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.5)
    # Degenerate (all strata zero) -> even split, never a divide-by-zero.
    pe = simulator.stratum_probs({"all": 0, "low": 0, "med": 0, "high": 0})
    assert pe.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# (5) simulator
# ---------------------------------------------------------------------------

def test_simulate_shape_and_reproducibility():
    srr_df, daily_df = _srr_table(), _daily_table()
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    out1 = simulator.simulate(s, radius_km=200.0, sim_years=50, n_realizations=200,
                              rng=np.random.default_rng(np.random.SeedSequence([7, 1])))
    out2 = simulator.simulate(s, radius_km=200.0, sim_years=50, n_realizations=200,
                              rng=np.random.default_rng(np.random.SeedSequence([7, 1])))
    # Same seed -> identical catalog (reproducible).
    pd.testing.assert_frame_equal(out1.catalog, out2.catalog)
    cat = out1.catalog
    assert set(cat.columns) == {"realization", "year", "event",
                                "intensity", "doy", "month", "day"}
    assert cat["realization"].between(1, 200).all()
    assert cat["year"].between(1, 50).all()
    assert cat["doy"].between(1, 365).all()
    # Daily spike on doy 245 -> every TC lands on Sep 2.
    assert (cat["doy"] == 245).all()
    assert (cat["month"] == 9).all() and (cat["day"] == 2).all()
    # Mean count per (realization, year) tracks lambda = 0.8 within MC noise.
    mean_rate = out1.n_events / (200 * 50)
    assert mean_rate == pytest.approx(0.8, abs=0.1)


def test_simulate_zero_rate_is_empty():
    s = srr_source.CRLSrr(crl_id=1, lat=0, lon=0,
                          annual={"all": 0.0, "low": 0.0, "med": 0.0, "high": 0.0},
                          doy_pmf=np.zeros((3, 365)), daily_used=False)
    out = simulator.simulate(s, radius_km=200.0, sim_years=10, n_realizations=10,
                             rng=np.random.default_rng(0))
    assert out.n_events == 0 and len(out.catalog) == 0


def test_event_ordinal_is_within_year():
    # A high rate so most cells hold several TCs; the in-year ordinal must run 1..count.
    s = srr_source.CRLSrr(crl_id=1, lat=0, lon=0,
                          annual={"all": 0.02, "low": 0.02, "med": 0.0, "high": 0.0},
                          doy_pmf=np.tile(np.ones(365) / 365, (3, 1)), daily_used=False)
    out = simulator.simulate(s, radius_km=200.0, sim_years=20, n_realizations=50,
                             rng=np.random.default_rng(1))
    g = out.catalog.groupby(["realization", "year"])
    # Within each (realization, year) the events are 1..n with no gaps.
    assert (g["event"].max() == g.size()).all()
    assert (g["event"].min() == 1).all()


# ---------------------------------------------------------------------------
# (6) writers
# ---------------------------------------------------------------------------

def test_summary_counts_and_additivity():
    srr_df, daily_df = _srr_table(), _daily_table()
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    out = simulator.simulate(s, radius_km=200.0, sim_years=40, n_realizations=100,
                             rng=np.random.default_rng(3))
    summ = writer.build_summary(out.catalog, 100)
    # Every realization 1..100 appears (including quiet ones).
    assert summ["realization"].tolist() == list(range(1, 101))
    # Per-stratum counts sum to the total, and totals sum to the catalog size.
    assert (summ["n_low"] + summ["n_med"] + summ["n_high"] == summ["n_tc"]).all()
    assert summ["n_tc"].sum() == len(out.catalog) == out.n_events


def test_writers_roundtrip(tmp_path):
    srr_df, daily_df = _srr_table(), _daily_table()
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    out = simulator.simulate(s, radius_km=200.0, sim_years=10, n_realizations=20,
                             rng=np.random.default_rng(5))
    cp = writer.write_catalog(out.catalog, tmp_path / "cat.csv")
    summ = writer.build_summary(out.catalog, 20)
    sp = writer.write_summary(summ, tmp_path / "sum.csv")
    assert pd.read_csv(cp).shape[0] == len(out.catalog)
    assert pd.read_csv(sp).shape[0] == 20


# ---------------------------------------------------------------------------
# (7)-(8) dispatch + end-to-end
# ---------------------------------------------------------------------------

def test_storm_type_dispatch_and_placeholder():
    from life_cycle_simulation.orchestrator import LCSOrchestrator
    assert LCSConfig(input_csv="x").storm_type == "tc"
    with pytest.raises(NotImplementedError):
        LCSOrchestrator(LCSConfig(input_csv="x", storm_type="etc")).run()


# ---------------------------------------------------------------------------
# (9) plot selection + suite rendering
# ---------------------------------------------------------------------------

def test_plots_validator_expands_all_and_rejects_unknown():
    from life_cycle_simulation.config import PLOT_KEYS
    # "all" -> the canonical key list, in display order.
    assert LCSConfig(input_csv="x").plots == list(PLOT_KEYS)
    # A subset is kept (de-duplicated, reordered to canonical order).
    cfg = LCSConfig(input_csv="x", plots=["cumulative", "annual_fan", "annual_fan"])
    assert cfg.plots == ["annual_fan", "cumulative"]
    # A scalar string is accepted.
    assert LCSConfig(input_csv="x", plots="diagnostic").plots == ["diagnostic"]
    with pytest.raises(Exception):
        LCSConfig(input_csv="x", plots=["bogus_plot"])


def test_count_matrix_invariants():
    from life_cycle_simulation import plots
    srr_df, daily_df = _srr_table(), _daily_table()
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    out = simulator.simulate(s, radius_km=200.0, sim_years=25, n_realizations=80,
                             rng=np.random.default_rng(11))
    counts = plots._count_matrix(out.catalog, 80, 25)
    assert counts.shape == (80, 25)
    # Row sums are each realization's total; the grand total is the catalog length.
    summ = writer.build_summary(out.catalog, 80)
    assert counts.sum(axis=1).tolist() == summ["n_tc"].tolist()
    assert int(counts.sum()) == len(out.catalog) == out.n_events


def test_render_suite_writes_selected_figures(tmp_path):
    from life_cycle_simulation import plots
    srr_df, daily_df = _srr_table(), _daily_table()
    s = srr_source.build_crl_srr(srr_df, daily_df, 1, day_method="daily")
    out = simulator.simulate(s, radius_km=200.0, sim_years=20, n_realizations=60,
                             rng=np.random.default_rng(13))
    summ = writer.build_summary(out.catalog, 60)
    try:
        paths = plots.render_suite(
            out.catalog, summ, s, lam=out.lam, p=out.p, sim_years=20,
            n_realizations=60, plots=["annual_fan", "cumulative", "diagnostic"],
            out_dir=tmp_path, tag="t")
    except RuntimeError:
        pytest.skip("matplotlib unavailable")
    names = {p.name for p in paths}
    assert names == {"lcs_annual_fan_t.png", "lcs_cumulative_t.png",
                     "lcs_diagnostic_t.png"}
    assert all(p.is_file() for p in paths)


def test_end_to_end_run(tmp_path):
    import api_life_cycle_simulation as api
    # Write a synthetic SRR table + its daily companion, then run the module API.
    srr = _srr_table().reset_index(drop=True)
    daily = _daily_table()
    in_csv = tmp_path / "srr_atlantic_1938-2025_20260227.csv"
    srr.to_csv(in_csv, index=False)
    daily.to_csv(tmp_path / "srr_daily_atlantic_1938-2025_20260227.csv", index=False)

    result = api.run({
        "input_csv": in_csv, "crl_ids": [1, 2], "radius_km": 200.0,
        "sim_years": 30, "n_realizations": 50, "seed": 99,
        "output_dir": tmp_path / "out",
    })
    assert set(result.results) == {1, 2}
    r1 = result.results[1]
    assert r1.daily_used is True
    assert r1.lam == pytest.approx(0.8)
    assert r1.catalog_path.is_file() and r1.summary_path.is_file()
    # CRL 2 has half the SRR of CRL 1 -> roughly half the rate.
    assert result.results[2].lam == pytest.approx(0.4)
