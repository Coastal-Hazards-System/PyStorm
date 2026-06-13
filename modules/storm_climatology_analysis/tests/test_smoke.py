"""test_smoke - smoke tests for the storm_climatology_analysis module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config basin defaults; (2) CRL loading (Atlantic CSV + Pacific tab); (3) Gaussian weights, Haversine distance, azimuth/doy circular wrapping; (4) storm selection within/beyond max_dist and high-pressure dropping; (5) daily and monthly SRR additivity; (6) SRR table/radius writers and units;
(7) storm_type tc/etc dispatch (etc placeholder raises NotImplementedError).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storm_climatology_analysis.config import SCAConfig
from storm_climatology_analysis.crls import load_crls
from storm_climatology_analysis.selection import (
    select_storms, gaussian_weights, _haversine_km, ymd_to_doy,
)
from storm_climatology_analysis.gkf import (
    azimuth_diff, heading_zero_degree_adj, compute_rates, doy_diff,
    HEADINGS, MONTHS, DOYS,
)


def _one_storm_hurdat(month=9, pmin=980.0, lat=(24.0, 25.0), lon=(-90.0, -90.0),
                      heading=0.0):
    """A two-fix storm near (25,-90); second fix is the closest approach."""
    return pd.DataFrame({
        "tc_no":   [1, 1],
        "year":    [2000, 2000],
        "nhc_id":  ["AL012000", "AL012000"],
        "name":    ["TESTSTORM", "TESTSTORM"],
        "ymd":     [20000900 + 1, 20000000 + month * 100 + 2],
        "lat":     list(lat),
        "lon":     list(lon),
        "vmax_kmh": [120.0, 130.0],
        "pmin_hpa": [pmin, pmin],
        "trans_kmh": [20.0, 20.0],
        "heading_deg": [heading, heading],
        "rmax_km": [40.0, 40.0],
    })


def test_config_basins_default_both():
    cfg = SCAConfig()
    assert cfg.basins == ["atlantic", "pacific"]     # both on now that Pacific CRLs exist
    assert cfg.pacific_crl_file == "CHS_PAC_CRLs_v1.2.txt"
    assert SCAConfig(basins="atlantic").basins == ["atlantic"]
    # raw/processed input convention
    assert cfg.raw_dir.name == "raw" and cfg.processed_dir.name == "processed"


def test_load_crls(tmp_path):
    p = tmp_path / "crls.csv"
    p.write_text("ID,lat,lon\n1,25.0,-90.0\n2,30.0,-80.0\n", encoding="utf-8")
    crls = load_crls(p)
    assert list(crls.columns) == ["id", "lat", "lon"]
    assert crls["id"].tolist() == [1, 2]
    assert crls["lat"].iloc[1] == 30.0


def test_load_crls_pacific_tab_format(tmp_path):
    # Pacific CRLs are tab-delimited with Latitude/Longitude/Region/ID headers.
    p = tmp_path / "pac.txt"
    p.write_text("Latitude\tLongitude\tRegion\tID\n"
                 "25.17\t-165.0\t1\t1\n24.99\t-164.66\t1\t2\n", encoding="utf-8")
    crls = load_crls(p)
    assert crls["id"].tolist() == [1, 2]
    assert crls["lat"].iloc[0] == pytest.approx(25.17)
    assert crls["lon"].iloc[1] == pytest.approx(-164.66)
    assert "region" in crls.columns


def test_gaussian_weights_peak():
    # At zero distance the weight is the kernel normalization 1/(sqrt(2pi)*K).
    assert gaussian_weights(200.0, np.array([0.0]))[0] == pytest.approx(
        1.0 / (np.sqrt(2 * np.pi) * 200.0))
    # Monotone decreasing with distance.
    w = gaussian_weights(200.0, np.array([0.0, 100.0, 400.0]))
    assert w[0] > w[1] > w[2]


def test_haversine_known_distance():
    # 1 degree of latitude ~ 111.2 km on the 6371 km sphere.
    d = _haversine_km(np.array([0.0]), np.array([0.0]),
                      np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(111.19, abs=0.5)


def test_azimuth_diff_wraps():
    d = azimuth_diff(HEADINGS, np.array([90.0]))     # (1, 360)
    # difference is zero at the grid heading 90 and 180 at the opposite heading.
    assert d[0, np.argmin(np.abs(HEADINGS - 90))] == pytest.approx(0.0)
    assert d[0, np.argmin(np.abs(HEADINGS + 90))] == pytest.approx(180.0)
    # symmetric around the storm heading
    assert d[0, np.argmin(np.abs(HEADINGS - 60))] == pytest.approx(
        d[0, np.argmin(np.abs(HEADINGS - 120))])


def test_heading_zero_degree_adj_recovers_mean():
    pdf0 = np.zeros_like(HEADINGS)
    pdf0[np.argmin(np.abs(HEADINGS - 45))] = 1.0     # all mass at +45 deg
    pdf, mean, stdv = heading_zero_degree_adj(pdf0)
    assert mean == pytest.approx(45.0, abs=1.0)
    assert pdf.sum() == pytest.approx(1.0)


def test_select_storms_within_and_beyond():
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(), crls, max_dist=600.0)
    assert len(sel) == 1
    assert sel["crl_id"].iloc[0] == 1
    assert sel["dp"].iloc[0] == pytest.approx(1013 - 980)
    assert sel["month"].iloc[0] == 9                 # closest-approach month
    assert sel["dist"].iloc[0] == pytest.approx(0.0, abs=1.0)   # CRL sits on the 2nd fix

    # A storm far away (in the Pacific) selects nothing.
    far = _one_storm_hurdat(lat=(5.0, 6.0), lon=(-150.0, -150.0))
    assert select_storms(far, crls, max_dist=600.0).empty


def test_ymd_to_doy():
    # Jan 1 -> 1, Dec 31 -> 365, and a Sep 2 closest approach -> 245.
    assert ymd_to_doy(np.array([20000101]))[0] == 1
    assert ymd_to_doy(np.array([20001231]))[0] == 365
    assert ymd_to_doy(np.array([20000902]))[0] == 245   # 243 (Aug end) + 2


def test_doy_diff_wraps_circularly():
    d = doy_diff(DOYS, np.array([1.0]))                  # (1, 365)
    # zero at day 1; near a full half-period from mid-year; wraps so Dec 31 is 1 day off.
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, np.argmin(np.abs(DOYS - 365))] == pytest.approx(1.0)  # Dec31 ~ 1 day from Jan1


def test_select_storms_records_doy():
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(month=9, pmin=980.0), crls)
    assert sel["doy"].iloc[0] == 245                     # Sep 2 closest approach


def test_compute_rates_daily_additivity():
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(month=9, pmin=980.0), crls)
    rates = compute_rates(sel, crls, k_size=200.0, dir_kernel=30.0, day_kernel=15.0,
                          start_year=2000, end_year=2000, min_dp=8.0,
                          dp_low=28.0, dp_med=48.0)
    b = rates["all"]
    # Daily SRR spans the full year and integrates (sums) to the annual rate.
    assert b["srr_daily"].shape == (1, 365)
    assert b["srr_daily"][0].sum() == pytest.approx(b["srr"][0], rel=1e-6)
    # The seasonal peak sits on the storm's closest-approach day-of-year (Sep 2 -> 245).
    assert int(np.argmax(b["srr_daily"][0])) + 1 == 245
    # Daily mass tracks the Med bin (dp=33); Low/High daily curves are flat zero.
    assert rates["med"]["srr_daily"][0].sum() == pytest.approx(b["srr"][0], rel=1e-6)
    assert rates["low"]["srr_daily"][0].sum() == pytest.approx(0.0)


def test_srr_daily_table(tmp_path):
    from storm_climatology_analysis import writer
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(month=9, pmin=980.0), crls)
    rates = compute_rates(sel, crls, k_size=200.0, dir_kernel=30.0, day_kernel=15.0,
                          start_year=2000, end_year=2000, min_dp=8.0,
                          dp_low=28.0, dp_med=48.0)
    p = writer.write_srr_daily_table(rates, crls, tmp_path / "srr_daily_atlantic.csv")
    d = pd.read_csv(p)
    # Long form: 365 rows for the single CRL, days 1..365.
    assert len(d) == 365
    assert d["doy"].tolist() == list(range(1, 366))
    assert (d["crl_id"] == 1).all()
    # The daily column sums to the annual SRR.
    assert d["srr_daily_all"].sum() == pytest.approx(rates["all"]["srr"][0], rel=1e-6)


def test_select_storms_drops_high_pressure():
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    # pmin above max_cp -> all fixes dropped -> nothing selected.
    assert select_storms(_one_storm_hurdat(pmin=1010.0), crls, max_cp=1005.0).empty


def test_compute_rates_monthly_additivity():
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(month=9, pmin=980.0), crls)
    rates = compute_rates(sel, crls, k_size=200.0, dir_kernel=30.0, day_kernel=15.0,
                          start_year=2000, end_year=2000, min_dp=8.0,
                          dp_low=28.0, dp_med=48.0)
    assert rates["_meta"]["nyrs"] == 1
    b = rates["all"]
    # The twelve monthly omnidirectional rates sum to the annual rate.
    assert b["srr_monthly"][0].sum() == pytest.approx(b["srr"][0])
    # All the mass is in September.
    assert b["srr_monthly"][0, 8] == pytest.approx(b["srr"][0])
    assert b["srr_monthly"][0, :8].sum() == pytest.approx(0.0)
    # Monthly directional rate sums to the annual directional rate.
    assert np.allclose(b["dsrr_rate_monthly"][0].sum(axis=0), b["dsrr_rate"][0])
    # dp=33 -> Med bin carries it, Low/High are empty.
    assert rates["med"]["srr"][0] == pytest.approx(b["srr"][0])
    assert rates["low"]["srr"][0] == pytest.approx(0.0)
    assert rates["high"]["srr"][0] == pytest.approx(0.0)


def test_created_date_from_ahd_name():
    from storm_climatology_analysis.hurdat_source import created_date
    # The NHC file date is parsed from the AHD filename; the output tag's start/end
    # years come from the rate period, not the source filename.
    assert created_date("augmented_hurdat2_atlantic_1851-2025_20260227.csv") == "20260227"
    assert created_date("augmented_hurdat2_pacific_1949-2025_20260227.csv") == "20260227"


def test_srr_radius_table(tmp_path):
    from storm_climatology_analysis import writer
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(_one_storm_hurdat(month=9, pmin=980.0), crls)
    rates = compute_rates(sel, crls, k_size=200.0, dir_kernel=30.0, day_kernel=15.0,
                          start_year=2000, end_year=2000, min_dp=8.0,
                          dp_low=28.0, dp_med=48.0)
    p = writer.write_srr_radius_table(rates, crls, 200.0, tmp_path / "srr200km_atlantic.csv")
    d = pd.read_csv(p)
    # SRR_200km = SRR * (2 * 200) = SRR * 400, units TC/yr.
    assert d["srr200km_all"].iloc[0] == pytest.approx(rates["all"]["srr"][0] * 400.0)
    # Monthly columns present and still sum to the annual value.
    mcols = [f"srr200km_all_{m}" for m in MONTHS]
    assert d[mcols].sum(axis=1).iloc[0] == pytest.approx(d["srr200km_all"].iloc[0])
    assert d["srr200km_all_Sep"].iloc[0] == pytest.approx(d["srr200km_all"].iloc[0])


def test_srr_units_value():
    # Single storm sitting exactly on the CRL, Nyrs=1: SRR = kernel peak value.
    crls = pd.DataFrame({"id": [1], "lat": [25.0], "lon": [-90.0]})
    sel = select_storms(
        _one_storm_hurdat(lat=(25.0, 25.0), lon=(-90.0, -90.0)), crls)
    rates = compute_rates(sel, crls, k_size=200.0, dir_kernel=30.0, day_kernel=15.0,
                          start_year=2000, end_year=2000, min_dp=8.0,
                          dp_low=28.0, dp_med=48.0)
    assert rates["all"]["srr"][0] == pytest.approx(
        1.0 / (np.sqrt(2 * np.pi) * 200.0), rel=1e-6)


def test_storm_type_dispatch_and_placeholder():
    from storm_climatology_analysis.orchestrator import SCAOrchestrator
    assert SCAConfig().storm_type == "tc"           # default is the implemented mode
    with pytest.raises(Exception):                  # invalid rejected by the validator
        SCAConfig(storm_type="bogus")
    with pytest.raises(NotImplementedError):        # etc placeholder raises on run
        SCAOrchestrator(SCAConfig(storm_type="etc")).run()
