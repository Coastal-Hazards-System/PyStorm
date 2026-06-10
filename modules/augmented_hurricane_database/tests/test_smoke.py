"""Smoke tests for the augmented_hurricane_database module."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from augmented_hurricane_database.config import AHDConfig
from augmented_hurricane_database.parser import HURDAT2, COLUMNS
from augmented_hurricane_database import sources
from augmented_hurricane_database import ebtrk as ebtrk_mod
from augmented_hurricane_database.orchestrator import (
    AHDOrchestrator, _nhc_created, _clean_stem,
)


# A two-fix synthetic storm moving due north 1° of latitude (~111 km) over 6 h.
# Modern HURDAT2 layout (21 data fields + trailing comma). Wind 50 kt, 34-kt NE
# radius 60 nm, Rmax 15 nm; the SW 34-kt radius is the -999 sentinel.
_HEADER = "AL092021,                TESTSTORM,      2,"
_ROW1 = ("20210826, 1200,  , TS, 20.0N,  80.0W,  50,  990,"
         "   60,   40, -999,   50,    0,    0,    0,    0,    0,    0,    0,    0,   15,")
_ROW2 = ("20210826, 1800,  , HU, 21.0N,  80.0W,  70,  980,"
         "   80,   60, -999,   70,   30,   20,   10,   20,    0,    0,    0,    0,   12,")


@pytest.fixture()
def atlantic_file(tmp_path):
    p = tmp_path / "hurdat2-1851-2021-040822.txt"
    p.write_text("\n".join([_HEADER, _ROW1, _ROW2]) + "\n", encoding="utf-8")
    return p


def test_parse_schema_and_units(atlantic_file):
    df = HURDAT2(atlantic_file, basin="atlantic").to_dataframe()

    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert df["basin"].unique().tolist() == ["atlantic"]
    assert df["nhc_id"].iloc[0] == "AL092021"
    assert df["year"].iloc[0] == 2021

    # knots -> km/h: 50 kt * 1.852 = 92.6 -> 93
    assert df["vmax_kmh"].iloc[0] == 93
    # pressure unchanged
    assert df["pmin_hpa"].iloc[0] == 990
    # nautical miles -> km: 60 nm * 1.852 = 111.12 -> 111
    assert df["radii34_ne_km"].iloc[0] == round(60 * 1.852)
    # -999 sentinel -> NaN
    assert math.isnan(df["radii34_sw_km"].iloc[0])
    # Rmax converted: 15 nm * 1.852 -> 28
    assert df["rmax_km"].iloc[0] == round(15 * 1.852)


def test_motion_north(atlantic_file):
    df = HURDAT2(atlantic_file, basin="atlantic").to_dataframe()

    # Due-north motion -> heading ~0°, both fixes filled (first forward-filled).
    assert df["heading_deg"].notna().all()
    assert df["trans_kmh"].notna().all()
    assert abs(df["heading_deg"].iloc[1]) <= 2
    # ~111 km over 6 h ~= 18-19 km/h.
    assert 15 <= df["trans_kmh"].iloc[1] <= 22
    # First fix forward-filled from the second.
    assert df["trans_kmh"].iloc[0] == df["trans_kmh"].iloc[1]


def test_basin_inferred_from_id(atlantic_file, tmp_path):
    # No basin passed -> inferred per storm from the id prefix (AL -> atlantic).
    df = HURDAT2(atlantic_file).to_dataframe()
    assert df["basin"].unique().tolist() == ["atlantic"]


def test_config_basin_expansion():
    assert AHDConfig(basins="both").basins == ["atlantic", "pacific"]
    assert AHDConfig(basins="atlantic").basins == ["atlantic"]
    assert AHDConfig(basins=["pacific", "atlantic"]).basins == ["atlantic", "pacific"]
    with pytest.raises(ValueError):
        AHDConfig(basins="indian")


_FAKE_LISTING = """
<a href="hurdat2-1851-2023-051124.txt">x</a>
<a href="hurdat2-1851-2024-040425.txt">x</a>
<a href="hurdat2-1851-2025-02272026.txt">x</a>
<a href="hurdat2-atl-02052024.txt">x</a>
<a href="hurdat2-nepac-1949-2024-031725.txt">x</a>
<a href="hurdat2-nepac-1949-2025-02272026.txt">x</a>
"""


def test_discover_latest_offline():
    # End-year dominates the ranking; the odd "atl"-without-year file is ignored.
    assert sources.discover_latest("atlantic", html=_FAKE_LISTING) == \
        "hurdat2-1851-2025-02272026.txt"
    assert sources.discover_latest("pacific", html=_FAKE_LISTING) == \
        "hurdat2-nepac-1949-2025-02272026.txt"


_CIRA_BASE = ("https://rammb2.cira.colostate.edu/wp-content/uploads/2020/11/")

# A CIRA-style listing: per code, an older new-format file, the newest new-format
# file, and an "old format" file that must be excluded from discovery.
_EBTRK_LISTING = "\n".join(
    f'<a href="{_CIRA_BASE}{name}">x</a>' for name in [
        "EBTRK_AL_final_1851-2019_new_format_01-Jan-2020.txt",
        "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt",   # newest AL
        "EBTRK_AL_final_1851-2021_old_format_02-Sep-2022.txt",     # excluded
        "EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt",     # newest EP
        "EBTRK_CP_final_1950-2020_new_format_15-Aug-2021.txt",
        "EBTRK_CP_final_1950-2021_new_format_02-Sep-2022-1.txt",   # newest CP
    ]
)


def test_discover_latest_ebtrk_offline():
    # Newest end-year wins; old-format files are excluded entirely.
    assert ebtrk_mod.discover_latest_ebtrk("AL", html=_EBTRK_LISTING) == \
        _CIRA_BASE + "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt"
    assert ebtrk_mod.discover_latest_ebtrk("EP", html=_EBTRK_LISTING) == \
        _CIRA_BASE + "EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt"
    assert ebtrk_mod.discover_latest_ebtrk("CP", html=_EBTRK_LISTING) == \
        _CIRA_BASE + "EBTRK_CP_final_1950-2021_new_format_02-Sep-2022-1.txt"


def test_discover_latest_ebtrk_relative_href():
    # Relative hrefs resolve against the listing page URL.
    html = ('<a href="/wp-content/uploads/2020/11/'
            'EBTRK_AL_final_1851-2024_new_format_10-Oct-2025.txt">x</a>')
    assert ebtrk_mod.discover_latest_ebtrk("AL", html=html) == (
        "https://rammb2.cira.colostate.edu/wp-content/uploads/2020/11/"
        "EBTRK_AL_final_1851-2024_new_format_10-Oct-2025.txt")


def test_resolve_ebtrk_url_override(tmp_path, monkeypatch):
    # A per-code URL override is fetched verbatim; discovery is bypassed.
    captured = []

    def fake_dl(url, dest_dir, *, overwrite=False):
        captured.append(url)
        return tmp_path / url.rsplit("/", 1)[-1]

    monkeypatch.setattr(ebtrk_mod, "download_ebtrk", fake_dl)
    # Guard: discovery must NOT run when an override is supplied.
    monkeypatch.setattr(ebtrk_mod, "discover_latest_ebtrk",
                        lambda *a, **k: pytest.fail("discovery should be bypassed"))

    override = "https://example.test/custom/EBTRK_AL_custom.txt"
    out = ebtrk_mod.resolve_ebtrk_sources(
        "atlantic", download=True, input_dir=tmp_path,
        code_urls={"AL": override})
    assert captured == [override]
    assert out[0].name == "EBTRK_AL_custom.txt"


def test_nhc_created_stamp():
    assert _nhc_created("02272026") == "20260227"   # MMDDYYYY -> YYYYMMDD
    assert _nhc_created("040822") == "20220408"      # MMDDYY   -> YYYYMMDD
    assert _nhc_created("nope") == ""                # unrecognized


def test_output_stem_augmented_name():
    cfg = AHDConfig()
    orch = AHDOrchestrator(cfg)

    # Atlantic: start year, end year, and NHC creation date all from the name.
    fields = orch._name_fields("atlantic", Path("hurdat2-1851-2025-02272026.txt"))
    assert fields == {"basin": "atlantic", "start_year": "1851",
                      "end_year": "2025", "created": "20260227"}
    assert _clean_stem(cfg.output_stem.format(**fields)) == \
        "augmented_hurdat2_atlantic_1851-2025_20260227"

    # Pacific nepac name carries its own start year (1949).
    pac = orch._name_fields("pacific", Path("hurdat2-nepac-1949-2025-02272026.txt"))
    assert _clean_stem(cfg.output_stem.format(**pac)) == \
        "augmented_hurdat2_pacific_1949-2025_20260227"

    # Unparseable custom name -> clean fallback, no dangling separators.
    odd = orch._name_fields("atlantic", Path("my_custom_file.txt"))
    assert _clean_stem(cfg.output_stem.format(**odd)) == \
        "augmented_hurdat2_atlantic_latest"


def test_plot_targets_master_switch():
    # Master None -> the four per-(basin, target) flags decide.
    c = AHDConfig(plot_atlantic_cp=True, plot_pacific_rmax=True)
    assert c.plot_targets_for("atlantic") == ["cp"]
    assert c.plot_targets_for("pacific") == ["rmax"]
    # Master True -> all four on (both targets, any basin).
    on = AHDConfig(plot_imputation=True)
    assert on.plot_targets_for("atlantic") == ["cp", "rmax"]
    assert on.plot_targets_for("pacific") == ["cp", "rmax"]
    # Master False overrides an individual True -> none.
    off = AHDConfig(plot_imputation=False, plot_atlantic_cp=True)
    assert off.plot_targets_for("atlantic") == []


def test_imputation_plots_written(tmp_path):
    pytest.importorskip("matplotlib")
    from augmented_hurricane_database import plots

    t0 = dt.datetime(1880, 8, 4, 0, tzinfo=dt.timezone.utc)
    times = [t0 + dt.timedelta(hours=6 * i) for i in range(3)]
    df = pd.DataFrame({
        "tc_no":    [1, 1, 1, 2, 2],
        "nhc_id":   ["AL021880"] * 3 + ["AL031880"] * 2,
        "name":     ["UNNAMED"] * 3 + ["TESTSTORM"] * 2,
        "year":     [1880] * 5,
        "time_utc": times + times[:2],
        "pmin_hpa": [990.0, 980.0, np.nan, 1000.0, np.nan],   # some imputed (NaN-free line not required)
        "rmax_km":  [np.nan] * 5,
    })
    obs = df["pmin_hpa"].notna().to_numpy()
    n = plots.plot_basin_imputation(
        df, basin="atlantic", target="cp", obs_mask=obs,
        out_dir=tmp_path, n_jobs=1, verbose=False)
    assert n == 2
    # Filenames carry the NHC basin/year/number, not an internal counter.
    assert (tmp_path / "DataImputation_Cp_HURDAT_atlantic_AL1880_02.png").exists()
    assert (tmp_path / "DataImputation_Cp_HURDAT_atlantic_AL1880_03.png").exists()

    # A target with no finite values anywhere -> nothing to plot.
    n_rmax = plots.plot_basin_imputation(
        df, basin="atlantic", target="rmax", obs_mask=np.zeros(len(df), bool),
        out_dir=tmp_path, n_jobs=1, verbose=False)
    assert n_rmax == 0


def test_imputation_plot_single_fix_no_xlim_warning(tmp_path):
    pytest.importorskip("matplotlib")
    import warnings
    from augmented_hurricane_database import plots

    # A one-fix storm has t.min() == t.max(); xlim must not go singular.
    df = pd.DataFrame({
        "tc_no":   [1],
        "nhc_id":  ["AL012000"],
        "name":    ["SOLO"],
        "year":    [2000],
        "time_utc": [dt.datetime(2000, 9, 1, 0, tzinfo=dt.timezone.utc)],
        "pmin_hpa": [995.0],
        "rmax_km":  [np.nan],
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = plots.plot_basin_imputation(
            df, basin="atlantic", target="cp", obs_mask=np.array([True]),
            out_dir=tmp_path, n_jobs=1, verbose=False)
    assert n == 1
    assert not any("identical low and high xlims" in str(w.message) for w in caught)


def test_find_local_ebtrk_date_aware(tmp_path):
    # Newest end-year/stamp wins among local new-format files.
    for name in ["EBTRK_AL_final_1851-2019_new_format_01-Jan-2020.txt",
                 "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = ebtrk_mod.find_local_ebtrk("AL", tmp_path)
    assert found.name == "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt"


def _ebtrk_line(nhc_id, month, day, hour, year, rmax) -> str:
    """Build a fixed-width EBTRK line at the columns parse_ebtrk reads."""
    s = [" "] * 80
    def put(a, text):
        for i, ch in enumerate(text):
            s[a + i] = ch
    put(0, nhc_id)                 # cols 1-8: cyclone id
    put(21, f"{month:02d}")
    put(23, f"{day:02d}")
    put(25, f"{hour:02d}")
    put(27, f"{year:5d}")          # right-justified 5-wide
    put(54, f"{rmax:4d}")          # Rmax (nm), right-justified 4-wide
    return "".join(s)


def test_ebtrk_parse_and_fill(tmp_path):
    # Two EBTRK rows at the SAME synoptic time but DIFFERENT storms - the per-id
    # join must pick AL071988's 46 nm, not AL081988's 30 nm.
    ebtrk_txt = "\n".join([
        _ebtrk_line("AL071988", 9, 8, 12, 1988, 46),
        _ebtrk_line("AL081988", 9, 8, 12, 1988, 30),
        _ebtrk_line("AL071988", 9, 9, 0, 1988, -99),   # sentinel -> dropped
    ]) + "\n"
    p = tmp_path / "EBTRK_AL_test.txt"
    p.write_text(ebtrk_txt, encoding="utf-8")

    eb = ebtrk_mod.parse_ebtrk(p)
    assert set(eb["nhc_id"]) == {"AL071988", "AL081988"}     # -99 row dropped
    assert eb.loc[eb.nhc_id == "AL071988", "rmax_km"].iloc[0] == round(46 * 1.852)

    df = pd.DataFrame({
        "nhc_id":  ["AL071988", "AL071988", "AL071988"],
        "ymd":     [19880908,   19880908,   19880909],
        "hhmm":    [1200,       1800,       0],
        "rmax_km": [np.nan,     99.0,       np.nan],
    })
    out, n = ebtrk_mod.fill_missing_rmax(df, eb)
    assert n == 1                                            # only row 0 matched
    assert out["rmax_km"].iloc[0] == round(46 * 1.852)      # per-storm, not 30 nm
    assert out["rmax_km"].iloc[1] == 99.0                   # existing untouched
    assert math.isnan(out["rmax_km"].iloc[2])               # no EBTRK match
