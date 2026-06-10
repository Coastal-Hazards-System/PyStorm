"""Smoke tests for the augmented_hurricane_database module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from augmented_hurricane_database.config import AHDConfig
from augmented_hurricane_database.parser import HURDAT2, COLUMNS
from augmented_hurricane_database import sources
from augmented_hurricane_database import ebtrk as ebtrk_mod


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
