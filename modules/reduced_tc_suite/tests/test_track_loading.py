"""Tests for ITCS TROP track loading and Y-row alignment in bbox_filter.

Covers the two storm-number resolution modes of ``load_tc_tracks``:
  * sequential 1..n (contiguous suites: na / la / pr / tx)
  * master-suite storm_ids (non-contiguous SACS subsets: gom / sa)
and the ``_storm_id_to_int`` coercion helper.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


def _write_trop(path: Path, pts):
    """Write a minimal TROP file: a header line then `t,?,lat,lon` rows."""
    lines = ["header to skip"]
    for lat, lon in pts:
        lines.append(f"0,0,{lat},{lon}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# _storm_id_to_int
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (65, 65),
    ("65", 65),
    ("0065", 65),
    ("SACCS_JPM0065_TROP", 65),
    (np.int64(1700), 1700),
    (None, None),
    ("no-digits", None),
])
def test_storm_id_to_int(raw, expected):
    from reduced_tc_suite.geo.bbox_filter import _storm_id_to_int
    assert _storm_id_to_int(raw) == expected


# ---------------------------------------------------------------------------
# load_tc_tracks - sequential fallback (contiguous suites)
# ---------------------------------------------------------------------------

def test_load_tracks_sequential_numbering(tmp_path):
    from reduced_tc_suite.geo.bbox_filter import load_tc_tracks
    pattern = "NACCS_JPM{:04d}_TROP.txt"
    _write_trop(tmp_path / pattern.format(1), [(30.0, -90.0)])
    _write_trop(tmp_path / pattern.format(3), [(31.0, -91.0), (31.5, -91.5)])
    # storm 2 has no file -> empty track

    tracks = load_tc_tracks(tmp_path, n_storms=3, file_pattern=pattern)
    assert len(tracks) == 3
    assert tracks[0].shape == (1, 2)
    assert tracks[1].shape == (0, 2)      # missing file 0002
    assert tracks[2].shape == (2, 2)


# ---------------------------------------------------------------------------
# load_tc_tracks - master-ID mapping (SACS subsets)
# ---------------------------------------------------------------------------

def test_load_tracks_by_master_storm_ids(tmp_path):
    """Y rows map to files by master ID, not by position. Two Y rows whose
    master IDs are 65 and 1700 must read files 0065 and 1700 - even though a
    sequential 1..2 scan would look for 0001 / 0002."""
    from reduced_tc_suite.geo.bbox_filter import load_tc_tracks
    pattern = "SACCS_JPM{:04d}_TROP.txt"
    _write_trop(tmp_path / pattern.format(65),   [(29.0, -89.0)])
    _write_trop(tmp_path / pattern.format(1700), [(28.0, -88.0), (28.5, -88.5)])

    tracks = load_tc_tracks(tmp_path, n_storms=2, file_pattern=pattern,
                            storm_ids=["65", "1700"])
    assert len(tracks) == 2
    assert tracks[0].shape == (1, 2)
    np.testing.assert_allclose(tracks[0][0], [29.0, -89.0])
    assert tracks[1].shape == (2, 2)


def test_load_tracks_unparseable_id_loads_empty(tmp_path):
    from reduced_tc_suite.geo.bbox_filter import load_tc_tracks
    pattern = "SACCS_JPM{:04d}_TROP.txt"
    _write_trop(tmp_path / pattern.format(65), [(29.0, -89.0)])

    tracks = load_tc_tracks(tmp_path, n_storms=2, file_pattern=pattern,
                            storm_ids=["65", "n/a"])
    assert tracks[0].shape == (1, 2)
    assert tracks[1].shape == (0, 2)      # unparseable id -> empty


# ---------------------------------------------------------------------------
# storm_ids_from_track_dir - derive Y-row IDs from filenames
# ---------------------------------------------------------------------------

def test_storm_ids_from_track_dir_subset_ascending(tmp_path):
    """Non-contiguous SACS subset: IDs parsed from filenames, sorted ascending."""
    from reduced_tc_suite.geo.bbox_filter import storm_ids_from_track_dir
    pattern = "SACCS_JPM{:04d}_TROP.txt"
    for sid in (1700, 65, 102):           # deliberately out of order on disk
        _write_trop(tmp_path / pattern.format(sid), [(29.0, -89.0)])
    (tmp_path / "README.txt").write_text("ignore me")   # non-matching file

    assert storm_ids_from_track_dir(tmp_path, pattern) == [65, 102, 1700]


def test_storm_ids_from_track_dir_contiguous(tmp_path):
    from reduced_tc_suite.geo.bbox_filter import storm_ids_from_track_dir
    pattern = "NACCS_JPM{:04d}_TROP.txt"
    for sid in range(1, 6):
        _write_trop(tmp_path / pattern.format(sid), [(30.0, -90.0)])
    assert storm_ids_from_track_dir(tmp_path, pattern) == [1, 2, 3, 4, 5]


def test_storm_ids_from_track_dir_alt_extension(tmp_path):
    """Pattern with a different extension (chs-tx: TC_JPM####.TROP)."""
    from reduced_tc_suite.geo.bbox_filter import storm_ids_from_track_dir
    pattern = "TC_JPM{:04d}.TROP"
    for sid in (3, 1, 2):
        _write_trop(tmp_path / pattern.format(sid), [(28.0, -88.0)])
    assert storm_ids_from_track_dir(tmp_path, pattern) == [1, 2, 3]
