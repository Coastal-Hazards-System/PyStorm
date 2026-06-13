"""test_storm_id_ingest - preprocess-time storm-ID resolution (Preprocessor._resolve_storm_ids).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) keep X-source IDs; (2) derive a subset from track filenames; (3) contiguous yields 1..n; (4) raise on file-count mismatch; (5) tolerate a missing or empty track folder; (6) store round-trip and the validate_store /storm_ids length assertion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


def _trop(dirpath: Path, pattern: str, sid: int):
    (dirpath / pattern.format(sid)).write_text("header\n200007,090000,18.8,-63.8\n")


def _preproc(cfg):
    from reduced_tc_suite.workflows.ingest import Preprocessor
    return Preprocessor(cfg)


# ---------------------------------------------------------------------------
# _resolve_storm_ids
# ---------------------------------------------------------------------------

def test_resolve_keeps_existing_ids(tmp_path):
    """IDs already present from the X source are returned untouched."""
    pp = _preproc({"storm_id_track_dir": str(tmp_path),
                   "storm_id_track_pattern": "SACCS_JPM{:04d}_TROP.txt"})
    existing = ["a", "b", "c"]
    assert pp._resolve_storm_ids(existing, 3) is existing


def test_resolve_derives_subset_from_filenames(tmp_path):
    """Non-contiguous SACS subset: derive ascending master IDs as strings."""
    pat = "SACCS_JPM{:04d}_TROP.txt"
    for sid in (1700, 65, 102):
        _trop(tmp_path, pat, sid)
    pp = _preproc({"storm_id_track_dir": str(tmp_path),
                   "storm_id_track_pattern": pat})
    assert pp._resolve_storm_ids([], n_storms=3) == ["65", "102", "1700"]


def test_resolve_contiguous_yields_1_to_n(tmp_path):
    pat = "NACCS_JPM{:04d}_TROP.txt"
    for sid in range(1, 5):
        _trop(tmp_path, pat, sid)
    pp = _preproc({"storm_id_track_dir": str(tmp_path),
                   "storm_id_track_pattern": pat})
    assert pp._resolve_storm_ids(None, n_storms=4) == ["1", "2", "3", "4"]


def test_resolve_raises_on_count_mismatch(tmp_path):
    """Files present but count != n_storms must raise, not silently misalign."""
    pat = "SACCS_JPM{:04d}_TROP.txt"
    for sid in (65, 102):
        _trop(tmp_path, pat, sid)
    pp = _preproc({"storm_id_track_dir": str(tmp_path),
                   "storm_id_track_pattern": pat})
    with pytest.raises(ValueError, match="must hold exactly one file per storm"):
        pp._resolve_storm_ids([], n_storms=5)


def test_resolve_tolerates_missing_dir():
    pp = _preproc({"storm_id_track_dir": "/no/such/folder",
                   "storm_id_track_pattern": "SACCS_JPM{:04d}_TROP.txt"})
    assert pp._resolve_storm_ids([], n_storms=3) == []


def test_resolve_tolerates_empty_dir(tmp_path):
    pp = _preproc({"storm_id_track_dir": str(tmp_path),
                   "storm_id_track_pattern": "SACCS_JPM{:04d}_TROP.txt"})
    assert pp._resolve_storm_ids([], n_storms=3) == []


def test_resolve_noop_without_track_dir():
    pp = _preproc({})
    assert pp._resolve_storm_ids([], n_storms=3) == []


# ---------------------------------------------------------------------------
# validate_store /storm_ids length check + read round-trip
# ---------------------------------------------------------------------------

def test_store_roundtrip_with_storm_ids(tmp_path):
    from reduced_tc_suite.io.store import write_store, read_store, validate_store
    X = np.arange(12.0).reshape(4, 3)
    Y = np.arange(8.0).reshape(4, 2)
    out = tmp_path / "tc_data.h5"
    write_store(path=out, X=X, Y=Y, param_names=["p0", "p1", "p2"],
                storm_ids=["65", "102", "1099", "1700"])
    validate_store(out)                       # must pass the length assertion
    data = read_store(out)
    assert data.storm_ids == ["65", "102", "1099", "1700"]


def test_validate_store_rejects_misaligned_storm_ids(tmp_path):
    """A /storm_ids length != n_storms must fail validation."""
    import h5py
    from reduced_tc_suite.io.store import write_store, validate_store
    X = np.arange(12.0).reshape(4, 3)
    Y = np.arange(8.0).reshape(4, 2)
    out = tmp_path / "tc_data.h5"
    write_store(path=out, X=X, Y=Y, param_names=["p0", "p1", "p2"],
                storm_ids=["65", "102", "1700"])   # 3 IDs but 4 storms
    with pytest.raises(AssertionError, match="storm_ids"):
        validate_store(out)
