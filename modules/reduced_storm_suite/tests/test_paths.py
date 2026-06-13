"""test_paths - reduced_storm_suite.io.paths.resolve_one_file resolution.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) single-match resolution; (2) suffix-variant resolution; (3) missing file raises; (4) missing folder raises; (5) multiple matches raise.
"""

import sys
from pathlib import Path

import pytest

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


def test_resolves_single_match(tmp_path):
    from reduced_storm_suite.io.paths import resolve_one_file
    (tmp_path / "CHS-LA_nodeID.mat").write_bytes(b"")
    p = resolve_one_file(tmp_path, "CHS-LA_nodeID*.mat")
    assert p.name == "CHS-LA_nodeID.mat"


def test_resolves_suffix_variant(tmp_path):
    """A _probQ-suffixed file is still found by the same prefix glob."""
    from reduced_storm_suite.io.paths import resolve_one_file
    (tmp_path / "CHS-LA_nodeID_probQ.mat").write_bytes(b"")
    p = resolve_one_file(tmp_path, "CHS-LA_nodeID*.mat")
    assert p.name == "CHS-LA_nodeID_probQ.mat"


def test_missing_file_raises(tmp_path):
    from reduced_storm_suite.io.paths import resolve_one_file
    with pytest.raises(FileNotFoundError, match="No file matching"):
        resolve_one_file(tmp_path, "CHS-LA_nodeID*.mat", label="bbox coords")


def test_missing_folder_raises(tmp_path):
    from reduced_storm_suite.io.paths import resolve_one_file
    with pytest.raises(FileNotFoundError, match="Folder does not exist"):
        resolve_one_file(tmp_path / "nope", "*.mat")


def test_multiple_matches_raises(tmp_path):
    """Ambiguity (e.g. both bare and _probQ on disk) must error, not pick one silently."""
    from reduced_storm_suite.io.paths import resolve_one_file
    (tmp_path / "CHS-LA_nodeID.mat").write_bytes(b"")
    (tmp_path / "CHS-LA_nodeID_probQ.mat").write_bytes(b"")
    with pytest.raises(ValueError, match="matched 2 files"):
        resolve_one_file(tmp_path, "CHS-LA_nodeID*.mat")
