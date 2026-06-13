"""test_bootstrap - the tc_data.h5 auto-bootstrap in main_reduced_tc_suite._ensure_h5_exists.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) an existing h5 skips bootstrap; (2) a missing h5 invokes the Preprocessor; (3) a Preprocessor that produces no output raises. The Preprocessor is patched, so no real input files are needed.
"""

import sys
from importlib import import_module
from pathlib import Path

import pytest

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))

# Orchestrator entry lives in backend/python (added to sys.path above); resolve
# it dynamically so there is no static import for the IDE to flag as unresolved.
rtcs_main = import_module("main_reduced_tc_suite")


class _FakePreprocessor:
    """Captures construction args and lets tests choose what .run() does."""
    instances = []
    run_action = None    # callable(instance) -> None ; tests set this

    def __init__(self, cfg):
        self.cfg = cfg
        _FakePreprocessor.instances.append(self)

    def run(self):
        if _FakePreprocessor.run_action is not None:
            _FakePreprocessor.run_action(self)


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakePreprocessor.instances = []
    _FakePreprocessor.run_action = None


def _install_fake_preprocessor(monkeypatch):
    """Insert _FakePreprocessor where the launcher's lazy import looks."""
    import reduced_tc_suite.workflows.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "Preprocessor", _FakePreprocessor)


_DATASET = "ds"
_RAW_FILES = {
    "x_param_table":       "X.mat",
    "y_surge":             "Y.mat",
    "nodeID":              "N.mat",
    "hc_benchmark":        "HC.mat",
    "pre_selected_storms": None,
}


def _stage_raw(root: Path):
    """Create empty dummy raw source files so _build_preprocess_config can
    resolve them (the faked Preprocessor never reads their contents)."""
    raw_dir = root / "data" / "inputs" / "raw" / _DATASET
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in _RAW_FILES.values():
        if fname:
            (raw_dir / fname).write_bytes(b"")


def _ensure(root, h5):
    rtcs_main._ensure_h5_exists(
        root=root, dataset=_DATASET, raw_files=_RAW_FILES,
        preprocess_metadata={}, track_file_patterns={}, h5_path=str(h5))


def test_existing_h5_skips_bootstrap(tmp_path, monkeypatch):
    """If the h5 already exists, the Preprocessor must not be constructed."""
    _install_fake_preprocessor(monkeypatch)
    h5 = tmp_path / "tc_data.h5"
    h5.write_bytes(b"")

    _ensure(tmp_path, h5)
    assert _FakePreprocessor.instances == []


def test_missing_h5_invokes_preprocessor(tmp_path, monkeypatch):
    """If the h5 is missing, Preprocessor.run() is invoked; success path."""
    _install_fake_preprocessor(monkeypatch)
    _stage_raw(tmp_path)
    h5 = tmp_path / "tc_data.h5"
    _FakePreprocessor.run_action = lambda inst: h5.write_bytes(b"")

    _ensure(tmp_path, h5)
    assert len(_FakePreprocessor.instances) == 1, "Preprocessor not invoked"
    assert h5.is_file()


def test_preprocessor_succeeds_but_no_output_raises(tmp_path, monkeypatch):
    """Defensive: run() returns but the h5 is still absent → SystemExit."""
    _install_fake_preprocessor(monkeypatch)
    _stage_raw(tmp_path)
    h5 = tmp_path / "tc_data.h5"
    _FakePreprocessor.run_action = lambda inst: None     # no-op

    with pytest.raises(SystemExit, match="still missing"):
        _ensure(tmp_path, h5)
