"""Tests for the scope (local vs regional) dispatch in main_reduced_storm_suite.run."""

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
rss_main = import_module("main_reduced_storm_suite")


def test_run_rejects_bad_scope():
    with pytest.raises(ValueError, match="scope"):
        rss_main.run(config={"output_dir": "x"}, mode="fixed", scope="nope")


def test_run_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        rss_main.run(config={"output_dir": "x"}, mode="nope", scope="local")


def test_regional_scope_forces_bbox_to_none(monkeypatch, tmp_path):
    """When scope=regional, bbox_config is ignored even if provided."""
    mod = rss_main

    seen = {"bbox_called": False, "workflow_called_with_cfg": None}

    def fake_apply_bbox(cfg, bbox_config):
        seen["bbox_called"] = True

    def fake_workflow(cfg):
        seen["workflow_called_with_cfg"] = dict(cfg)
        return ("ok", {})

    monkeypatch.setattr(mod, "_apply_bbox", fake_apply_bbox)
    # patch the lazy import target in the dispatch branch
    import reduced_storm_suite.workflows.rtcs_selection as rtcs
    monkeypatch.setattr(rtcs, "run_rtcs_selection", fake_workflow)

    bogus_bbox = {"bbox": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1}}
    mod.run(config={"output_dir": str(tmp_path)},
            bbox_config=bogus_bbox, mode="fixed", scope="regional")

    assert seen["bbox_called"] is False, "regional scope must skip the bbox filter"
    out = seen["workflow_called_with_cfg"]["output_dir"]
    assert out.endswith(str(Path("regional") / "fixed")), \
        f"regional output_dir not routed correctly: {out}"


def test_local_scope_invokes_bbox_when_given(monkeypatch, tmp_path):
    mod = rss_main

    seen = {"bbox_called": False}

    def fake_apply_bbox(cfg, bbox_config):
        seen["bbox_called"] = True

    def fake_workflow(cfg):
        return ("ok", {})

    monkeypatch.setattr(mod, "_apply_bbox", fake_apply_bbox)
    import reduced_storm_suite.workflows.rtcs_selection as rtcs
    monkeypatch.setattr(rtcs, "run_rtcs_selection", fake_workflow)

    bbox = {"bbox": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1}}
    mod.run(config={"output_dir": str(tmp_path)},
            bbox_config=bbox, mode="fixed", scope="local")

    assert seen["bbox_called"] is True, "local scope with bbox_config must invoke filter"


def test_local_scope_without_bbox_skips_filter(monkeypatch, tmp_path):
    """Backwards-compat: scope='local' with no bbox_config still works."""
    mod = rss_main

    seen = {"bbox_called": False}

    monkeypatch.setattr(mod, "_apply_bbox",
                        lambda *a, **k: seen.__setitem__("bbox_called", True))
    import reduced_storm_suite.workflows.rtcs_selection as rtcs
    monkeypatch.setattr(rtcs, "run_rtcs_selection", lambda cfg: ("ok", {}))

    mod.run(config={"output_dir": str(tmp_path)},
            bbox_config=None, mode="fixed", scope="local")

    assert seen["bbox_called"] is False
