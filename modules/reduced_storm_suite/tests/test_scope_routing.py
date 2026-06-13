"""test_scope_routing - the run(config) API and the scope (local vs regional) dispatch in api_reduced_storm_suite.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) run() rejects a bad scope; (2) run() rejects a bad mode; (3) run() rejects a bad storm_type; (4) run() raises NotImplementedError for the etc placeholder; (5) run() returns a single RSSResult for one dataset and a dict for a batch; (6) regional scope forces bbox to None; (7) local scope invokes the bbox filter when given; (8) local scope without a bbox skips the filter.
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
rss_main = import_module("api_reduced_storm_suite")


# ── run(config) control validation (no data needed; checks happen first) ─────

def test_run_rejects_bad_scope():
    with pytest.raises(ValueError, match="scope"):
        rss_main.run({"scope": "nope", "mode": "fixed"})


def test_run_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        rss_main.run({"mode": "nope", "scope": "local"})


def test_run_rejects_bad_storm_type():
    with pytest.raises(ValueError, match="storm_type"):
        rss_main.run({"storm_type": "bogus"})


def test_etc_storm_type_is_placeholder():
    with pytest.raises(NotImplementedError):
        rss_main.run({"storm_type": "etc"})


# ── run(config) batch fan-out and return shape ───────────────────────────────

def test_run_returns_single_result_for_one_dataset(monkeypatch):
    """One dataset -> a single RSSResult (not a dict)."""
    monkeypatch.setattr(rss_main, "_resolve_dataset", lambda ds, *a: ({}, {}))
    monkeypatch.setattr(
        rss_main, "_launch_one",
        lambda root, ds, *a, **k: rss_main.RSSResult(ds, "fixed", "regional",
                                                      "out", ("ok", {})))
    res = rss_main.run({
        "root": ".", "dataset": "d1", "mode": "fixed", "scope": "regional",
        "raw_files_by_dataset": {}, "units_by_dataset": {}})
    assert isinstance(res, rss_main.RSSResult)
    assert res.dataset == "d1"


def test_run_returns_dict_for_batch(monkeypatch):
    """Multiple datasets -> a {dataset: RSSResult} mapping."""
    monkeypatch.setattr(rss_main, "_resolve_dataset", lambda ds, *a: ({}, {}))
    monkeypatch.setattr(
        rss_main, "_launch_one",
        lambda root, ds, *a, **k: rss_main.RSSResult(ds, "fixed", "regional",
                                                     "out", None))
    res = rss_main.run({
        "root": ".", "datasets": ["d1", "d2"], "mode": "fixed",
        "scope": "regional", "raw_files_by_dataset": {}, "units_by_dataset": {}})
    assert set(res) == {"d1", "d2"}
    assert all(isinstance(v, rss_main.RSSResult) for v in res.values())


# ── per-dataset scope dispatch (_run_selection, the internal step) ───────────

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
    import reduced_storm_suite.workflows.rss_selection as rss
    monkeypatch.setattr(rss, "run_rss_selection", fake_workflow)

    bogus_bbox = {"bbox": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1}}
    mod._run_selection(config={"output_dir": str(tmp_path)},
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
    import reduced_storm_suite.workflows.rss_selection as rss
    monkeypatch.setattr(rss, "run_rss_selection", fake_workflow)

    bbox = {"bbox": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1}}
    mod._run_selection(config={"output_dir": str(tmp_path)},
                       bbox_config=bbox, mode="fixed", scope="local")

    assert seen["bbox_called"] is True, "local scope with bbox_config must invoke filter"


def test_local_scope_without_bbox_skips_filter(monkeypatch, tmp_path):
    """Backwards-compat: scope='local' with no bbox_config still works."""
    mod = rss_main
    seen = {"bbox_called": False}

    monkeypatch.setattr(mod, "_apply_bbox",
                        lambda *a, **k: seen.__setitem__("bbox_called", True))
    import reduced_storm_suite.workflows.rss_selection as rss
    monkeypatch.setattr(rss, "run_rss_selection", lambda cfg: ("ok", {}))

    mod._run_selection(config={"output_dir": str(tmp_path)},
                       bbox_config=None, mode="fixed", scope="local")

    assert seen["bbox_called"] is False
