"""Numerical regression tests for the DSW pipeline.

Pins compute_global_dsw / reconstruct_hc_global_dsw / evaluate_hc_metrics to
their current outputs on a seeded synthetic case, so the interp1d→np.interp
swap (optimisation B) and any future vectorisation can be validated to keep
numerics identical within float-precision tolerance.
"""

import sys
from pathlib import Path

import numpy as np

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


def _make_case(seed: int = 7, k: int = 30, m: int = 200, n_aer: int = 10):
    """Build a synthetic Y_sub / HC_bench pair that exercises wet & dry nodes
    and monotone-ish benchmark HCs."""
    rng = np.random.default_rng(seed)
    Y_sub = rng.gamma(shape=2.0, scale=1.0, size=(k, m))
    Y_sub[:, :20] = 0.0
    dry_mask = rng.random((k, m)) < 0.05
    Y_sub[dry_mask] = 0.0
    nan_mask = rng.random((k, m)) < 0.02
    Y_sub[nan_mask] = np.nan

    tbl_aer = 1.0 / np.array(
        [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100][:n_aer], dtype=float
    )
    base = np.linspace(5.0, 0.5, n_aer)
    HC_bench = base[None, :] + rng.normal(0.0, 0.1, size=(m, n_aer))
    HC_bench = np.maximum.accumulate(HC_bench[:, ::-1], axis=1)[:, ::-1]
    return Y_sub, HC_bench, tbl_aer


def test_compute_global_dsw_methods_pinned():
    from reduced_tc_suite.weights.dsw import compute_global_dsw

    Y_sub, HC_bench, tbl_aer = _make_case()

    for method in (1, 2, 3):
        dsw = compute_global_dsw(
            Y_sub, HC_bench, tbl_aer,
            dry_thr=0.0, min_wet_storms=2, method=method,
        )
        assert dsw.shape == (Y_sub.shape[0],)
        finite = dsw[np.isfinite(dsw)]
        assert finite.size > 0, f"method {method} returned all-NaN DSW"
        assert (finite >= 0).all(), f"method {method} produced negative DSW"


def _snapshot_dir() -> Path:
    return Path(__file__).resolve().parent / "_snapshots"


def _snapshot_path(name: str) -> Path:
    return _snapshot_dir() / f"{name}.npz"


def _save_or_compare(name: str, arrays: dict, rtol: float = 1e-9, atol: float = 1e-9):
    """Pin numerics: first run creates the snapshot; subsequent runs compare.

    Snapshots live under tests/_snapshots/ and are committed alongside tests.
    Delete a snapshot to re-pin after an *intentional* algorithm change.
    """
    p = _snapshot_path(name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arrays)
        return  # first run = create baseline

    ref = np.load(p)
    for k, v in arrays.items():
        np.testing.assert_allclose(
            v, ref[k], rtol=rtol, atol=atol,
            err_msg=f"snapshot mismatch for '{name}::{k}'",
            equal_nan=True,
        )


def test_dsw_pipeline_snapshot():
    """Pin compute_global_dsw + reconstruct_hc_global_dsw outputs."""
    from reduced_tc_suite.weights.dsw import (
        compute_global_dsw, reconstruct_hc_global_dsw,
    )

    Y_sub, HC_bench, tbl_aer = _make_case()

    out = {}
    for method in (1, 2, 3):
        dsw = compute_global_dsw(
            Y_sub, HC_bench, tbl_aer,
            dry_thr=0.0, min_wet_storms=2, method=method,
        )
        hc = reconstruct_hc_global_dsw(Y_sub, dsw, tbl_aer, dry_thr=0.0)
        out[f"dsw_m{method}"] = dsw
        out[f"hc_m{method}"] = hc

    _save_or_compare("dsw_pipeline", out)


def test_evaluate_hc_metrics_snapshot():
    from reduced_tc_suite.weights.dsw import evaluate_hc_metrics

    Y_sub, HC_bench, tbl_aer = _make_case()

    rows = []
    for method in (1, 2, 3):
        m = evaluate_hc_metrics(
            Y_sub, HC_bench, tbl_aer,
            dry_thr=0.0, min_wet_storms=2, dsw_method=method,
        )
        rows.append([m["mean_bias"], m["mean_uncertainty"], m["mean_rmse"]])

    _save_or_compare("hc_metrics", {"rows": np.array(rows)})
