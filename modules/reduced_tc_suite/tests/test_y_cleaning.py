"""Tests for the -99999 → NaN conversion in read_store and the
dry-node strategies in reduce_output."""

import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


# ---------------------------------------------------------------------------
# read_store sentinel cleanup
# ---------------------------------------------------------------------------

def _make_store(tmp_path, Y, X=None, HC=None):
    """Write a minimal tc_data.h5 for testing read_store."""
    from reduced_tc_suite.io.store import write_store
    n_storms, m_nodes = Y.shape
    X = X if X is not None else np.random.default_rng(0).standard_normal((n_storms, 3))
    out = tmp_path / "tc_data.h5"
    write_store(path=out, X=X, Y=Y, param_names=["X0", "X1", "X2"], HC=HC)
    return out


def test_read_store_converts_minus_99999_to_nan(tmp_path):
    from reduced_tc_suite.io.store import read_store
    Y = np.array([
        [1.0, -99999.0, 0.5],
        [0.3, -99999.0, 0.7],
        [-99999.0, 0.2, -99999.0],
    ], dtype=np.float64)
    p = _make_store(tmp_path, Y)
    data = read_store(p)
    sentinel_mask_after = data.Y < -9000
    assert not sentinel_mask_after.any(), "sentinels survived read"
    assert int(np.isnan(data.Y).sum()) == 4, (
        f"expected 4 NaN entries, got {int(np.isnan(data.Y).sum())}")


def test_read_store_drops_not_simulated_keeps_dry(tmp_path):
    """An all-NaN row (never simulated = failed HPC run) is dropped, but an
    all-(-99999) row (simulated, everywhere dry = valid storm) is KEPT. The
    failed-storm test must run BEFORE the -99999 -> NaN normalisation."""
    from reduced_tc_suite.io.store import read_store
    Y = np.array([
        [1.0, 0.5],
        [np.nan, np.nan],         # not simulated -> failed -> DROP
        [-99999.0, -99999.0],     # simulated but dry -> valid -> KEEP
        [0.3, 0.7],
    ], dtype=np.float64)
    X = np.arange(12.0).reshape(4, 3)
    p = _make_store(tmp_path, Y, X=X)
    data = read_store(p)
    assert data.X.shape == (3, 3), "only the not-simulated storm should drop"
    assert data.Y.shape == (3, 2)
    # the kept dry storm is now all-NaN (sentinels normalised) but still present
    assert int(np.isnan(data.Y).all(axis=1).sum()) == 1, "dry storm wrongly dropped"


def test_read_store_drops_failed_storm_and_subsets_storm_ids(tmp_path):
    """X, Y, and storm_ids drop in lockstep so survivors keep their true IDs."""
    from reduced_tc_suite.io.store import write_store, read_store
    Y = np.array([
        [1.0, 0.5],
        [np.nan, np.nan],         # storm '102' not simulated -> dropped
        [0.3, 0.7],
    ], dtype=np.float64)
    X = np.arange(9.0).reshape(3, 3)
    out = tmp_path / "tc_data.h5"
    write_store(path=out, X=X, Y=Y, param_names=["a", "b", "c"],
                storm_ids=["65", "102", "1700"])
    data = read_store(out)
    assert data.storm_ids == ["65", "1700"], "storm_ids not subset in lockstep"
    assert data.X.shape == (2, 3)
    assert data.Y.shape == (2, 2)


def test_read_store_passes_through_when_no_sentinels(tmp_path):
    from reduced_tc_suite.io.store import read_store
    Y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p = _make_store(tmp_path, Y)
    data = read_store(p)
    assert not np.isnan(data.Y).any()
    np.testing.assert_allclose(data.Y.astype(np.float32).astype(np.float64), Y)


# ---------------------------------------------------------------------------
# reduce_output dry strategies
# ---------------------------------------------------------------------------

def test_pca_drop_always_dry(capsys):
    """All-NaN columns are dropped; remaining NaN are zero-filled."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((20, 8))
    Y[:, 0] = np.nan              # entire column NaN -> must be dropped
    Y[0, 2] = np.nan              # single NaN -> must be zero-filled
    Y[3, 5] = np.nan

    Y_r, pca = reduce_output(Y, variance_threshold=0.9,
                             dry_strategy="drop_always_dry")
    out = capsys.readouterr().out
    assert "dropped 1" in out, f"expected drop message; got: {out!r}"
    assert "kept" in out
    # 7 kept columns (1 dropped) so PCA fits on (20, 7)
    assert pca.n_features_in_ == 7


def test_pca_zero_strategy_does_not_drop():
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((15, 6))
    Y[:, 0] = np.nan
    Y[5, 3] = np.nan
    _, pca = reduce_output(Y, variance_threshold=0.9, dry_strategy="zero")
    assert pca.n_features_in_ == 6, "zero strategy must preserve all columns"


def test_pca_wet_only_drops_partial_nan_columns():
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((10, 5))
    Y[0, 1] = np.nan      # column 1 has a NaN -> drop
    Y[:, 3] = np.nan      # column 3 fully NaN -> drop
    _, pca = reduce_output(Y, variance_threshold=0.9, dry_strategy="wet_only")
    assert pca.n_features_in_ == 3


def test_pca_node_mean_imputes_with_per_node_mean():
    from reduced_tc_suite.sampling.pca import reduce_output
    Y = np.array([
        [1.0, 2.0, np.nan],
        [3.0, np.nan, 4.0],
        [5.0, 6.0, 8.0],
    ])
    _, pca = reduce_output(Y, variance_threshold=0.99, dry_strategy="node_mean")
    assert pca.n_features_in_ == 3


def test_pca_wet_ratio_floor_drops_below_threshold(capsys):
    """Nodes wet for < min_wet_fraction of storms are dropped; the rest are
    zero-filled. With 10 storms and a 0.5 floor, a node must be wet for >= 5
    storms to survive."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((10, 4))
    Y[:8, 0] = np.nan     # col 0 wet for 2/10 = 0.2  -> drop  (< 0.5)
    Y[:6, 1] = np.nan     # col 1 wet for 4/10 = 0.4  -> drop  (< 0.5)
    Y[:4, 2] = np.nan     # col 2 wet for 6/10 = 0.6  -> keep, zero-fill 4 NaN
    # col 3 fully wet -> keep
    _, pca = reduce_output(Y, variance_threshold=0.99,
                           dry_strategy="wet_ratio_floor",
                           min_wet_fraction=0.5)
    out = capsys.readouterr().out
    assert pca.n_features_in_ == 2, "expected 2 nodes to clear the 0.5 floor"
    assert "wet_ratio_floor" in out and "dropped 2" in out, f"got: {out!r}"


def test_pca_wet_ratio_floor_small_floor_keeps_partially_wet():
    """A small positive floor drops only fully-dry nodes while retaining
    nodes wet for even a single storm - matching drop_always_dry node count."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((10, 5))
    Y[:, 0] = np.nan      # fully dry -> wet fraction 0.0, dropped (< 0.05)
    Y[:9, 1] = np.nan     # wet for 1/10 = 0.1 -> kept (>= 0.05)
    _, pca = reduce_output(Y, variance_threshold=0.99,
                           dry_strategy="wet_ratio_floor",
                           min_wet_fraction=0.05)
    assert pca.n_features_in_ == 4, "small floor keeps nodes wet for >=1 storm"


def test_pca_raises_clear_error_when_strategy_drops_all_nodes():
    """wet_only on a matrix where every node has a NaN drops everything; the
    error must name the strategy and suggest alternatives, not surface the
    opaque sklearn '0 feature(s)' message."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((10, 4))
    for j in range(4):
        Y[j % 10, j] = np.nan      # every column has at least one NaN
    with pytest.raises(ValueError, match="removed every node"):
        reduce_output(Y, 0.95, dry_strategy="wet_only")


def test_pca_wet_ratio_floor_too_high_raises_with_knob_hint():
    """An over-aggressive min_wet_fraction (>max wet fraction) drops all nodes;
    the message should point at pca_min_wet_fraction."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((10, 3))
    Y[0, :] = np.nan               # each node wet for 9/10 = 0.9 at most
    with pytest.raises(ValueError, match="pca_min_wet_fraction"):
        reduce_output(Y, 0.95, dry_strategy="wet_ratio_floor",
                      min_wet_fraction=0.95)


def test_pca_no_nan_path_unchanged():
    """When Y is fully finite, all strategies are identical."""
    from reduced_tc_suite.sampling.pca import reduce_output
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((30, 12))
    r_default, _ = reduce_output(Y, 0.95)
    r_zero, _    = reduce_output(Y, 0.95, dry_strategy="zero")
    r_wet, _     = reduce_output(Y, 0.95, dry_strategy="wet_only")
    np.testing.assert_allclose(r_default, r_zero, atol=1e-12)
    np.testing.assert_allclose(r_default, r_wet,  atol=1e-12)
