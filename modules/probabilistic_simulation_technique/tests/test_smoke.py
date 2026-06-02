"""test_smoke — smoke tests for the PST module (config, bootstrap, threshold).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config construction + invariants, (2) BootstrapGenerator
contract (C++ when available, pure-Python fallback otherwise), (3) GPD
threshold search runs and returns a value inside its candidate band,
(4) PSTOrchestrator end-to-end on a synthetic POT sample.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the in-tree package importable when running tests without `pip install -e`.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))

from probabilistic_simulation_technique import (
    PSTConfig, PSTOrchestrator, CPP_KERNEL_AVAILABLE,
)
from probabilistic_simulation_technique.sampling import (
    BootstrapGenerator, select_gpd_threshold_qdm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_pot():
    rng = np.random.default_rng(628)
    return np.sort(rng.gamma(shape=2.0, scale=1.5, size=300))[::-1]


@pytest.fixture
def pot_csv(tmp_path, synthetic_pot):
    path = tmp_path / "synthetic_POT.csv"
    pd.DataFrame({"value": np.sort(synthetic_pot)}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_rejects_bad_truncation():
    with pytest.raises(Exception):
        PSTConfig(
            input_csv           = Path("a.csv"),
            output_dir          = Path("out"),
            plots_dir           = Path("plots"),
            record_length_years = 100,
            bootstrap           = {"distribution": "gaussian", "truncation": (1.0, -1.0)},
        )


def test_config_rejects_bad_percentile_band():
    with pytest.raises(Exception):
        PSTConfig(
            input_csv                 = Path("a.csv"),
            output_dir                = Path("out"),
            plots_dir                 = Path("plots"),
            record_length_years       = 100,
            threshold_min_percentile  = 80,
            threshold_max_percentile  = 20,
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_shape_and_order(synthetic_pot):
    gen = BootstrapGenerator(seed=1)
    mat = gen.generate(synthetic_pot, n_simulations=20)
    assert mat.shape == (synthetic_pot.size, 20)
    # Each column is descending-sorted.
    assert np.all(np.diff(mat, axis=0) <= 0)


def test_bootstrap_python_fallback_matches_shape(synthetic_pot):
    gen = BootstrapGenerator(seed=1, use_cpp=False)
    mat = gen.generate(synthetic_pot, n_simulations=10)
    assert mat.shape == (synthetic_pot.size, 10)
    assert np.all(np.diff(mat, axis=0) <= 0)


def test_bootstrap_rejects_ascending():
    gen = BootstrapGenerator(seed=1)
    asc = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gen.generate(asc, n_simulations=5)


# ---------------------------------------------------------------------------
# GPD threshold
# ---------------------------------------------------------------------------
def test_select_threshold_in_band(synthetic_pot):
    lambda_val  = synthetic_pot.size / 100.0
    weibull_aef = (np.arange(1, synthetic_pot.size + 1) / (synthetic_pot.size + 1)) * lambda_val
    th, wmse, candidates = select_gpd_threshold_qdm(
        synthetic_pot, weibull_aef, lambda_val,
        min_percentile=20, max_percentile=80, n_candidates=50,
    )
    assert candidates.min() <= th <= candidates.max()
    assert wmse.shape == candidates.shape


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------
def test_orchestrator_end_to_end(tmp_path, pot_csv):
    out  = tmp_path / "out"
    plot = tmp_path / "plots"
    cfg  = PSTConfig(
        input_csv           = pot_csv,
        output_dir          = out,
        plots_dir           = plot,
        record_length_years = 100,
        num_simulations     = 50,
        random_seed         = 42,
    )
    result = PSTOrchestrator(cfg).run()

    # Output files exist.
    base = pot_csv.stem.rsplit("_", 1)[0] if "_" in pot_csv.stem else pot_csv.stem
    assert (out  / f"{base}_PST.csv").is_file()
    assert (out  / f"{base}_PST_HC_BE_tbl.csv").is_file()
    assert (out  / f"{base}_PST_HC_CB_tbl.csv").is_file()
    assert (out  / f"{base}_PST_HC_BE_plt.csv").is_file()
    assert (out  / f"{base}_PST_HC_CB_plt.csv").is_file()
    assert (plot / f"{base}_PST_HC.png").is_file()

    # Result invariants.
    assert result.ensemble.shape[0] == 50
    assert result.aef_table.shape   == (22,)
    assert result.used_cpp_kernel == CPP_KERNEL_AVAILABLE
