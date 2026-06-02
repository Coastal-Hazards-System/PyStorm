"""test_smoke — smoke tests for the POT module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) config construction; (2) segmenters produce sane output on a
synthetic record; (3) the threshold search converges on a synthetic series
crafted to hit ~target events/year; (4) end-to-end POTOrchestrator run.
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

from peaks_over_threshold import (
    POTConfig, POTOrchestrator, CPP_KERNEL_AVAILABLE,
)
from peaks_over_threshold.sampling     import IterativeThresholdSearch
from peaks_over_threshold.segmentation import segment_hydrograph, segment_peak_gap


# ---------------------------------------------------------------------------
# Synthetic record (hourly samples over 1 year, embedded "storm" pulses)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_series():
    rng       = np.random.default_rng(628)
    n_hours   = 24 * 365
    times_sec = np.arange(n_hours, dtype=np.float64) * 3600.0
    base      = 0.5 * rng.standard_normal(n_hours)
    # Inject 12 well-separated storm pulses (≈ 1 / month).
    pulse_centers = np.linspace(72, n_hours - 72, 12).astype(int)
    for c in pulse_centers:
        base[c] += 5.0 + rng.uniform(0, 1)
    return times_sec, base


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_normalizes_legacy_method_name(tmp_path):
    cfg = POTConfig(
        input_csv  = tmp_path / "a.csv",
        output_dir = tmp_path / "out",
        plots_dir  = tmp_path / "plots",
        method     = "peaks",
    )
    assert cfg.method == "peak_gap"


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------
def test_hydrograph_groups_by_gap():
    # Two storm pulses with values [5, 4] and [6], separated by > interevent.
    values    = np.array([0, 0, 5, 4, 0, 0, 0, 6], dtype=np.float64)
    times_sec = np.arange(values.size, dtype=np.float64) * 3600.0
    exceed    = np.flatnonzero(values > 1.0)
    peaks     = segment_hydrograph(values, times_sec, exceed, interevent_sec=2 * 3600.0)
    assert peaks.tolist() == [2, 7]   # argmax of each group


def test_peak_gap_sequential_drop():
    values    = np.array([5, 4, 0, 6, 5, 0, 7], dtype=np.float64)
    times_sec = np.arange(values.size, dtype=np.float64) * 3600.0
    exceed    = np.flatnonzero(values > 1.0)
    peaks     = segment_peak_gap(values, times_sec, exceed, interevent_sec=2 * 3600.0)
    # Indices 0, 3, 6 survive; the decreasing neighbours within the gap drop out.
    assert peaks.tolist() == [0, 3, 6]


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def test_threshold_search_converges(synthetic_series):
    times_sec, values = synthetic_series
    searcher = IterativeThresholdSearch(
        interevent_sec         = 48 * 3600.0,
        method                 = "hydrograph",
        target_events_per_year = 10.0,
        tolerance              = 0.5,
        start_percentile       = 75.0,
        step_size              = 0.05,
    )
    result = searcher.run(values, times_sec)
    assert result.converged
    assert abs(result.events_per_year - 10.0) < 0.5
    # Peak indices must lie within the input range.
    assert result.peak_indices.min() >= 0
    assert result.peak_indices.max() < values.size


def test_threshold_search_python_fallback_matches_shape(synthetic_series):
    times_sec, values = synthetic_series
    searcher = IterativeThresholdSearch(
        interevent_sec         = 48 * 3600.0,
        method                 = "hydrograph",
        target_events_per_year = 10.0,
        tolerance              = 0.5,
        use_cpp                = False,
    )
    result = searcher.run(values, times_sec)
    assert result.peak_indices.ndim == 1
    assert result.used_cpp_kernel is False


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------
def test_orchestrator_end_to_end(tmp_path, synthetic_series):
    times_sec, values = synthetic_series
    timestamps = pd.to_datetime(times_sec, unit="s")
    df = pd.DataFrame({"Date Time": timestamps, "Storm Surge": values})

    input_csv = tmp_path / "synth.csv"
    df.to_csv(input_csv, index=False)

    cfg = POTConfig(
        input_csv              = input_csv,
        output_dir             = tmp_path / "out",
        plots_dir              = tmp_path / "plots",
        target_events_per_year = 10.0,
        tolerance              = 0.5,
        step_size              = 0.05,
    )
    result = POTOrchestrator(cfg).run()

    assert (tmp_path / "out"   / f"{input_csv.stem}_POT.csv").is_file()
    assert (tmp_path / "plots" / f"{input_csv.stem}_POT.png").is_file()
    assert result.used_cpp_kernel == CPP_KERNEL_AVAILABLE
