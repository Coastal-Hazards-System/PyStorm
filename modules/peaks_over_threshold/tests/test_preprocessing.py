"""test_preprocessing — tests for the NTR-pipeline engines.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates the pure engines that build the POT input:
  (1) detrend recovers a known linear trend and preserves NaN gaps;
  (2) NTR removes the tide so only the residual remains;
  (3) the stage list canonicalizes/dedups in execution order.

The download stage is network-bound and not exercised here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))

from peaks_over_threshold import (
    PreprocessConfig, detrend_time_series, estimate_ntr,
)
from peaks_over_threshold.preprocessing import fill_missing_time_steps

SECONDS_PER_YEAR = 365.2425 * 24 * 3600.0


def _epoch_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    """Resolution-independent epoch seconds for a DatetimeIndex."""
    return idx.to_numpy("datetime64[s]").astype("int64").astype(float)


def _synthetic_wl(slope_per_year=0.003, noise=0.05, seed=0):
    """Hourly, MSL-referenced (~0 mean) water level: linear trend + tide + noise."""
    rng = np.random.default_rng(seed)
    t   = pd.date_range("2000-01-01", "2003-01-01", freq="h")
    sec = _epoch_seconds(t)
    trend = slope_per_year / SECONDS_PER_YEAR * (sec - sec.mean())
    tide  = 0.5 * np.sin(2 * np.pi * sec / 44712) + 0.2 * np.sin(2 * np.pi * sec / 86400)
    wl    = trend + tide + rng.normal(0, noise, len(t))
    return pd.DataFrame({"datetime": t, "value": wl})


# ---------------------------------------------------------------------------
# Detrend
# ---------------------------------------------------------------------------
def test_detrend_recovers_known_slope():
    wl = _synthetic_wl(slope_per_year=0.003)
    det_o, trend_o, slope_o = detrend_time_series(wl, method="ordinary")
    _, _, slope_m = detrend_time_series(wl, method="midpoint", ntde_range=(2001, 2001))

    assert abs(slope_o - 0.003) < 5e-4          # ordinary = unbiased OLS
    assert abs(slope_m - 0.003) < 2e-3          # midpoint = through NTDE midpoint
    assert len(det_o) == len(wl) and len(trend_o) == len(wl)


def test_detrend_preserves_nan_gaps():
    wl = _synthetic_wl()
    wl.loc[100:200, "value"] = np.nan
    det, _, _ = detrend_time_series(wl, method="ordinary")
    assert det["value"].isna().sum() >= 101     # the masked block stays NaN


def test_detrend_rejects_too_few_points():
    one = pd.DataFrame({"datetime": pd.to_datetime(["2000-01-01"]), "value": [1.0]})
    try:
        detrend_time_series(one)
    except ValueError:
        return
    raise AssertionError("expected ValueError for <2 valid samples")


# ---------------------------------------------------------------------------
# NTR
# ---------------------------------------------------------------------------
def test_estimate_ntr_removes_tide():
    wl = _synthetic_wl(slope_per_year=0.0, noise=0.05)
    det, _, _ = detrend_time_series(wl, method="ordinary")

    # 6-minute tide identical in form to the embedded tide.
    tt   = pd.date_range("2000-01-01", "2003-01-01", freq="6min")
    tsec = _epoch_seconds(tt)
    tide = 0.5 * np.sin(2 * np.pi * tsec / 44712) + 0.2 * np.sin(2 * np.pi * tsec / 86400)
    tide_df = pd.DataFrame({"datetime": tt, "value": tide})

    ntr = estimate_ntr(det, tide_df)
    assert list(ntr.columns) == ["datetime", "wl", "tide", "ntr"]
    # Tide variance (~0.38 std) removed; only the ~0.05 noise should remain.
    assert np.nanstd(ntr["ntr"]) < 0.1


def test_fill_missing_time_steps_completes_grid():
    t  = pd.date_range("2000-01-01", "2000-01-05", freq="h")
    df = pd.DataFrame({"datetime": t, "value": np.arange(len(t), dtype=float)})
    sparse = df.drop(index=range(10, 30)).reset_index(drop=True)   # punch a gap
    filled = fill_missing_time_steps(sparse, freq="h")
    assert len(filled) == len(t)
    assert filled["value"].isna().sum() == 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_preprocess_config_canonicalizes_stages(tmp_path):
    cfg = PreprocessConfig(
        stages=["pot", "download", "ntr", "download"],   # unordered + dup
        station_id="8518750",
        raw_dir=tmp_path, processed_dir=tmp_path, plots_dir=tmp_path,
    )
    assert cfg.stages == ["download", "ntr", "pot"]      # canonical order, deduped
