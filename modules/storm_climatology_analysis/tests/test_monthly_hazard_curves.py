"""test_monthly_hazard_curves - smoke tests for the monthly hazard-curve scaler.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Validates: (1) annual all-bin column detection for srr and srr<R>km tables;
(2) per-month fractions f_m = SRR_all_<Mon> / SRR_all and that they sum to 1;
(3) AER scaling and additivity (the 12 monthly curves sum to the annual at every
magnitude); (4) blank-AER rows dropped and the optional band merged on load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The analysis tools are standalone scripts, not part of the backend package, so put
# the analysis/ dir on the path explicitly (backend/python is already there via conftest).
_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
import monthly_hazard_curves as mhc                      # noqa: E402
from storm_climatology_analysis.gkf import MONTHS        # noqa: E402


def _srr_csv(tmp_path, prefix="srr200km", crl=74):
    """A one-CRL SRR table whose 12 monthly all-bin rates sum to the annual rate."""
    # Concentrate the season in Aug/Sep/Oct (a realistic Atlantic shape).
    weights = {m: 0.0 for m in MONTHS}
    weights.update({"Aug": 0.25, "Sep": 0.40, "Oct": 0.10, "Nov": 0.05, "Jul": 0.20})
    srr_all = 0.6
    row = {"crl_id": crl, "lat": 30.0, "lon": -80.0, f"{prefix}_all": srr_all}
    for m in MONTHS:
        row[f"{prefix}_all_{m}"] = srr_all * weights[m]
    p = tmp_path / f"{prefix}_atlantic.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    return p


def _hc_csv(tmp_path, name="be.csv"):
    """An annual best-estimate curve (AER, BE) with trailing blank-AER padding rows."""
    aer = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
    be = [1.10, 1.22, 1.36, 1.44, 1.55, 1.90]
    df = pd.DataFrame({"AER": aer, "BE": be})
    pad = pd.DataFrame({"AER": [np.nan, np.nan], "BE": [np.nan, np.nan]})
    p = tmp_path / name
    pd.concat([df, pad], ignore_index=True).to_csv(p, index=False)
    return p


def test_detect_all_col_for_both_prefixes():
    assert mhc.detect_all_col(["crl_id", "srr_all", "srr_all_Jan"]) == "srr_all"
    assert mhc.detect_all_col(["crl_id", "srr200km_all", "srr200km_all_Sep"]) == "srr200km_all"
    with pytest.raises(ValueError):
        mhc.detect_all_col(["crl_id", "lat", "lon"])


def test_monthly_fractions_sum_to_one(tmp_path):
    srr = _srr_csv(tmp_path, prefix="srr200km", crl=74)
    fractions, srr_all, all_col = mhc.monthly_fractions(srr, 74)
    assert all_col == "srr200km_all"
    assert srr_all == pytest.approx(0.6)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["Sep"] == pytest.approx(0.40)
    assert fractions["Oct"] == pytest.approx(0.10)
    assert fractions["Jan"] == pytest.approx(0.0)
    with pytest.raises(KeyError):
        mhc.monthly_fractions(srr, 999)


def test_load_drops_blank_aer_and_merges_band(tmp_path):
    hc = _hc_csv(tmp_path)
    df, value_cols = mhc.load_hazard_curve(hc)
    assert value_cols == ["BE"]
    assert df["AER"].notna().all() and len(df) == 6   # the two blank rows dropped


def test_scale_to_monthly_aer_and_additivity(tmp_path):
    srr = _srr_csv(tmp_path, crl=74)
    hc, _ = mhc.load_hazard_curve(_hc_csv(tmp_path))
    fractions, _, _ = mhc.monthly_fractions(srr, 74)
    long_df = mhc.scale_to_monthly(hc, fractions)

    # 13 blocks (Annual + 12 months); Annual AER is unscaled, Sep AER = f_Sep * annual.
    assert long_df["month"].nunique() == 13
    ann = long_df[long_df["month"] == "Annual"].reset_index(drop=True)
    sep = long_df[long_df["month"] == "Sep"].reset_index(drop=True)
    assert np.allclose(ann["AER"], hc["AER"])
    assert np.allclose(sep["AER"], hc["AER"] * fractions["Sep"])
    assert np.allclose(sep["BE"], hc["BE"])            # magnitudes unchanged

    # Additivity: at each magnitude row, the 12 monthly AERs sum to the annual AER.
    monthly = np.vstack([long_df[long_df["month"] == m]["AER"].to_numpy()
                         for m in MONTHS])
    assert np.allclose(monthly.sum(axis=0), hc["AER"].to_numpy())
