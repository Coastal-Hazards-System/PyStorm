"""srr_source - read the SCA SRR tables and build the per-CRL driving rates.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Loads the storm_climatology_analysis outputs for one CRL and turns them into the
three drivers the Monte-Carlo needs:

  * the annual omnidirectional SRR (TC/km/yr) per stratum -> the Poisson rate and
    the intensity split,
  * a per-stratum day-of-year probability mass over days 1..365 -> when in the
    season each synthetic TC lands.

The day-of-year mass comes either from the smooth daily SRR table (day_method=
"daily") or, as a fallback, from the monthly SRR spread uniformly within each
calendar month (day_method="monthly"). No simulation logic lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from life_cycle_simulation.calendar365 import (
    DAYS_IN_MONTH, MONTH_START_DOY, MONTHS, NDOY,
)

# Strata used for the Poisson split and the day-of-year shape. "all" is the union
# (its rate sets lambda); low/med/high partition it.
STRATA = ("low", "med", "high")
_BINS = ("all",) + STRATA


@dataclass
class CRLSrr:
    """Driving SRR for one CRL: annual rates per stratum and the day-of-year mass."""
    crl_id: int
    lat: float
    lon: float
    annual: Dict[str, float]          # {"all","low","med","high"} TC/km/yr
    doy_pmf: np.ndarray               # [3, 365] per-stratum (low/med/high) day-of-year pmf
    daily_used: bool                  # True if the smooth daily table backed doy_pmf


# ---------------------------------------------------------------------------
# SRR table loading
# ---------------------------------------------------------------------------

def _detect_prefix(columns) -> str:
    """Infer the SRR column prefix (e.g. ``srr`` or ``srr200km``) from the header.

    The annual columns are exactly ``<prefix>_all``/``_low``/``_med``/``_high``;
    monthly columns add a ``_<Mon>`` suffix. The unique column ending in ``_all``
    (a month never spells "all") fixes the prefix.
    """
    for c in columns:
        if c.endswith("_all"):
            return c[: -len("_all")]
    raise ValueError(
        "input_csv does not look like an SCA SRR table: no '<prefix>_all' column "
        "found. Expected srr_<basin>_<v>.csv with columns srr_all, srr_low, ...")


def load_srr_table(input_csv) -> "pd.DataFrame":
    """Read the annual + monthly SRR table (TC/km/yr), indexed by ``crl_id``."""
    path = Path(input_csv)
    if not path.is_file():
        raise FileNotFoundError(f"SRR input not found: {path}")
    df = pd.read_csv(path)
    if "crl_id" not in df.columns:
        raise ValueError(f"{path.name} has no 'crl_id' column.")
    return df.set_index("crl_id", drop=False)


def locate_daily_companion(input_csv) -> Optional[Path]:
    """Find srr_daily_<basin>_<v>.csv sitting next to a srr_<basin>_<v>.csv input."""
    path = Path(input_csv)
    # Insert "daily" right after the leading "srr" token: srr_atlantic_... ->
    # srr_daily_atlantic_...; tolerant of an "srr<R>km" prefix too.
    cand_name = re.sub(r"^(srr[^_]*)_", r"\1_daily_", path.name, count=1)
    cand = path.with_name(cand_name)
    return cand if cand.is_file() else None


def locate_selection_companion(input_csv) -> Optional[Path]:
    """Find selection_<basin>_<v>.csv sitting next to a srr_<basin>_<v>.csv input."""
    path = Path(input_csv)
    cand = path.with_name(re.sub(r"^srr[^_]*_", "selection_", path.name, count=1))
    return cand if cand.is_file() else None


def load_selection_table(path, crl_ids=None) -> "pd.DataFrame":
    """Load the per-CRL selected-TC table (crl_id, year, dist) for calibration.

    Reads only the columns the correlation calibration needs; optionally keeps just
    the requested CRLs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SCA selection table not found: {path}")
    df = pd.read_csv(path, usecols=["crl_id", "year", "dist"])
    if crl_ids is not None:
        df = df[df["crl_id"].isin(set(int(c) for c in crl_ids))]
    return df


def load_daily_table(daily_csv, crl_ids) -> "pd.DataFrame":
    """Read the long-form daily SRR table, keeping only the requested CRLs.

    Reads just the columns the day-of-year model needs (crl_id, doy, and the three
    stratum daily rates) so the multi-hundred-MB all-CRL file stays cheap to scan.
    """
    path = Path(daily_csv)
    if not path.is_file():
        raise FileNotFoundError(f"daily SRR table not found: {path}")
    head = pd.read_csv(path, nrows=0)
    prefix = "srr_daily"
    needed = ["crl_id", "doy"] + [f"{prefix}_{s}" for s in STRATA]
    missing = [c for c in needed if c not in head.columns]
    if missing:
        raise ValueError(f"{path.name} missing daily columns {missing}.")
    want = set(int(c) for c in crl_ids)
    df = pd.read_csv(path, usecols=needed)
    return df[df["crl_id"].isin(want)]


# ---------------------------------------------------------------------------
# Per-CRL driver assembly
# ---------------------------------------------------------------------------

def _annual_rates(row, prefix) -> Dict[str, float]:
    """Annual SRR (TC/km/yr) per bin for one CRL row, clamped to be non-negative."""
    return {b: max(0.0, float(row[f"{prefix}_{b}"])) for b in _BINS}


def _monthly_doy_pmf(row, prefix) -> np.ndarray:
    """Per-stratum day-of-year pmf [3,365] from the monthly SRR (uniform within month).

    Each month's stratum SRR is spread evenly over that month's days, giving a
    piecewise-constant seasonal shape. Rows that are all zero are left as zeros
    (that stratum is never drawn because its annual rate is zero).
    """
    pmf = np.zeros((len(STRATA), NDOY), dtype=float)
    for s, name in enumerate(STRATA):
        monthly = np.array([float(row[f"{prefix}_{name}_{m}"]) for m in MONTHS])
        monthly = np.clip(monthly, 0.0, None)
        for mi in range(12):
            start = MONTH_START_DOY[mi] - 1                 # 0-based slice start
            ndays = DAYS_IN_MONTH[mi]
            pmf[s, start:start + ndays] = monthly[mi] / ndays
        total = pmf[s].sum()
        if total > 0:
            pmf[s] /= total
    return pmf


def _daily_doy_pmf(daily_df, crl_id) -> np.ndarray:
    """Per-stratum day-of-year pmf [3,365] from the smooth daily SRR table."""
    sub = daily_df[daily_df["crl_id"] == crl_id].sort_values("doy")
    if len(sub) != NDOY:
        raise ValueError(
            f"daily SRR for CRL {crl_id} has {len(sub)} rows, expected {NDOY}.")
    pmf = np.zeros((len(STRATA), NDOY), dtype=float)
    for s, name in enumerate(STRATA):
        curve = np.clip(sub[f"srr_daily_{name}"].to_numpy(float), 0.0, None)
        total = curve.sum()
        if total > 0:
            pmf[s] = curve / total
    return pmf


def build_crl_srr(srr_df, daily_df, crl_id, *, day_method: str) -> CRLSrr:
    """Assemble the driving SRR (annual rates + day-of-year pmf) for one CRL."""
    if crl_id not in srr_df.index:
        raise KeyError(
            f"CRL id {crl_id} not in the SRR table (ids "
            f"{int(srr_df['crl_id'].min())}..{int(srr_df['crl_id'].max())}).")
    row = srr_df.loc[crl_id]
    prefix = _detect_prefix(srr_df.columns)
    annual = _annual_rates(row, prefix)

    if day_method == "daily" and daily_df is not None:
        pmf = _daily_doy_pmf(daily_df, crl_id)
        daily_used = True
    else:
        pmf = _monthly_doy_pmf(row, prefix)
        daily_used = False

    # A stratum with annual rate but an all-zero day pmf (e.g. monthly columns all
    # zero from rounding) falls back to the "all" seasonal shape so its TCs still
    # land somewhere sensible rather than failing the day draw.
    fallback = pmf.sum(axis=0)
    if fallback.sum() > 0:
        fallback = fallback / fallback.sum()
        for s, name in enumerate(STRATA):
            if annual[name] > 0 and pmf[s].sum() == 0:
                pmf[s] = fallback

    return CRLSrr(crl_id=int(crl_id), lat=float(row["lat"]), lon=float(row["lon"]),
                  annual=annual, doy_pmf=pmf, daily_used=daily_used)
