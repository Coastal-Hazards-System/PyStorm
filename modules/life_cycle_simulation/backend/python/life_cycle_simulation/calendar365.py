"""calendar365 - the fixed 365-day (non-leap) climatological calendar.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The driving SRR is defined on day-of-year 1..365 (CF "noleap"/"365_day"), the same
calendar the storm_climatology_analysis daily SRR uses. This module owns the month
boundaries and the day-of-year <-> (month, day) maps so the simulator and the SRR
loader agree on one calendar. No domain logic; pure lookup tables.
"""

from __future__ import annotations

import numpy as np

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Days per calendar month on the non-leap year (Feb = 28); sums to 365.
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)

# Day-of-year of the 1st of each month (1-based): Jan 1 -> 1, Feb 1 -> 32, ...
MONTH_START_DOY = np.concatenate([[1], 1 + np.cumsum(DAYS_IN_MONTH)[:-1]]).astype(int)

NDOY = 365
DOYS = np.arange(1, NDOY + 1, dtype=int)

# doy (1..365) -> calendar month (1..12) and day-of-month (1..31). Precomputed once.
_MONTH_OF_DOY = np.repeat(np.arange(1, 13), DAYS_IN_MONTH)          # [365]
_DAY_OF_DOY = np.concatenate([np.arange(1, d + 1) for d in DAYS_IN_MONTH])  # [365]


def doy_to_month(doy: np.ndarray) -> np.ndarray:
    """Calendar month (1..12) for each day-of-year (1..365)."""
    return _MONTH_OF_DOY[np.asarray(doy, dtype=int) - 1]


def doy_to_day(doy: np.ndarray) -> np.ndarray:
    """Day-of-month (1..31) for each day-of-year (1..365)."""
    return _DAY_OF_DOY[np.asarray(doy, dtype=int) - 1]
