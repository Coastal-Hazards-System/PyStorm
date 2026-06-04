"""detrend — linear detrending of a water-level time series.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Engine contract: DataFrame in, DataFrames out. No I/O, no plotting.

Removes a linear sea-level trend from a water-level series by least-squares
regression on time (POSIX seconds). Two centerings are offered:

  "midpoint"  centre time at the midpoint of the National Tidal Datum Epoch
              (NTDE), e.g. 2012–2016 for Louisiana. Trend is reported relative
              to that epoch midpoint, matching NOAA datum conventions.
  "ordinary"  centre time at the mean of the record.

The fit is computed on the non-NaN samples only, then merged back onto the
full input timeline so gaps (NaN values) are preserved for downstream
gap-aware plotting.
"""

from __future__ import annotations

from typing   import Optional, Tuple

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.2425 * 24 * 3600.0


def decimal_year_to_timestamp(year: float) -> pd.Timestamp:
    """Convert a (possibly fractional) calendar year to a Timestamp.

    Whole years map to Jan 1; a fraction advances linearly through that
    calendar year (leap-year aware), so 2012.5 lands at mid-2012.
    """
    y    = int(np.floor(float(year)))
    frac = float(year) - y
    start = pd.Timestamp(y, 1, 1)
    if frac == 0.0:
        return start
    return start + (pd.Timestamp(y + 1, 1, 1) - start) * frac


def ntde_midpoint_timestamp(ntde_range: Tuple[float, float]) -> pd.Timestamp:
    """Midpoint Timestamp of the NTDE, treating the end year as inclusive.

    The epoch spans ``[start, end + 1)`` in (decimal) years — e.g. 1983–2001
    covers [1983, 2002), midpoint mid-1992 — matching NOAA datum conventions.
    """
    start = decimal_year_to_timestamp(ntde_range[0])
    end   = decimal_year_to_timestamp(ntde_range[1] + 1)
    return start + (end - start) / 2


def detrend_time_series(
    df:             pd.DataFrame,
    method:         str = "midpoint",
    ntde_range:     Tuple[float, float] = (2012, 2016),
    slope_per_year: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Detrend a normalized water-level series.

    Parameters
    ----------
    df : DataFrame
        Columns ``("datetime", "value")``. ``datetime`` must be parseable;
        NaN ``value`` rows are allowed and preserved in the output.
    method : {"midpoint", "ordinary"}
        Time-centering strategy (see module docstring).
    ntde_range : (float, float)
        Inclusive NTDE start/end years (may be fractional, e.g. 2012.42). Only
        used when ``method="midpoint"``. The midpoint is taken over
        ``[start, (end + 1))`` in decimal years.
    slope_per_year : float, optional
        If given, use this slope (value-units per year, e.g. +0.0048 m/yr)
        directly instead of fitting one from the record. The trend line still
        pivots about the chosen time centre. ``None`` (default) fits by least
        squares.

    Returns
    -------
    detrended_df : DataFrame  columns ("datetime", "value")
        Water level with the linear trend removed.
    trend_df : DataFrame      columns ("datetime", "value")
        The fitted linear trend, evaluated at each timestamp.
    slope_per_year : float
        Trend slope in value-units per year (e.g. m/yr) — the fitted value, or
        the supplied override.
    """
    method = str(method).lower().strip()
    if method not in ("midpoint", "ordinary"):
        raise ValueError(f"method must be 'midpoint' or 'ordinary', got {method!r}")

    d = df[["datetime", "value"]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["value"]    = pd.to_numeric(d["value"], errors="coerce")

    clean = (d.dropna(subset=["datetime", "value"])
              .sort_values("datetime")
              .reset_index(drop=True))
    if len(clean) < 2:
        raise ValueError(
            f"need at least 2 valid (datetime, value) samples to fit a trend; "
            f"got {len(clean)}"
        )

    # POSIX seconds, resolution-independent (pandas datetime64 may be ns or us).
    t_sec = clean["datetime"].to_numpy("datetime64[s]").astype(np.int64).astype(np.float64)
    y     = clean["value"].to_numpy(dtype=np.float64)

    if method == "midpoint":
        mid_ts = ntde_midpoint_timestamp(ntde_range).timestamp()
        x = t_sec - mid_ts
    else:  # ordinary
        x = t_sec - float(np.mean(t_sec))

    if slope_per_year is not None:
        slope = float(slope_per_year) / SECONDS_PER_YEAR   # override (units/sec)
    else:
        denom = float(np.dot(x, x))
        if denom == 0.0:
            raise ValueError("timestamps are constant — cannot compute a trend")
        slope = float(np.dot(x, y) / denom)   # fitted value-units per second
    trend     = slope * x
    detrended = y - trend

    fit = pd.DataFrame({
        "datetime":  clean["datetime"].values,
        "trend":     trend,
        "detrended": detrended,
    })

    # Merge back onto the full input timeline so NaN gaps are preserved.
    merged = pd.merge(d[["datetime"]], fit, on="datetime", how="left", sort=True)

    detrended_df = merged[["datetime", "detrended"]].rename(columns={"detrended": "value"})
    trend_df     = merged[["datetime", "trend"]].rename(columns={"trend": "value"})

    return detrended_df, trend_df, slope * SECONDS_PER_YEAR


def fill_missing_time_steps(
    df:   pd.DataFrame,
    freq: str = "h",
) -> pd.DataFrame:
    """Reindex a ``("datetime", "value")`` frame onto a complete time grid.

    Gaps become explicit NaN rows at the given frequency, so plots break the
    line across missing periods instead of bridging them.
    """
    d = df[["datetime", "value"]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d = d.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    full = pd.date_range(start=d.index.min(), end=d.index.max(), freq=freq)
    d = d.reindex(full)
    d.index.name = "datetime"
    return d.reset_index()
