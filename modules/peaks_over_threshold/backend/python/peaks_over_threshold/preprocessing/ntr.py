"""ntr - non-tidal residual (NTR) estimation.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Engine contract: DataFrames in, DataFrame out. No I/O, no plotting.

The non-tidal residual is the observed (detrended) water level minus the
predicted astronomical tide:

    NTR(t) = WL_detrended(t) - tide(t)

Both series are expected on the same HOURLY grid (water level from
``hourly_height``; tide from ``predictions`` requested with ``interval=h``).
The tide is sampled onto the water-level timestamps with a time-weighted
``interpolate`` + ``reindex``: on matched hourly stamps this is an exact no-op,
and it only does real work as a safety net when a tide hour is missing (then it
bridges the gap) or stamps are slightly offset. Water-level gaps (NaN) are
preserved so the NTR series carries the same coverage as the input record.
"""

from __future__ import annotations

import pandas as pd


def estimate_ntr(
    wl_df:   pd.DataFrame,
    tide_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the non-tidal residual from detrended WL and tide predictions.

    Parameters
    ----------
    wl_df : DataFrame
        Detrended water level; columns ``("datetime", "value")``.
    tide_df : DataFrame
        Astronomical tide prediction; columns ``("datetime", "value")``.

    Returns
    -------
    DataFrame
        Columns ``("datetime", "wl", "tide", "ntr")`` aligned to the water
        level timestamps. ``ntr`` is the non-tidal residual; ``wl`` and
        ``tide`` are retained for diagnostics. NaN WL rows propagate to NaN
        NTR rows.
    """
    wl = wl_df[["datetime", "value"]].copy()
    td = tide_df[["datetime", "value"]].copy()

    for frame in (wl, td):
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["value"]    = pd.to_numeric(frame["value"], errors="coerce")

    wl = (wl.dropna(subset=["datetime"]).sort_values("datetime")
            .set_index("datetime"))
    td = (td.dropna(subset=["datetime"]).sort_values("datetime")
            .set_index("datetime"))

    # Interpolate tide in time, then sample at the water-level timestamps.
    tide_at_wl = td["value"].interpolate(method="time").reindex(wl.index)

    out = pd.DataFrame(index=wl.index)
    out["wl"]   = wl["value"]
    out["tide"] = tide_at_wl
    out["ntr"]  = out["wl"] - out["tide"]
    out.index.name = "datetime"
    return out.reset_index()
