"""noaa_download — fetch NOAA water level / tide prediction CSVs.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Downloads a NOAA Tides & Currents product for a station, one calendar year at
a time, and concatenates the valid years into a single CSV in NOAA's original
column layout. Network access is intrinsic to this engine; it owns its own
HTTP session with retry/backoff to tolerate the API's transient 5xx/504s.

Products
--------
  "hourly_height"  observed water level   -> water_level_<station>.csv
  "predictions"    astronomical tide      -> tide_prediction_<station>.csv
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing  import Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

_PRODUCT_BASENAME = {
    "hourly_height": "water_level",
    "predictions":   "tide_prediction",
}


def download_noaa_wl_data(
    station_id:   str,
    years:        Iterable[int],
    product:      str,
    output_dir:   Path,
    datum:        str   = "MSL",
    time_zone:    str   = "GMT",
    units:        str   = "metric",
    interval:     Optional[str] = None,
    *,
    timeout_sec:    float = 60.0,
    max_retries:    int   = 99,
    retry_backoff:  float = 1.5,
    sleep_between:  float = 0.2,
) -> Optional[Path]:
    """Download a NOAA product by year and save one concatenated CSV.

    Parameters
    ----------
    station_id : str
        NOAA station ID (e.g. "8518750").
    years : iterable of int
        Calendar years to request (inclusive). Years with no valid data are
        skipped.
    product : {"hourly_height", "predictions"}
        Observed water level or astronomical tide prediction.
    output_dir : Path
        Destination directory (created if absent). The file is named
        ``water_level_<station>.csv`` or ``tide_prediction_<station>.csv``.
    datum, time_zone, units : str
        Passed through to the NOAA API.
    interval : str or None
        Sampling interval passed to the API. For ``predictions`` use ``"h"`` to
        request HOURLY tide (default 6-minute otherwise). Ignored by
        ``hourly_height``, which is inherently hourly. ``None`` omits the param.

    Returns
    -------
    Path or None
        Path to the saved CSV, or None if no valid data was found.
    """
    product = product.strip().lower()
    if product not in _PRODUCT_BASENAME:
        raise ValueError("product must be 'hourly_height' or 'predictions'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _session_with_retries(max_retries, retry_backoff)

    dfs: List[pd.DataFrame] = []
    for year in sorted({int(y) for y in years}):
        url = _build_noaa_url(product, station_id, year, datum, time_zone, units, interval)
        try:
            resp = session.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), dtype=str, on_bad_lines="skip")
            if not df.empty and _has_datetime_value_pair(df):
                dfs.append(df)
                print(f"  [download] {station_id} {product} {year}: ok")
            else:
                print(f"  [download] {station_id} {product} {year}: no valid data")
        except requests.exceptions.RequestException as e:
            print(f"  [download] {station_id} {product} {year}: error — {e}")
        finally:
            if sleep_between > 0:
                time.sleep(sleep_between)

    if not dfs:
        print(f"  [download] {station_id} {product}: no valid data in any year")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

    out_path = output_dir / f"{_PRODUCT_BASENAME[product]}_{station_id}.csv"
    combined.to_csv(out_path, index=False)
    print(f"  [download] saved: {out_path}")
    return out_path


def _session_with_retries(max_retries: int, backoff_factor: float) -> requests.Session:
    """A requests Session that auto-retries on 429/5xx (incl. 504)."""
    session = requests.Session()
    retry = Retry(
        total=max_retries, connect=max_retries, read=max_retries, status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _build_noaa_url(
    product:   str,
    station_id: str,
    year:      int,
    datum:     str,
    time_zone: str,
    units:     str,
    interval:  Optional[str] = None,
) -> str:
    """Build the NOAA Tides & Currents datagetter URL for one calendar year."""
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"product={product}"
        f"&begin_date={year}0101&end_date={year}1231"
        f"&datum={datum}"
        f"&station={station_id}"
        f"&time_zone={time_zone}"
        f"&units={units}"
        f"&format=csv"
    )
    if interval:
        url += f"&interval={interval}"
    return url


def _has_datetime_value_pair(df: pd.DataFrame) -> bool:
    """True if any adjacent (datetime, numeric) column pair has a valid row."""
    ncol = len(df.columns)
    for i in range(ncol - 1):
        dt  = pd.to_datetime(df.iloc[:, i], errors="coerce")
        val = pd.to_numeric(df.iloc[:, i + 1], errors="coerce")
        if (dt.notna() & val.notna()).any():
            return True
    return False
