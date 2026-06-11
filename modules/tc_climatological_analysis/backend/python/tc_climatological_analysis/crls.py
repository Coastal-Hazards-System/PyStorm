"""CRL (Coastal Reference Location) loading.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A CRL set is a CSV of coastal nodes (``ID,lat,lon``) along which the storm
recurrence rate is evaluated. This is the CHS ``CHS_Atl_CRLs_v1.6.csv`` format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _match(cols, *prefixes) -> str | None:
    """First column whose lower-cased name starts with any of ``prefixes``."""
    for c in cols:
        cl = c.lower().strip()
        if any(cl.startswith(p) for p in prefixes):
            return c
    return None


def load_crls(path) -> pd.DataFrame:
    """Load a CRL file into a tidy frame with columns ``id``, ``lat``, ``lon``.

    Handles both the Atlantic CSV (``ID,lat,lon``) and the Pacific tab-delimited
    text (``Latitude  Longitude  Region  ID``): the delimiter is auto-detected and
    columns are matched by name prefix (lat*/lon*/id*). A ``region`` column is
    carried through when present; ``id`` defaults to a 1-based index if absent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CRL file not found: {path}")
    # sep=None + the python engine sniffs comma vs tab vs whitespace.
    df = pd.read_csv(path, sep=None, engine="python")

    lat_c = _match(df.columns, "lat")
    lon_c = _match(df.columns, "lon")
    if lat_c is None or lon_c is None:
        raise ValueError(
            f"CRL file {path.name} must have latitude/longitude columns; got "
            f"{list(df.columns)}.")
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_c], errors="coerce").to_numpy(float),
        "lon": pd.to_numeric(df[lon_c], errors="coerce").to_numpy(float),
    })
    id_c = _match(df.columns, "id")
    out.insert(0, "id", df[id_c].astype(int).to_numpy() if id_c
               else range(1, len(out) + 1))
    region_c = _match(df.columns, "region")
    if region_c is not None:
        out["region"] = df[region_c].to_numpy()
    # Drop rows with an unparseable lat/lon.
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
