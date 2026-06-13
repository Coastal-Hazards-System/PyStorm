"""writer - output writers for the climatological analysis (SRR / DSRR tables + arrays).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tc_climatological_analysis.gkf import MONTHS, DOYS


def write_selection(selection: pd.DataFrame, path) -> Path:
    """Write the per-CRL selected-TC table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(path, index=False)
    return path


def write_srr_table(rates: dict, crls: pd.DataFrame, path, *,
                    scale: float = 1.0, prefix: str = "srr") -> Path:
    """Wide SRR CSV: per CRL, the annual + monthly omnidirectional rate per bin.

    Columns: crl_id, lat, lon, then for each bin ``<prefix>_<bin>`` and
    ``<prefix>_<bin>_<Mon>``. Monthly columns sum to the annual. ``scale`` (and
    ``prefix``) produce the SRR_<R>km variant: SRR (storms/km/yr) times the 2R-km
    diameter -> storms/year within R of the CRL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "crl_id": crls["id"].to_numpy(),
        "lat": crls["lat"].to_numpy(),
        "lon": crls["lon"].to_numpy(),
    })
    for name in rates["_meta"]["bins"]:
        b = rates[name]
        out[f"{prefix}_{name}"] = b["srr"] * scale
        for j, mon in enumerate(MONTHS):
            out[f"{prefix}_{name}_{mon}"] = b["srr_monthly"][:, j] * scale
    out.to_csv(path, index=False)
    return path


def write_srr_daily_table(rates: dict, crls: pd.DataFrame, path, *,
                          scale: float = 1.0, prefix: str = "srr_daily") -> Path:
    """Continuous daily SRR CSV (long form): one row per CRL x day-of-year.

    Columns: crl_id, lat, lon, doy (1..365), then ``<prefix>_<bin>`` for each
    intensity bin. Values are the kernel-smoothed seasonal rate density in TC/km/yr per
    day-of-year (the annual SRR spread across the calendar); summing a CRL's 365 days
    recovers its annual SRR. ``scale`` (and ``prefix``) produce the SRR_<R>km variant
    (TC/yr within R, per day-of-year).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    crl_id = crls["id"].to_numpy()
    lat = crls["lat"].to_numpy()
    lon = crls["lon"].to_numpy()
    ncrl = len(crl_id)
    ndoy = DOYS.size
    doy_int = DOYS.astype(int)
    cols = {
        "crl_id": np.repeat(crl_id, ndoy),
        "lat": np.repeat(lat, ndoy),
        "lon": np.repeat(lon, ndoy),
        "doy": np.tile(doy_int, ncrl),
    }
    for name in rates["_meta"]["bins"]:
        cols[f"{prefix}_{name}"] = (rates[name]["srr_daily"] * scale).reshape(-1)
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def write_srr_radius_table(rates: dict, crls: pd.DataFrame, radius_km: float,
                           path) -> Path:
    """SRR_<R>km CSV: SRR times the 2R-km diameter (storms/year within R of the CRL)."""
    r = int(round(radius_km))
    return write_srr_table(rates, crls, path, scale=2.0 * radius_km, prefix=f"srr{r}km")


def write_dsrr_summary(rates: dict, crls: pd.DataFrame, path) -> Path:
    """DSRR summary CSV: per CRL the heading mean/stdv (deg) per intensity bin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "crl_id": crls["id"].to_numpy(),
        "lat": crls["lat"].to_numpy(),
        "lon": crls["lon"].to_numpy(),
    })
    for name in rates["_meta"]["bins"]:
        b = rates[name]
        out[f"dsrr_mean_{name}"] = b["mean"]
        out[f"dsrr_stdv_{name}"] = b["stdv"]
    out.to_csv(path, index=False)
    return path


def write_dsrr_arrays(rates: dict, crls: pd.DataFrame, path) -> Path:
    """Full DSRR arrays per bin in a compressed .npz (headings, rate, pdf, cdf,
    monthly rate), plus the SRR arrays and CRL coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = rates["_meta"]
    arrays = {
        "crl_id": crls["id"].to_numpy(),
        "lat": crls["lat"].to_numpy(),
        "lon": crls["lon"].to_numpy(),
        "headings": meta["headings"],
        "months": np.array(meta["months"]),
        "doys": meta["doys"],
        "nyrs": np.array(meta["nyrs"]),
    }
    for name in meta["bins"]:
        b = rates[name]
        arrays[f"srr_{name}"] = b["srr"]
        arrays[f"srr_monthly_{name}"] = b["srr_monthly"]
        arrays[f"srr_daily_{name}"] = b["srr_daily"]
        arrays[f"dsrr_rate_{name}"] = b["dsrr_rate"]
        arrays[f"dsrr_rate_monthly_{name}"] = b["dsrr_rate_monthly"]
        arrays[f"dsrr_pdf_{name}"] = b["pdf"]
        arrays[f"dsrr_cdf_{name}"] = b["cdf"]
        arrays[f"dsrr_mean_{name}"] = b["mean"]
        arrays[f"dsrr_stdv_{name}"] = b["stdv"]
    np.savez_compressed(path, **arrays)
    return path if path.suffix == ".npz" else Path(str(path) + ".npz")
