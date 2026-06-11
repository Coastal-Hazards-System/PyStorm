"""Output writers for the climatological analysis (SRR / DSRR tables + arrays).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tc_climatological_analysis.gkf import MONTHS


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
        "nyrs": np.array(meta["nyrs"]),
    }
    for name in meta["bins"]:
        b = rates[name]
        arrays[f"srr_{name}"] = b["srr"]
        arrays[f"srr_monthly_{name}"] = b["srr_monthly"]
        arrays[f"dsrr_rate_{name}"] = b["dsrr_rate"]
        arrays[f"dsrr_rate_monthly_{name}"] = b["dsrr_rate_monthly"]
        arrays[f"dsrr_pdf_{name}"] = b["pdf"]
        arrays[f"dsrr_cdf_{name}"] = b["cdf"]
        arrays[f"dsrr_mean_{name}"] = b["mean"]
        arrays[f"dsrr_stdv_{name}"] = b["stdv"]
    np.savez_compressed(path, **arrays)
    return path if path.suffix == ".npz" else Path(str(path) + ".npz")
