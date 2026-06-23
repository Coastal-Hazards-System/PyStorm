"""writer - output writers for the joint distribution model (marginals + copula).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Three products: the tidy marginal-distribution parameters (CSV), the meta-Gaussian
copula tau/rho matrices per intensity (compressed NPZ), and the adjusted, intensity-
labeled per-storm parameters (CSV, the input to the copula and the plots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from joint_distribution_model.config import INTENSITY_BINS, PARAM_NAMES


def write_marginals(records: List[dict], path) -> Path:
    """Write the tidy marginal-parameter table (one row per CRL x intensity x param)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["crl_id", "lat", "lon", "intensity", "param", "dist",
            "p1", "p2", "trunc_lo", "trunc_hi", "n"]
    pd.DataFrame(records, columns=cols).to_csv(path, index=False)
    return path


def write_copula(copula: Dict[str, dict], crl_ids, lat, lon, path) -> Path:
    """Write the per-CRL meta-Gaussian copula matrices to a compressed NPZ.

    ``copula`` maps each intensity bin to ``{"tau": [ncrl,4,4], "rho": [ncrl,4,4]}``.
    The 4x4 axes follow ``PARAM_NAMES`` ([Hd, Dp, Rmax, Vt]).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "crl_id": np.asarray(crl_ids),
        "lat": np.asarray(lat, float),
        "lon": np.asarray(lon, float),
        "param_names": np.array(PARAM_NAMES),
    }
    for b in INTENSITY_BINS:
        arrays[f"tau_{b}"] = copula[b]["tau"]
        arrays[f"rho_{b}"] = copula[b]["rho"]
    np.savez_compressed(path, **arrays)
    return path if path.suffix == ".npz" else Path(str(path) + ".npz")


def write_adjusted(adjusted_rows: List[dict], path) -> Path:
    """Write the adjusted, intensity-labeled per-storm parameters (long form).

    One row per CRL x selected TC (the All bin), with a ``stratum`` label (high/med/
    low) and the adjusted [Hd, Dp, Rmax, Vt]. HI/MI/LI subsets are recoverable by
    filtering ``stratum``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["crl_id", "stratum", "Hd", "Dp", "Rmax", "Vt"]
    pd.DataFrame(adjusted_rows, columns=cols).to_csv(path, index=False)
    return path
