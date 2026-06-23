"""sca_source - locate and load the storm_climatology_analysis (SCA) inputs.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

JDM consumes two SCA products per basin: the per-CRL selected-TC table
(selection_<basin>_<v>.csv, one row per CRL x selected TC) and the DSRR arrays
(dsrr_<basin>_<v>.npz, the per-CRL directional heading mean/stdv/cdf per intensity
bin). This module finds the newest matching files (or a pinned path), loads them,
and exposes a small DSRR accessor keyed by CRL id. No fitting logic here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def _locate(basin: str, prefix: str, suffix: str,
            explicit_file: Optional[Union[str, Path]],
            input_dir: Path, sca_outputs_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve a SCA file: pinned path, else newest ``<prefix>_<basin>_*<suffix>``.

    A pinned file is absolute or relative to ``input_dir``. Otherwise the newest
    match (by modified time) under ``sca_outputs_dir`` is used.
    """
    if explicit_file is not None:
        p = Path(explicit_file)
        p = p if p.is_absolute() else Path(input_dir) / p
        if not p.is_file():
            raise FileNotFoundError(f"Pinned SCA file not found: {p}")
        return p
    if sca_outputs_dir is None:
        raise FileNotFoundError(
            f"No {prefix} file pinned and sca_outputs_dir is unset; cannot locate "
            f"{prefix}_{basin}_*{suffix}.")
    base = Path(sca_outputs_dir)
    matches = sorted(base.glob(f"{prefix}_{basin}_*{suffix}"),
                     key=lambda q: q.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No {prefix}_{basin}_*{suffix} under {base}. Run the "
            f"storm_climatology_analysis module first.")
    return matches[0]


def locate_selection(basin, explicit_file, input_dir, sca_outputs_dir) -> Path:
    """Newest (or pinned) SCA selection table for a basin."""
    return _locate(basin, "selection", ".csv", explicit_file, input_dir, sca_outputs_dir)


def locate_dsrr(basin, explicit_file, input_dir, sca_outputs_dir) -> Path:
    """Newest (or pinned) SCA DSRR arrays for a basin."""
    return _locate(basin, "dsrr", ".npz", explicit_file, input_dir, sca_outputs_dir)


def version_tag(selection_path: Path, basin: str) -> str:
    """Output tag from the selection filename, e.g. ``atlantic_1938-2025_20260227``.

    Falls back to the basin alone if the filename does not carry a version suffix.
    """
    m = re.match(rf"selection_({re.escape(basin)}_.+)$", Path(selection_path).stem)
    return m.group(1) if m else basin


def load_selection(path, usecols=None) -> pd.DataFrame:
    """Load the per-CRL selected-TC table (optionally only ``usecols``).

    Reading just the columns the fit needs keeps the wide multi-MB table cheap to
    load; ``usecols`` must include ``crl_id``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SCA selection table not found: {path}")
    df = pd.read_csv(path, usecols=usecols)
    if "crl_id" not in df.columns:
        raise ValueError(f"{path.name} has no 'crl_id' column.")
    return df


@dataclass
class DSRR:
    """Per-CRL directional heading stats per intensity bin, from the SCA DSRR npz."""
    headings: np.ndarray                       # [nhd] heading grid (deg), -179..180
    crl_ids: np.ndarray                        # [ncrl]
    lat: np.ndarray                            # [ncrl] CRL latitude
    lon: np.ndarray                            # [ncrl] CRL longitude
    mean: Dict[str, np.ndarray]                # bin -> [ncrl] circular-mean heading
    stdv: Dict[str, np.ndarray]                # bin -> [ncrl] heading stdv
    cdf: Dict[str, np.ndarray]                 # bin -> [ncrl, nhd(+1)] recentered cdf
    _row: Dict[int, int] = None                # crl_id -> row index

    def __post_init__(self):
        self._row = {int(c): i for i, c in enumerate(self.crl_ids)}

    def row(self, crl_id: int) -> int:
        """Row index for a CRL id (KeyError if absent)."""
        return self._row[int(crl_id)]

    def coord(self, crl_id: int):
        """(lat, lon) of a CRL."""
        i = self.row(crl_id)
        return float(self.lat[i]), float(self.lon[i])

    def heading_stats(self, crl_id: int, bin_name: str):
        """(mean, stdv) heading for a CRL and intensity bin."""
        i = self.row(crl_id)
        return float(self.mean[bin_name][i]), float(self.stdv[bin_name][i])

    def heading_cdf(self, crl_id: int, bin_name: str) -> np.ndarray:
        """Recentered heading cdf for a CRL and intensity bin."""
        return self.cdf[bin_name][self.row(crl_id)]


def load_dsrr(path, bins=("all", "high", "med", "low")) -> DSRR:
    """Load the SCA DSRR arrays into a per-CRL accessor over the requested bins."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SCA DSRR arrays not found: {path}")
    z = np.load(path, allow_pickle=False)
    mean = {b: z[f"dsrr_mean_{b}"] for b in bins}
    stdv = {b: z[f"dsrr_stdv_{b}"] for b in bins}
    cdf = {b: z[f"dsrr_cdf_{b}"] for b in bins}
    return DSRR(headings=z["headings"], crl_ids=z["crl_id"],
                lat=z["lat"], lon=z["lon"], mean=mean, stdv=stdv, cdf=cdf)
