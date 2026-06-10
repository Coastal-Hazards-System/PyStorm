"""Feature engineering and model specs for the Cp / Rmax GP metamodels.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Mirrors the predictor sets of GPM_Cp.m / the DI scripts:

  Central pressure (response = 1013 - pmin, the deficit):
    Cp6  [lat, lon, vmax, Vf, sin Hd, cos Hd]   - fixes with known motion
    Cp3  [lat, lon, vmax]                        - fixes lacking Vf/Hd
  Radius of maximum wind (response = rmax):
    Rm7  [lat, lon, vmax, Cp-deficit, Vf, sin Hd, cos Hd]
    Rm4  [lat, lon, vmax, Cp-deficit]

A fix is routed to the smaller model when its translation speed OR heading is
missing - i.e. single-point TCs, the first fix (no motion), and stationary
fixes (Vf=0 was set to NaN). sin/cos make the heading sign-wrap irrelevant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CP_BASE = 1013.0           # reference pressure for the deficit (matches MATLAB)
RMAX_MIN, RMAX_MAX = 8.0, 600.0   # physical clamp on imputed Rmax (km)


def motion_known(df: pd.DataFrame) -> np.ndarray:
    """True where both translation speed and heading are present."""
    return df["trans_kmh"].notna().to_numpy() & df["heading_deg"].notna().to_numpy()


def _sincos(df: pd.DataFrame):
    h = np.deg2rad(df["heading_deg"].to_numpy(float))
    return np.sin(h), np.cos(h)


def _stack(*cols) -> np.ndarray:
    return np.column_stack([np.asarray(c, float) for c in cols])


def cp_features_full(df: pd.DataFrame) -> np.ndarray:
    """Cp6 inputs: [lat, lon, vmax, Vf, sin Hd, cos Hd]."""
    s, c = _sincos(df)
    return _stack(df["lat"], df["lon"], df["vmax_kmh"], df["trans_kmh"], s, c)


def cp_features_small(df: pd.DataFrame) -> np.ndarray:
    """Cp3 inputs: [lat, lon, vmax]."""
    return _stack(df["lat"], df["lon"], df["vmax_kmh"])


def rm_features_full(df: pd.DataFrame, pmin: np.ndarray) -> np.ndarray:
    """Rm7 inputs: [lat, lon, vmax, Cp-deficit, Vf, sin Hd, cos Hd]."""
    s, c = _sincos(df)
    return _stack(df["lat"], df["lon"], df["vmax_kmh"],
                  CP_BASE - np.asarray(pmin, float), df["trans_kmh"], s, c)


def rm_features_small(df: pd.DataFrame, pmin: np.ndarray) -> np.ndarray:
    """Rm4 inputs: [lat, lon, vmax, Cp-deficit]."""
    return _stack(df["lat"], df["lon"], df["vmax_kmh"],
                  CP_BASE - np.asarray(pmin, float))


def finite_rows(X: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose every feature is finite."""
    return np.isfinite(X).all(axis=1)
