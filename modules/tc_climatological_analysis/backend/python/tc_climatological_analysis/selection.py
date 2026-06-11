"""Per-CRL storm selection (Gaussian-weighted representative point).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of ``CHS_Atlantic_StormSelection.m``. For every CRL, each tropical cyclone
whose track passes within ``max_dist`` contributes one record: a representative
track point chosen as the fix that maximizes the Gaussian distance weight times
the central-pressure deficit (proximity x intensity), plus the closest-approach
distance that drives the recurrence-rate kernel.

The MATLAB loops CRL-by-CRL then TC-by-TC; here the inner CRL sweep is fully
vectorized (one TC at a time, all CRLs at once), which is far faster while
reproducing the same selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_R_KM = 6371.0   # matches the MATLAB distance()/deg2km default sphere

# Output columns of the selection table (one row per selected TC at a CRL).
SELECT_COLUMNS = [
    "crl_id", "year", "storm_no", "name", "genesis_yyyymm", "month", "doy",
    "lat", "lon", "dist_rep", "trans_kmh", "heading_deg", "vmax_kmh",
    "cp_gauss", "cp_mindist", "gaussW", "dist", "rmax_km", "dp",
]

# Cumulative days before each month on a fixed 365-day (non-leap) calendar; used
# to map a YYYYMMDD closest-approach date to a day-of-year in [1, 365].
_CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def ymd_to_doy(ymd: np.ndarray) -> np.ndarray:
    """Day-of-year in [1, 365] for an array of YYYYMMDD integers (fixed 365-day
    calendar; a Feb-29 closest approach folds onto its non-leap ordinal)."""
    ymd = np.asarray(ymd, dtype=np.int64)
    month = ((ymd // 100) % 100).astype(int)
    day = (ymd % 100).astype(int)
    month = np.clip(month, 1, 12)
    doy = _CUM_DAYS[month - 1] + day
    return np.clip(doy, 1, 365).astype(int)


def gaussian_weights(k_size: float, dist_km: np.ndarray) -> np.ndarray:
    """Gaussian distance weight, 1/(sqrt(2*pi)*K) * exp(-0.5*(d/K)^2). Port of GaussianWeights.m."""
    return 1.0 / (np.sqrt(2.0 * np.pi) * k_size) * np.exp(-0.5 * (dist_km / k_size) ** 2)


def _haversine_km(lat1, lon1, lat2, lon2, r: float = _EARTH_R_KM):
    """Great-circle distance (km). Broadcasts; matches MATLAB distance()+distdim."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _storm_no(nhc_id: str) -> int:
    """Storm number within the season from the cyclone id (e.g. 'AL011851' -> 1)."""
    try:
        return int(str(nhc_id)[2:4])
    except (ValueError, TypeError):
        return 0


def select_storms(
    hurdat: pd.DataFrame,
    crls: pd.DataFrame,
    *,
    k_size: float = 200.0,
    max_dist: float = 600.0,
    max_cp: float = 1005.0,
    ref_pressure: float = 1013.0,
) -> pd.DataFrame:
    """Select, per CRL, the representative record of every TC within ``max_dist``.

    ``hurdat`` is the augmented best-track table; ``crls`` has columns id/lat/lon.
    Returns a long table (one row per CRL x selected-TC) with ``SELECT_COLUMNS``.
    """
    # Global fix filter (as in the MATLAB): usable central pressure and position.
    df = hurdat
    keep = (df["pmin_hpa"].notna() & (df["pmin_hpa"] <= max_cp)
            & df["lat"].notna() & df["lon"].notna())
    df = df.loc[keep]
    if df.empty:
        return pd.DataFrame(columns=SELECT_COLUMNS)

    crl_id = crls["id"].to_numpy()
    clat = crls["lat"].to_numpy(float)[:, None]      # (Ncrl, 1)
    clon = crls["lon"].to_numpy(float)[:, None]
    ncrl = len(crl_id)
    arange = np.arange(ncrl)

    cols: dict[str, list] = {c: [] for c in SELECT_COLUMNS}

    for _tc_no, trk in df.groupby("tc_no", sort=True):
        tlat = trk["lat"].to_numpy(float)
        tlon = trk["lon"].to_numpy(float)
        tpmin = trk["pmin_hpa"].to_numpy(float)
        tvmax = trk["vmax_kmh"].to_numpy(float)
        ttrans = trk["trans_kmh"].to_numpy(float)
        thead = trk["heading_deg"].to_numpy(float)
        trmax = trk["rmax_km"].to_numpy(float)
        tymd = trk["ymd"].to_numpy(np.int64)
        year = int(trk["year"].iloc[0])
        name = str(trk["name"].iloc[0])
        storm_no = _storm_no(trk["nhc_id"].iloc[0])
        genesis_yyyymm = int(tymd[0] // 100)

        # Distances from every CRL to every fix of this TC -> (Ncrl, K).
        d = _haversine_km(clat, clon, tlat[None, :], tlon[None, :])
        min_d = d.min(axis=1)
        hit = min_d <= max_dist
        if not hit.any():
            continue

        dh = d[hit]                                  # (nhit, K)
        nhit = dh.shape[0]
        gw = gaussian_weights(k_size, dh)
        dp = ref_pressure - tpmin                    # (K,)
        wcp = gw * dp[None, :]
        iw = wcp.argmax(axis=1)                      # representative fix per CRL
        idm = dh.argmin(axis=1)                      # closest-approach fix per CRL
        rows = np.arange(nhit)

        cp_g = tpmin[iw]
        cols["crl_id"].append(crl_id[hit])
        cols["year"].append(np.full(nhit, year))
        cols["storm_no"].append(np.full(nhit, storm_no))
        cols["name"].append(np.full(nhit, name, dtype=object))
        cols["genesis_yyyymm"].append(np.full(nhit, genesis_yyyymm))
        cols["month"].append(((tymd[idm] // 100) % 100).astype(int))
        cols["doy"].append(ymd_to_doy(tymd[idm]))
        cols["lat"].append(tlat[iw])
        cols["lon"].append(tlon[iw])
        cols["dist_rep"].append(dh[rows, iw])
        cols["trans_kmh"].append(ttrans[iw])
        cols["heading_deg"].append(thead[iw])
        cols["vmax_kmh"].append(tvmax[iw])
        cols["cp_gauss"].append(cp_g)
        cols["cp_mindist"].append(tpmin[idm])
        cols["gaussW"].append(gw[rows, iw])
        cols["dist"].append(min_d[hit])              # closest-approach distance (drives SRR)
        cols["rmax_km"].append(trmax[iw])
        cols["dp"].append(ref_pressure - cp_g)

    if not cols["crl_id"]:
        return pd.DataFrame(columns=SELECT_COLUMNS)
    data = {c: np.concatenate(v) for c, v in cols.items()}
    out = pd.DataFrame(data, columns=SELECT_COLUMNS)
    return out.sort_values(["crl_id", "year", "storm_no"]).reset_index(drop=True)
