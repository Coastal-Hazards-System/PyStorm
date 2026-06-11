"""Gaussian Kernel Function (GKF) storm recurrence rates.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of ``CHS_SRR_GKF.m`` (with ``AzimuthDiff.m`` and ``headingZeroDegree_adj.m``).
For each CRL and intensity bin:

  * SRR  - omnidirectional storm recurrence rate, ``(1/Nyrs) * sum_i Wi`` with the
    distance kernel ``Wi = 1/(sqrt(2*pi)*K) * exp(-0.5*(D/K)^2)``. Units:
    storms / km / year.
  * DSRR - directional rate ``Ld(theta) = (1/Nyrs) * sum_i Wd_i(theta) * Wi`` with
    the heading kernel ``Wd = 1/(sqrt(2*pi)*sigma) * exp(-0.5*(dtheta/sigma)^2)``.
    Units: storms / degree / year. The normalized shape ``Ld/sum(Ld)`` is recentered
    on its circular mean (``headingZeroDegree_adj``) into a heading pdf/cdf with a
    mean and standard deviation.

Monthly (Jan-Dec) rates partition each CRL's selected storms by the calendar month
of their closest approach and apply the same kernels, still normalized by Nyrs - so
the twelve monthly rates sum exactly to the annual rate.

Intensity bins use the central-pressure deficit ``dp = 1013 - Cp``:
  All ``dp >= min_dp`` | Low ``[min_dp, dp_low)`` | Med ``[dp_low, dp_med)`` | High ``[dp_med, inf)``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from tc_climatological_analysis.selection import gaussian_weights

# Heading grid (-179..180 deg), matching the MATLAB ``hding``.
HEADINGS = np.arange(-179, 181, dtype=float)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Intensity bins: name -> (dp_min, dp_max) on the deficit dp = 1013 - Cp.
def intensity_bins(min_dp: float, dp_low: float, dp_med: float) -> dict:
    return {
        "all":  (min_dp, np.inf),
        "low":  (min_dp, dp_low),
        "med":  (dp_low, dp_med),
        "high": (dp_med, np.inf),
    }


def azimuth_diff(hding: np.ndarray, hdgr: np.ndarray) -> np.ndarray:
    """Angular difference |wrap_180(hding - heading)| per (storm, heading grid).

    Vectorized equivalent of ``AzimuthDiff.m``: for each storm heading the smallest
    signed difference to every grid heading, wrapped to (-180, 180] and taken
    absolute. NaN storm headings (stationary fixes) propagate to NaN.
    """
    diff = hding[None, :] - hdgr[:, None]            # (N, 360)
    diff = (diff + 180.0) % 360.0 - 180.0            # wrap to (-180, 180]
    return np.abs(diff)


def heading_zero_degree_adj(hd_in: np.ndarray):
    """Recenter a heading pdf on its circular mean (``headingZeroDegree_adj`` type 1).

    Returns (pdf, mean, stdv). ``hd_in`` is a non-negative shape over ``HEADINGS``.
    """
    s = np.nansum(hd_in)
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(HEADINGS), np.nan, np.nan
    hd_in = np.nan_to_num(hd_in)
    hd_mean = float(np.sum(HEADINGS * hd_in))
    hd_stdv = float(np.sqrt(np.sum(hd_in * (HEADINGS - hd_mean) ** 2)))
    aux = HEADINGS - hd_mean
    aux = np.where(aux > 180.0, aux - 360.0, aux)
    aux = np.where(aux < -180.0, aux + 360.0, aux)
    order = np.argsort(aux)
    pdf = np.interp(HEADINGS, aux[order], hd_in[order])   # recentered shape
    pdf[pdf < 0] = 0.0
    tot = pdf.sum()
    if tot > 0:
        pdf = pdf / tot
    return pdf, hd_mean, hd_stdv


def _bin_mask(dp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (dp >= lo) & (dp < hi)


def compute_rates(
    selection: pd.DataFrame,
    crls: pd.DataFrame,
    *,
    k_size: float,
    dir_kernel: float,
    start_year: int,
    end_year: int,
    min_dp: float,
    dp_low: float,
    dp_med: float,
) -> Dict[str, dict]:
    """Compute SRR + DSRR (annual and monthly) per CRL for every intensity bin.

    Returns ``{bin_name: {srr, srr_monthly, dsrr_rate, dsrr_rate_monthly, pdf,
    cdf, mean, stdv}}`` plus a top-level ``{"_meta": {...}}`` entry. Arrays are
    indexed by CRL order in ``crls``.
    """
    nyrs = int(end_year) - int(start_year) + 1
    if nyrs <= 0:
        raise ValueError(f"Non-positive Nyrs ({nyrs}); check start_year/end_year.")
    bins = intensity_bins(min_dp, dp_low, dp_med)
    crl_ids = crls["id"].to_numpy()
    ncrl = len(crl_ids)
    nhd = HEADINGS.size

    # Pre-group the selection by CRL, season-filtered once.
    sel = selection[selection["year"] >= start_year]
    by_crl = {cid: g for cid, g in sel.groupby("crl_id")}

    out: Dict[str, dict] = {}
    for name in bins:
        out[name] = {
            "srr": np.zeros(ncrl),
            "srr_monthly": np.zeros((ncrl, 12)),
            "dsrr_rate": np.zeros((ncrl, nhd)),
            "dsrr_rate_monthly": np.zeros((ncrl, 12, nhd)),
            "pdf": np.zeros((ncrl, nhd)),
            "cdf": np.zeros((ncrl, nhd + 1)),
            "mean": np.full(ncrl, np.nan),
            "stdv": np.full(ncrl, np.nan),
        }

    for i, cid in enumerate(crl_ids):
        g = by_crl.get(cid)
        if g is None or g.empty:
            continue
        dp_all = g["dp"].to_numpy(float)
        dist_all = g["dist"].to_numpy(float)
        head_all = g["heading_deg"].to_numpy(float)
        month_all = g["month"].to_numpy(int)

        for name, (lo, hi) in bins.items():
            m = _bin_mask(dp_all, lo, hi)
            if not m.any():
                continue
            dist = dist_all[m]
            head = head_all[m]
            month = month_all[m]

            wi = gaussian_weights(k_size, dist)                       # (N,)
            srr = float(np.nansum(wi)) / nyrs

            hdiff = azimuth_diff(HEADINGS, head)                      # (N, 360)
            wd = (1.0 / (np.sqrt(2.0 * np.pi) * dir_kernel)
                  * np.exp(-0.5 * (hdiff / dir_kernel) ** 2))
            contrib = np.nan_to_num(wd * wi[:, None])                # (N, 360)
            ld = contrib.sum(axis=0) / nyrs                          # (360,)

            # Monthly partition via a 12xN indicator (months sum to the annual rate).
            valid = (month >= 1) & (month <= 12)
            ind = np.zeros((12, len(month)))
            ind[month[valid] - 1, np.nonzero(valid)[0]] = 1.0
            srr_m = ind @ wi / nyrs                                  # (12,)
            ld_m = ind @ contrib / nyrs                             # (12, 360)

            b = out[name]
            b["srr"][i] = srr
            b["srr_monthly"][i] = srr_m
            b["dsrr_rate"][i] = ld
            b["dsrr_rate_monthly"][i] = ld_m

            tot = ld.sum()
            if tot > 0:
                pdf, hmean, hstdv = heading_zero_degree_adj(ld / tot)
                b["pdf"][i] = pdf
                b["cdf"][i] = np.concatenate([[0.0], np.cumsum(pdf)])
                b["mean"][i] = hmean
                b["stdv"][i] = hstdv

    out["_meta"] = {"nyrs": nyrs, "start_year": int(start_year),
                    "end_year": int(end_year), "k_size": k_size,
                    "dir_kernel": dir_kernel, "headings": HEADINGS,
                    "months": MONTHS, "bins": list(bins)}
    return out
