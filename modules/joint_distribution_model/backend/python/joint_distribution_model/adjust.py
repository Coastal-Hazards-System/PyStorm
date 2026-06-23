"""adjust - Stage 1: distance-weighted parameter adjustment and intensity binning.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

For each CRL, every selected TC's parameters [Heading, Dp, Rmax, Vt] are
rescaled so the sample mean/std are replaced by the Gaussian-distance-weighted
mean/std while each storm keeps its z-score; heading is recentered on the DSRR
circular mean. The adjusted storms are then split into intensity bins by deficit Dp.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def distance_weighted_adj(param: np.ndarray, wght: np.ndarray):
    """Distance-weighted adjustment of a parameter.

    Returns
    -------
    param_adj : ndarray   z-scores rescaled to the weighted mean/std:
                          ``param_adj = z*sigma_dw + mu_dw``, ``z = (x-mu)/sigma``.
    stat : dict           {mu, mu_dw, sigma, sigma_dw} for inspection/tests.
    """
    param = np.asarray(param, dtype=float)
    wght = np.asarray(wght, dtype=float)
    n = param.size                                      # count of all rows (NaNs included)
    mu = np.nansum(param) / n
    sigma = np.sqrt(np.nansum((param - mu) ** 2) / (n - 1)) if n > 1 else 0.0
    sw = np.nansum(wght)
    mu_dw = np.nansum(param * wght) / sw
    sigma_dw = np.sqrt(np.nansum(wght * (param - mu_dw) ** 2)
                       / ((n - 1) / n * sw)) if n > 1 else 0.0
    z = np.zeros_like(param) if sigma == 0 else (param - mu) / sigma
    param_adj = z * sigma_dw + mu_dw
    return param_adj, {"mu": mu, "mu_dw": mu_dw, "sigma": sigma, "sigma_dw": sigma_dw}


def distance_weighted_adj_heading(param: np.ndarray, mu_dw: float, sigma_dw: float):
    """Heading variant: the weighted mean/std are supplied (the DSRR mean/stdv),
    not computed from distance weights."""
    param = np.asarray(param, dtype=float)
    n = param.size
    mu = np.nansum(param) / n
    sigma = np.sqrt(np.nansum((param - mu) ** 2) / (n - 1)) if n > 1 else 0.0
    z = np.zeros_like(param) if sigma == 0 else (param - mu) / sigma
    return z * sigma_dw + mu_dw


def heading_zero_degree_adj(hd_in: np.ndarray, hd_mean: float) -> np.ndarray:
    """Recenter headings on ``hd_mean`` (the DSRR circular mean) and wrap to (-180, 180]."""
    out = np.asarray(hd_in, dtype=float) - hd_mean
    out = np.where(out > 180, out - 360, out)
    out = np.where(out < -180, out + 360, out)
    return out


def _wrap_180(x: np.ndarray) -> np.ndarray:
    """Wrap angles into (-180, 180] by single +/-360 shifts."""
    x = np.where(x < -180, x + 360, x)
    x = np.where(x > 180, x - 360, x)
    return x


def adjust_crl(*, heading, cp, rmax, vt, gaussW, year,
               dsrr_mean_all: float, dsrr_stdv_all: float,
               ref_pressure: float, start_year: int,
               min_dp: float, dp_low: float, dp_med: float,
               vt_clip=(1.0, 152.0), rmax_clip=(8.0, 200.0)) -> Dict[str, np.ndarray]:
    """Adjust one CRL's selected TCs and split into intensity bins.

    Returns a dict mapping the SCA bin name (all/high/med/low) to an ``[N, 4]`` array
    of adjusted ``[Hd, Dp, Rmax, Vt]``. The distance-weighted statistics use the full
    record; the ``start_year`` filter and the binning are applied afterward.
    """
    cp = np.asarray(cp, dtype=float)
    dp = ref_pressure - cp
    w = np.asarray(gaussW, dtype=float)

    dp_adj, _ = distance_weighted_adj(dp, w)
    vt_adj, _ = distance_weighted_adj(np.asarray(vt, float), w)
    vt_adj = np.clip(vt_adj, vt_clip[0], vt_clip[1])
    rm_adj, _ = distance_weighted_adj(np.asarray(rmax, float), w)
    rm_adj = np.clip(rm_adj, rmax_clip[0], rmax_clip[1])

    hd_adj = distance_weighted_adj_heading(np.asarray(heading, float),
                                           dsrr_mean_all, dsrr_stdv_all)
    hd_adj = _wrap_180(hd_adj)
    hd_adj = heading_zero_degree_adj(hd_adj, dsrr_mean_all)

    # Drop pre-start_year storms, then bin by adjusted deficit.
    keep = np.asarray(year, dtype=float) >= start_year
    hd_adj, dp_adj, rm_adj, vt_adj = (a[keep] for a in (hd_adj, dp_adj, rm_adj, vt_adj))
    data = np.column_stack([hd_adj, dp_adj, rm_adj, vt_adj])   # [N, 4]: Hd, Dp, Rmax, Vt
    d = data[:, 1]                                             # adjusted Dp

    return {
        "all":  data[d >= min_dp],
        "high": data[d >= dp_med],
        "med":  data[(d >= dp_low) & (d < dp_med)],
        "low":  data[(d >= min_dp) & (d < dp_low)],
    }
