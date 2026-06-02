"""curve — GPD ensemble construction, empirical-tail blending, AEF interpolation.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Builds the hazard-curve ensemble from a bootstrap matrix of POT exceedances,
splices the GPD upper tail onto the empirical lower tail, and interpolates
the merged curve onto the standard 22-AER reporting grid.

Public API
----------
  make_aef_grids()
      Return (aef_table, aef_plot) — the standard 22-AER table and the dense
      plotting grid (log-spaced, 1e-3 .. 10).

  fit_gpd_ensemble(boot_matrix, threshold, aef_plot, lambda_th,
                   shape_clip_low, shape_clip_high)
      Fit a GPD per bootstrap column and evaluate the ICDF at aef_plot.
      Returns (ensemble, gpd_be, gpd_cb10, gpd_cb90, aef_gpd_mask).

  assemble_hazard_curve(aef_gpd, gpd_be, gpd_cb10, gpd_cb90,
                        aef_below_th, pot_below_th)
      Concatenate the GPD-tail and empirical-bulk segments to a single
      hazard curve (aef, be, cb10, cb90).

  interpolate_to_table(aef_table, hc_aef, hc_be, hc_cb10, hc_cb90)
      Log-AEF interpolation of the merged curve onto the table grid.
"""

import warnings
from typing import Tuple

import numpy as np
from scipy.stats import genpareto


# ── 22-AER reporting grid (return periods 0.1 yr ... 1e6 yr) ──────────────
_RETURN_PERIODS = np.array(
    [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
     1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1e6],
    dtype=np.float64,
)


def make_aef_grids() -> Tuple[np.ndarray, np.ndarray]:
    """Return (aef_table, aef_plot).

    aef_table : (22,) float64 — 1 / return_period; entries past index 12 (>1e-3)
                are masked to NaN to suppress extrapolation in reporting.
    aef_plot  : (361,) float64 — dense log-decade grid from 10 down to 1e-3.
    """
    aef_table = 1.0 / _RETURN_PERIODS
    aef_table[13:] = np.nan

    # Match the v1 sub-decade fill: 1/90 in log10, seven decades wide, trim to >=1e-3.
    d = 1 / 90
    v = 10 ** np.arange(1, -d, -d)
    chunks = [v]
    x = 10.0
    for _ in range(6):
        chunks.append(v[1:] / x)
        x *= 10.0
    aef_plot = np.concatenate(chunks)
    aef_plot = np.flip(aef_plot)
    aef_plot = aef_plot[270:]
    return aef_table, aef_plot


def fit_gpd_ensemble(
    boot_matrix:     np.ndarray,
    threshold:       float,
    aef_plot:        np.ndarray,
    lambda_th:       float,
    shape_clip_low:  float,
    shape_clip_high: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit GPD per bootstrap column, evaluate at the GPD-relevant aef_plot mask.

    Parameters
    ----------
    boot_matrix : (n_pot, n_sims) float64
        Descending-sorted bootstrap realizations above the threshold.
    threshold : float
        GPD location parameter (held fixed).
    aef_plot : (n_aef_plot,) float64
        Dense plotting AEF grid.
    lambda_th : float
        Exceedance rate above the threshold (events / yr).
    shape_clip_low, shape_clip_high : float
        Luceño-style admissible bounds for the GPD shape parameter.

    Returns
    -------
    ensemble       : (n_sims, n_aef_plot) float64; NaN where AEF >= lambda_th
                     (outside the GPD-valid band) or where a fit failed.
    gpd_be         : (n_aef_gpd,) float64 mean across realizations
    gpd_cb10       : (n_aef_gpd,) float64 10th-percentile bound
    gpd_cb90       : (n_aef_gpd,) float64 90th-percentile bound
    aef_gpd_mask   : (n_aef_plot,) bool — True where aef_plot < lambda_th
    """
    n_pot, n_sims = boot_matrix.shape
    n_aef         = aef_plot.size

    aef_gpd_mask  = aef_plot < lambda_th
    aef_gpd       = aef_plot[aef_gpd_mask]
    quantiles_gpd = 1.0 - aef_plot / lambda_th  # only valid where mask is True

    ensemble = np.full((n_sims, n_aef), np.nan, dtype=np.float64)
    for j in range(n_sims):
        sample = boot_matrix[:, j]
        try:
            c, _loc, scale = genpareto.fit(sample, floc=threshold)
            c = max(min(c, shape_clip_high), shape_clip_low)
            ensemble[j, :] = genpareto.ppf(quantiles_gpd, c,
                                           loc=threshold, scale=scale)
        except Exception:
            continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gpd_be   = np.nanmean(ensemble[:, aef_gpd_mask],       axis=0)
        gpd_cb90 = np.nanpercentile(ensemble[:, aef_gpd_mask], 90, axis=0)
        gpd_cb10 = np.nanpercentile(ensemble[:, aef_gpd_mask], 10, axis=0)

    return ensemble, gpd_be, gpd_cb10, gpd_cb90, aef_gpd_mask


def assemble_hazard_curve(
    aef_gpd:      np.ndarray,
    gpd_be:       np.ndarray,
    gpd_cb10:     np.ndarray,
    gpd_cb90:     np.ndarray,
    aef_below_th: np.ndarray,
    pot_below_th: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splice the GPD tail and empirical bulk into a single hazard curve.

    Empirical-bulk uncertainty is taken as zero (CB10 = CB90 = best estimate)
    per the v1 convention.
    """
    aef  = np.concatenate([aef_gpd,  aef_below_th])
    be   = np.concatenate([gpd_be,   pot_below_th])
    cb10 = np.concatenate([gpd_cb10, pot_below_th])
    cb90 = np.concatenate([gpd_cb90, pot_below_th])
    return aef, be, cb10, cb90


def interpolate_to_table(
    aef_table: np.ndarray,
    hc_aef:    np.ndarray,
    hc_be:     np.ndarray,
    hc_cb10:   np.ndarray,
    hc_cb90:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the merged hazard curve onto the 22-AER table grid.

    Interpolation is performed in log10(AEF) space. Table entries that the
    source curve cannot cover (NaN slots in ``aef_table``) remain NaN.
    """
    log_src  = np.log10(hc_aef)
    valid    = ~np.isnan(aef_table)
    log_tbl  = np.log10(aef_table[valid])

    out_be   = np.full_like(aef_table, np.nan)
    out_cb10 = np.full_like(aef_table, np.nan)
    out_cb90 = np.full_like(aef_table, np.nan)

    out_be  [valid] = np.interp(log_tbl, log_src, hc_be)
    out_cb10[valid] = np.interp(log_tbl, log_src, hc_cb10)
    out_cb90[valid] = np.interp(log_tbl, log_src, hc_cb90)

    return out_be, out_cb10, out_cb90
