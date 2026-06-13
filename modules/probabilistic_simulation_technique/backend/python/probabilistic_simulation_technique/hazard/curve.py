"""curve - GPD ensemble construction, empirical-tail blending, AER interpolation.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Builds the hazard-curve ensemble from a bootstrap matrix of POT exceedances,
splices the GPD upper tail onto the empirical lower tail, and interpolates
the merged curve onto the standard 22-AER reporting grid.

Public API
----------
  make_aer_grids()
      Return (aer_table, aer_plot) - the standard 22-AER table and the dense
      plotting grid (log-spaced, 1e-3 .. 10).

  fit_gpd_ensemble(boot_matrix, threshold, aer_plot, lambda_mu,
                   shape_clip_low, shape_clip_high)
      Fit a GPD per bootstrap column and evaluate the ICDF at aer_plot.
      Returns (ensemble, gpd_be, gpd_cb10, gpd_cb90, aer_gpd_mask).

  assemble_hazard_curve(aer_gpd, gpd_be, gpd_cb10, gpd_cb90,
                        aer_below_th, pot_below_th)
      Concatenate the GPD-tail and empirical-bulk segments to a single
      hazard curve (aer, be, cb10, cb90).

  interpolate_to_table(aer_table, hc_aer, hc_be, hc_cb10, hc_cb90)
      Log-AER interpolation of the merged curve onto the table grid.
"""

import warnings
from typing import Tuple

import numpy as np
from scipy.stats import genpareto

from ..gpd_fit import fit_gpd_clipped


# ── 22-AER reporting grid (mean return intervals 0.1 yr ... 1e6 yr) ────────
_MEAN_RETURN_INTERVALS = np.array(
    [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
     1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1e6],
    dtype=np.float64,
)


def make_aer_grids() -> Tuple[np.ndarray, np.ndarray]:
    """Return (aer_table, aer_plot).

    aer_table : (22,) float64 - 1 / MRI (mean return interval); entries past
                index 12 (>1e-3) are masked to NaN to suppress extrapolation.
    aer_plot  : (361,) float64 - dense log-decade grid from 10 down to 1e-3.
    """
    aer_table = 1.0 / _MEAN_RETURN_INTERVALS
    aer_table[13:] = np.nan

    # Match the v1 sub-decade fill: 1/90 in log10, seven decades wide, trim to >=1e-3.
    d = 1 / 90
    v = 10 ** np.arange(1, -d, -d)
    chunks = [v]
    x = 10.0
    for _ in range(6):
        chunks.append(v[1:] / x)
        x *= 10.0
    aer_plot = np.concatenate(chunks)
    aer_plot = np.flip(aer_plot)
    aer_plot = aer_plot[270:]
    return aer_table, aer_plot


def fit_gpd_ensemble(
    boot_matrix:     np.ndarray,
    threshold:       float,
    aer_plot:        np.ndarray,
    lambda_mu:       float,
    shape_clip_low:  float,
    shape_clip_high: float,
    fit_method:      str = "mle",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit GPD per bootstrap column, evaluate at the GPD-relevant aer_plot mask.

    Parameters
    ----------
    boot_matrix : (n_pot, n_sims) float64
        Descending-sorted bootstrap realizations above the threshold.
    threshold : float
        GPD location parameter μ (held fixed; the QDO-optimized threshold).
    aer_plot : (n_aer_plot,) float64
        Dense plotting AER grid.
    lambda_mu : float
        Exceedance rate above μ, λ_μ (events / yr).
    shape_clip_low, shape_clip_high : float
        Luceño-style admissible bounds for the GPD shape parameter.

    Returns
    -------
    ensemble       : (n_sims, n_aer_plot) float64; NaN where AER >= lambda_mu
                     (outside the GPD-valid band) or where a fit failed.
    gpd_be         : (n_aer_gpd,) float64 mean across realizations
    gpd_cb10       : (n_aer_gpd,) float64 10th-percentile bound
    gpd_cb90       : (n_aer_gpd,) float64 90th-percentile bound
    aer_gpd_mask   : (n_aer_plot,) bool - True where aer_plot < lambda_mu
    """
    n_pot, n_sims = boot_matrix.shape
    n_aer         = aer_plot.size

    aer_gpd_mask  = aer_plot < lambda_mu
    aer_gpd       = aer_plot[aer_gpd_mask]
    quantiles_gpd = 1.0 - aer_plot / lambda_mu  # only valid where mask is True

    ensemble = np.full((n_sims, n_aer), np.nan, dtype=np.float64)
    for j in range(n_sims):
        sample = boot_matrix[:, j]
        try:
            # Shared fit: ξ clipped to [shape_clip_low, shape_clip_high] and σ
            # refit when ξ is clipped - identical to the QDO objective's model.
            c, _loc, scale = fit_gpd_clipped(
                sample, threshold, shape_clip_low, shape_clip_high,
                method=fit_method)
            ensemble[j, :] = genpareto.ppf(quantiles_gpd, c,
                                           loc=threshold, scale=scale)
        except Exception:
            continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gpd_be   = np.nanmean(ensemble[:, aer_gpd_mask],       axis=0)
        gpd_cb90 = np.nanpercentile(ensemble[:, aer_gpd_mask], 90, axis=0)
        gpd_cb10 = np.nanpercentile(ensemble[:, aer_gpd_mask], 10, axis=0)

    return ensemble, gpd_be, gpd_cb10, gpd_cb90, aer_gpd_mask


def assemble_hazard_curve(
    aer_gpd:      np.ndarray,
    gpd_be:       np.ndarray,
    gpd_cb10:     np.ndarray,
    gpd_cb90:     np.ndarray,
    aer_below_th: np.ndarray,
    pot_below_th: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splice the GPD tail and empirical bulk into a single hazard curve.

    Empirical-bulk uncertainty is taken as zero (CB10 = CB90 = best estimate)
    per the v1 convention.
    """
    aer  = np.concatenate([aer_gpd,  aer_below_th])
    be   = np.concatenate([gpd_be,   pot_below_th])
    cb10 = np.concatenate([gpd_cb10, pot_below_th])
    cb90 = np.concatenate([gpd_cb90, pot_below_th])
    return aer, be, cb10, cb90


def interpolate_to_table(
    aer_table: np.ndarray,
    hc_aer:    np.ndarray,
    hc_be:     np.ndarray,
    hc_cb10:   np.ndarray,
    hc_cb90:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the merged hazard curve onto the 22-AER table grid.

    Interpolation is performed in log10(AER) space. Table entries that the
    source curve cannot cover (NaN slots in ``aer_table``) remain NaN.
    """
    log_src  = np.log10(hc_aer)
    valid    = ~np.isnan(aer_table)
    log_tbl  = np.log10(aer_table[valid])

    out_be   = np.full_like(aer_table, np.nan)
    out_cb10 = np.full_like(aer_table, np.nan)
    out_cb90 = np.full_like(aer_table, np.nan)

    out_be  [valid] = np.interp(log_tbl, log_src, hc_be)
    out_cb10[valid] = np.interp(log_tbl, log_src, hc_cb10)
    out_cb90[valid] = np.interp(log_tbl, log_src, hc_cb90)

    return out_be, out_cb10, out_cb90
