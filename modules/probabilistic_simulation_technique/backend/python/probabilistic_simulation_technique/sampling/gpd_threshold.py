"""gpd_threshold — Quantile-Delta-Method (QDM) GPD-threshold search.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Picks a Generalized-Pareto threshold by minimizing a frequency-weighted
mean-square error (WMSE) between Weibull-plotting-position empirical AEFs
and the GPD ICDF predictions, scanned across a candidate grid in the
configured percentile band.

A 5% tolerance on the normalized WMSE selects the lowest-threshold candidate
within that tolerance to favour data-rich fits, matching the v1 behaviour.

Public API
----------
  select_gpd_threshold_qdm(
      values_pot, weibull_aef, lambda_val,
      min_percentile=20, max_percentile=80, n_candidates=50,
  ) -> (best_threshold, wmse_all, candidate_thresholds)
"""

import warnings
from typing import Tuple

import numpy as np
from scipy.stats import genpareto


def select_gpd_threshold_qdm(
    values_pot:     np.ndarray,
    weibull_aef:    np.ndarray,
    lambda_val:     float,
    min_percentile: float = 20.0,
    max_percentile: float = 80.0,
    n_candidates:   int   = 50,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (best threshold, WMSE per candidate, candidate threshold grid).

    Parameters
    ----------
    values_pot : (n,) float64
        Descending-sorted POT magnitudes.
    weibull_aef : (n,) float64
        Empirical AEFs at each POT value (Weibull plotting positions scaled
        by the sample intensity).
    lambda_val : float
        Sample intensity ``len(values_pot) / record_length_years``.
    min_percentile, max_percentile : float
        Bounds of the candidate-threshold scan, expressed as percentiles of
        the POT range. Must satisfy 0 <= min < max <= 100.
    n_candidates : int
        Number of threshold candidates uniformly spaced in the band.
    """
    if values_pot.size == 0:
        raise ValueError("values_pot is empty")
    if not (0.0 <= min_percentile < max_percentile <= 100.0):
        raise ValueError("require 0 <= min_percentile < max_percentile <= 100")
    if n_candidates <= 1:
        raise ValueError("n_candidates must be > 1")

    resp_min, resp_max = float(np.min(values_pot)), float(np.max(values_pot))
    resp_range         = resp_max - resp_min

    candidate_thresholds = np.round(
        np.linspace(
            resp_min + 0.01 * min_percentile * resp_range,
            resp_min + 0.01 * max_percentile * resp_range,
            int(n_candidates),
        ),
        2,
    )

    wmse_all = np.full_like(candidate_thresholds, np.nan, dtype=np.float64)

    for i, th in enumerate(candidate_thresholds):
        mask_above = values_pot > th
        pot        = values_pot[mask_above]
        aef        = weibull_aef[mask_above]
        if len(np.unique(pot)) <= 1:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                c, _loc, scale = genpareto.fit(pot, floc=th)
                pred = genpareto.ppf(1.0 - aef / len(pot) * lambda_val,
                                     c, loc=0.0, scale=scale)
            weights_mask = aef < 1.0
            if not np.any(weights_mask):
                continue
            weights         = 1.0 / aef[weights_mask]
            squared_errors  = (pot[weights_mask] - pred[weights_mask]) ** 2
            wmse_all[i]     = float(np.sum(weights * squared_errors)
                                    / np.sum(weights))
        except Exception:
            wmse_all[i] = np.nan

    finite = np.isfinite(wmse_all)
    if np.sum(finite) > 1:
        wmin = np.nanmin(wmse_all)
        wmax = np.nanmax(wmse_all)
        norm = (wmse_all - wmin) / (wmax - wmin) if wmax > wmin else wmse_all * 0
        below_tol = np.where(norm < 0.05)[0]
        best_idx  = int(below_tol[0]) if below_tol.size else int(np.nanargmin(wmse_all))
    elif np.any(finite):
        best_idx = int(np.nanargmin(wmse_all))
    else:
        raise RuntimeError(
            "GPD threshold search failed: no candidate produced a finite WMSE"
        )

    return float(candidate_thresholds[best_idx]), wmse_all, candidate_thresholds
