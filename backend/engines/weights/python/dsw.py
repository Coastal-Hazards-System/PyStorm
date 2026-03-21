"""
backend/engines/weights/dsw.py
================================
Discrete Storm Weight (DSW) back-computation and JPM hazard-curve
reconstruction.

Algorithm (from tc_subset_selection_v3_hdf5.py, Section 6)
-----------------------------------------------------------
Given selected storm surges Y_sub [k x m] and benchmark HCs HC_bench [m x N]:

Step 1 — Nodal DSW back-computation
    Sort k surges at each node in descending order.
    Interpolate HC_bench[node] in log-AER space at each sorted surge.
    Finite-difference AER sequence → nodal DSWs.
    Map back to original storm order.  Clip negatives to zero.

Step 2 — Global DSW set
    DSW_global[j] = nanmean across nodes of nodal DSW for storm j.

Step 3 — HC reconstruction (JPM)
    At each node: sort storms by descending surge, cumsum global DSWs,
    interpolate surge vs cumulative AER onto tbl_aer.

Step 4 — Residual metrics
    resid = HC_recon - HC_bench
    node_bias, node_unc, node_rmse → nanmean across nodes.

Engine contract: arrays in, arrays/dict out.  No config, no I/O.

Developed by: Norberto C. Nadal-Caraballo, PhD

Public API
----------
  compute_global_dsw(Y_sub, HC_bench, tbl_aer)            -> ndarray [k]
  reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer,
                             dry_thr)                      -> ndarray [m x N]
  evaluate_hc_metrics(Y_sub, HC_bench, tbl_aer, dry_thr)  -> dict
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Private helpers  (verbatim from Section 6)
# ---------------------------------------------------------------------------

def _surge_to_aer(hc_surge, hc_aer, query):
    valid = ~np.isnan(hc_surge)
    if valid.sum() < 2:
        return np.full(len(query), np.nan)
    s = hc_surge[valid]; a = hc_aer[valid]
    _, uid = np.unique(s, return_index=True)
    fn = interp1d(s[uid], np.log(a[uid]),
                  kind="linear", bounds_error=False, fill_value=np.nan)
    return np.exp(fn(query))


def _jpm_integrate(resp, dsw, tbl_aer, dry_thr):
    n_aer = len(tbl_aer)
    valid = (~np.isnan(resp)) & (~np.isnan(dsw)) & (resp > dry_thr)
    if valid.sum() < 2:
        return np.full(n_aer, np.nan)
    desc    = np.argsort(resp[valid])[::-1]
    surge   = resp[valid][desc]
    cum_aer = np.cumsum(dsw[valid][desc])
    log_aer = np.log(cum_aer)
    finite  = np.isfinite(log_aer)
    surge   = surge[finite]; log_aer = log_aer[finite]
    if len(surge) < 2:
        return np.full(n_aer, np.nan)
    _, ia = np.unique(log_aer, return_index=True)
    try:
        fn = interp1d(log_aer[ia], surge[ia],
                      kind="linear", bounds_error=False, fill_value=np.nan)
        return fn(np.log(tbl_aer))
    except Exception:
        return np.full(n_aer, np.nan)


def _hc_residual_metrics(HC_recon, HC_bench):
    resid = HC_recon - HC_bench
    with warnings.catch_warnings():  # all-NaN rows (skipped nodes) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        node_bias = np.nanmean(resid,            axis=1)
        node_unc  = np.nanstd( resid,            axis=1)
        node_rmse = np.sqrt(np.nanmean(resid**2, axis=1))
    return {
        "mean_bias":        float(np.nanmean(node_bias)),
        "mean_uncertainty": float(np.nanmean(node_unc)),
        "mean_rmse":        float(np.nanmean(node_rmse)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_global_dsw(
    Y_sub:    np.ndarray,
    HC_bench: np.ndarray,
    tbl_aer:  np.ndarray,
    method:   int = 1,
) -> np.ndarray:
    """
    Back-compute one global DSW per selected storm.

    Parameters
    ----------
    method : int
        1 = simple mean (equal node weights)
        2 = surge-weighted mean (per-storm-per-node weight)
        3 = variance-weighted mean (fixed per-node weight = surge variance)

    Returns
    -------
    DSW_global : [k]  one scalar weight per storm in original order
    """
    from backend.engines.weights.dsw import _compute_node_weights

    k, m = Y_sub.shape
    node_w = _compute_node_weights(Y_sub, HC_bench, tbl_aer, method, dry_thr=0.0)
    DSW_node_orig = np.full((m, k), np.nan)
    sort_idx   = np.argsort(Y_sub, axis=0)[::-1]
    sort_idx_T = sort_idx.T
    inv_perm   = np.empty((m, k), dtype=int)
    row_idx    = np.arange(m)[:, np.newaxis]
    rank_idx   = np.tile(np.arange(k), (m, 1))
    inv_perm[row_idx, sort_idx_T] = rank_idx

    for node in range(m):
        resp_sorted = Y_sub[sort_idx[:, node], node]
        valid       = ~np.isnan(resp_sorted)
        n_valid     = int(valid.sum())
        if n_valid < 2:
            continue
        aer_q = _surge_to_aer(HC_bench[node, :], tbl_aer, resp_sorted[valid])
        if np.all(np.isnan(aer_q)):
            continue
        dsw_sorted    = np.full(k, np.nan)
        dsw_valid     = np.empty(n_valid)
        dsw_valid[0]  = aer_q[0]
        dsw_valid[1:] = np.diff(aer_q)
        neg = dsw_valid < 0
        if neg.any():
            warnings.warn(
                f"Node {node}: {neg.sum()} negative DSW(s) clipped to 0.",
                RuntimeWarning, stacklevel=2,
            )
            dsw_valid = np.clip(dsw_valid, 0.0, None)
        valid_positions             = np.where(valid)[0]
        dsw_sorted[valid_positions] = dsw_valid
        DSW_node_orig[node, :]      = dsw_sorted[inv_perm[node, :]]

    active    = ~np.isnan(DSW_node_orig)
    dsw_clean = np.where(active, DSW_node_orig, 0.0)

    if method == 2:
        # Per-storm surge weight
        surge = np.where(np.isnan(Y_sub.T), 0.0, np.maximum(Y_sub.T, 0.0))  # [m x k]
        weighted_sum = np.sum(dsw_clean * surge * active, axis=0)
        weight_total = np.sum(surge * active, axis=0)
    else:
        # Methods 1, 3a-3e: fixed per-node weight
        w = node_w[:, np.newaxis]  # [m x 1]
        weighted_sum = np.sum(dsw_clean * w * active, axis=0)
        weight_total = np.sum(w * active, axis=0)

    with np.errstate(invalid="ignore"):
        return np.where(weight_total > 0, weighted_sum / weight_total, np.nan)


def reconstruct_hc_global_dsw(
    Y_sub:      np.ndarray,
    DSW_global: np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
) -> np.ndarray:
    """
    Reconstruct hazard curve at every node via JPM integration.

    Returns
    -------
    HC_recon : [m x N_AER]
    """
    k, m = Y_sub.shape
    HC_recon = np.full((m, len(tbl_aer)), np.nan)
    for node in range(m):
        HC_recon[node, :] = _jpm_integrate(
            Y_sub[:, node], DSW_global, tbl_aer, dry_thr)
    return HC_recon


def evaluate_hc_metrics(
    Y_sub:      np.ndarray,
    HC_bench:   np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
    dsw_method: int = 1,
) -> dict:
    """
    Run the full DSW pipeline (Steps 1-4) and return scalar HC quality metrics.

    Returns dict with keys: mean_bias, mean_uncertainty, mean_rmse
    """
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer, method=dsw_method)
    HC_recon   = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
    return _hc_residual_metrics(HC_recon, HC_bench)


def _bias_at_return_periods(
    HC_recon:  np.ndarray,
    HC_bench:  np.ndarray,
    tbl_aer:   np.ndarray,
    report_rp: list,
) -> dict:
    """
    Mean nodal bias at specific return period levels.

    Parameters
    ----------
    report_rp : list of ints  (e.g. [10, 100, 1000])
        Return periods in years.  The closest AER column is used.

    Returns dict with keys: bias_rp10, bias_rp100, bias_rp1000, ...
    """
    tbl_aer = np.asarray(tbl_aer)
    result  = {}
    for rp in report_rp:
        col = int(np.argmin(np.abs(tbl_aer - 1.0 / rp)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[f"bias_rp{rp}"] = float(np.nanmean(HC_recon[:, col] - HC_bench[:, col]))
    return result


def evaluate_hc_reconstruction(
    Y_sub:      np.ndarray,
    HC_bench:   np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
    report_rp:  list  = None,
    dsw_method: int = 1,
) -> tuple:
    """
    Full DSW + HC reconstruction + global and per-return-period metrics.

    Returns
    -------
    HC_recon : ndarray [m x N_AER]
    metrics  : dict  (mean_bias, mean_uncertainty, mean_rmse,
                      bias_rp<N> for each N in report_rp)
    """
    if report_rp is None:
        report_rp = []
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer, method=dsw_method)
    HC_recon   = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
    metrics    = {
        **_hc_residual_metrics(HC_recon, HC_bench),
        **_bias_at_return_periods(HC_recon, HC_bench, tbl_aer, report_rp),
    }
    return HC_recon, metrics
