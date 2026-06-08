"""Discrete Storm Weight (DSW) back-computation and JPM-OS hazard-curve reconstruction.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Algorithm
---------
Given selected storm surges Y_sub [k x m] and benchmark HCs HC_bench [m x N]:

Step 1 — Nodal DSW back-computation
    At each node i, sort the k surge values in descending order.
    Interpolate HC_bench[i] in log-AER space at each sorted surge → AER values.
    Finite-difference the AER sequence → nodal DSWs (sorted order).
    Map back to original storm order.  Clip negatives to zero.

    A node is considered ACTIVE for averaging only if at least
    `min_wet_storms` of the k selected storms produce a non-dry, non-NaN surge
    AND the interpolation returns at least one finite AER value.
    Nodes where fewer storms are active are excluded from the global average
    (Step 2) to prevent dry-node bias.

Step 2 — Global DSW set
    DSW_global[j] = weighted mean across ACTIVE nodes of nodal DSW for storm j.

Step 3 — HC reconstruction (JPM-OS)
    At each node: sort storms by descending surge, cumsum global DSWs,
    interpolate surge vs cumulative AER onto tbl_aer.

Step 4 — Residual metrics
    resid = HC_recon - HC_bench
    Per-node: bias, uncertainty, rmse.  Scalar = nanmean across nodes.

Engine contract: arrays in, arrays/dict out.  No config, no I/O.

Public API
----------
  compute_global_dsw(Y_sub, HC_bench, tbl_aer, dry_thr, min_wet_storms, method)
      -> ndarray [k]
  reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
      -> ndarray [m x N]
  evaluate_hc_metrics(Y_sub, HC_bench, tbl_aer, dry_thr, min_wet_storms, dsw_method)
      -> dict
  evaluate_hc_reconstruction(Y_sub, HC_bench, tbl_aer, dry_thr, report_aer, dsw_method)
      -> (HC_recon, metrics)
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _surge_to_aer(hc_surge, hc_aer, query):
    """Map surge values → AER by linear interpolation in log-AER space."""
    valid = ~np.isnan(hc_surge)
    if valid.sum() < 2:
        return np.full(len(query), np.nan)
    s = hc_surge[valid]
    a = hc_aer[valid]
    s_uniq, uid = np.unique(s, return_index=True)
    # np.interp is ~10× faster than scipy.interpolate.interp1d for linear
    # interpolation; out-of-bounds queries get NaN to match the previous
    # interp1d(bounds_error=False, fill_value=nan) behavior.
    log_a = np.log(a[uid])
    return np.exp(np.interp(query, s_uniq, log_a, left=np.nan, right=np.nan))


def _jpm_integrate(resp, dsw, tbl_aer, dry_thr):
    """Reconstruct the hazard curve at a single node via JPM-OS integration."""
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
    log_aer_uniq, ia = np.unique(log_aer, return_index=True)
    return np.interp(np.log(tbl_aer), log_aer_uniq, surge[ia],
                     left=np.nan, right=np.nan)


def _hc_residual_metrics(HC_recon, HC_bench):
    resid = HC_recon - HC_bench
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        node_bias = np.nanmean(resid,             axis=1)
        node_unc  = np.nanstd( resid,             axis=1)
        node_rmse = np.sqrt(np.nanmean(resid**2,  axis=1))
    return {
        "mean_bias":        float(np.nanmean(node_bias)),
        "mean_uncertainty": float(np.nanmean(node_unc)),
        "mean_rmse":        float(np.nanmean(node_rmse)),
    }


def _bias_at_aer_levels(
    HC_recon:   np.ndarray,
    HC_bench:   np.ndarray,
    tbl_aer:    np.ndarray,
    report_aer: list,
) -> dict:
    """Mean nodal bias at specific AER hazard levels.

    Each level in ``report_aer`` is labelled by its MRI year N; the AER it
    targets is 1/N (so N=1000 → AER=1e-3), and the result key is ``bias_aer{N}``.
    """
    tbl_aer = np.asarray(tbl_aer)
    result  = {}
    for n in report_aer:                       # n = MRI-year level; AER = 1/n
        col = int(np.argmin(np.abs(tbl_aer - 1.0 / n)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[f"bias_aer{n}"] = float(np.nanmean(HC_recon[:, col] - HC_bench[:, col]))
    return result


def _compute_node_weights(Y_sub, HC_bench, tbl_aer, method, dry_thr):
    """Pre-compute a single scalar weight per node for DSW aggregation."""
    m = Y_sub.shape[1]
    Y_clean = np.where(np.isnan(Y_sub) | (Y_sub <= dry_thr), 0.0, Y_sub)

    if method == 1:
        return np.ones(m, dtype=np.float64)
    elif method == 2:
        return None
    elif method == 3:
        return np.var(Y_clean, axis=0)
    else:
        raise ValueError(f"Unknown dsw_method: {method!r}.  Valid: 1, 2, 3")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_global_dsw(
    Y_sub:          np.ndarray,
    HC_bench:       np.ndarray,
    tbl_aer:        np.ndarray,
    dry_thr:        float = 0.0,
    min_wet_storms: int   = 2,
    method:         int   = 1,
) -> np.ndarray:
    """Back-compute one global DSW per selected storm by aggregating nodal DSWs
    across active nodes only.

    A node is active for averaging when:
      - at least `min_wet_storms` of the k selected surges exceed dry_thr, AND
      - the HC interpolation returns at least one finite AER value.

    method:
      1 = simple mean (equal node weights — classic JPM-OS)
      2 = surge-weighted mean (per-storm-per-node)
      3 = variance-weighted mean (fixed per-node weight = surge variance)
    """
    k, m = Y_sub.shape

    node_w = _compute_node_weights(Y_sub, HC_bench, tbl_aer, method, dry_thr)

    wet_counts = np.sum((~np.isnan(Y_sub)) & (Y_sub > dry_thr), axis=0)

    sort_idx   = np.argsort(Y_sub, axis=0)[::-1]
    sort_idx_T = sort_idx.T

    inv_perm = np.empty((m, k), dtype=int)
    row_idx  = np.arange(m)[:, np.newaxis]
    rank_idx = np.tile(np.arange(k), (m, 1))
    inv_perm[row_idx, sort_idx_T] = rank_idx

    dsw_sum    = np.zeros(k, dtype=np.float64)
    weight_sum = np.zeros(k, dtype=np.float64)

    n_clipped_total = 0

    for node in range(m):
        if wet_counts[node] < min_wet_storms:
            continue

        resp_sorted = Y_sub[sort_idx[:, node], node]
        valid       = (~np.isnan(resp_sorted)) & (resp_sorted > dry_thr)
        n_valid     = int(valid.sum())
        if n_valid < 2:
            continue

        aer_q = _surge_to_aer(HC_bench[node, :], tbl_aer, resp_sorted[valid])
        if np.all(np.isnan(aer_q)):
            continue

        dsw_valid     = np.empty(n_valid)
        dsw_valid[0]  = np.where(np.isnan(aer_q[0]), 0.0, aer_q[0])
        dsw_valid[1:] = np.where(
            np.isnan(aer_q[1:]) | np.isnan(aer_q[:-1]),
            0.0,
            np.diff(aer_q),
        )

        neg = dsw_valid < 0
        if neg.any():
            n_clipped_total += int(neg.sum())
            dsw_valid = np.clip(dsw_valid, 0.0, None)

        dsw_sorted = np.zeros(k, dtype=np.float64)
        valid_pos  = np.where(valid)[0]
        dsw_sorted[valid_pos] = dsw_valid

        dsw_orig = dsw_sorted[inv_perm[node, :]]
        active   = dsw_orig > 0

        if method == 2:
            surge_orig = np.maximum(Y_sub[:, node], 0.0)
            surge_orig = np.where(np.isnan(surge_orig), 0.0, surge_orig)
            dsw_sum    += dsw_orig * surge_orig
            weight_sum += surge_orig * active.astype(np.float64)
        else:
            w = node_w[node]
            dsw_sum    += dsw_orig * w
            weight_sum += w * active.astype(np.float64)

    if n_clipped_total:
        warnings.warn(
            f"{n_clipped_total} negative nodal DSW(s) clipped to 0 across all nodes "
            "(caused by non-monotone benchmark HC segments).",
            RuntimeWarning, stacklevel=2,
        )

    with np.errstate(invalid="ignore"):
        DSW_global = np.where(weight_sum > 0, dsw_sum / weight_sum, np.nan)

    return DSW_global


def reconstruct_hc_global_dsw(
    Y_sub:      np.ndarray,
    DSW_global: np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
) -> np.ndarray:
    """Reconstruct the hazard curve at every node via JPM-OS integration."""
    k, m     = Y_sub.shape
    HC_recon = np.full((m, len(tbl_aer)), np.nan)
    for node in range(m):
        HC_recon[node, :] = _jpm_integrate(
            Y_sub[:, node], DSW_global, tbl_aer, dry_thr)
    return HC_recon


def evaluate_hc_metrics(
    Y_sub:          np.ndarray,
    HC_bench:       np.ndarray,
    tbl_aer:        np.ndarray,
    dry_thr:        float = 0.0,
    min_wet_storms: int   = 2,
    dsw_method:     int   = 1,
) -> dict:
    """Run the full DSW pipeline and return scalar HC quality metrics.

    Returns dict with keys: mean_bias, mean_uncertainty, mean_rmse.
    """
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer, dry_thr,
                                    min_wet_storms, method=dsw_method)
    HC_recon   = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
    return _hc_residual_metrics(HC_recon, HC_bench)


def evaluate_hc_reconstruction(
    Y_sub:          np.ndarray,
    HC_bench:       np.ndarray,
    tbl_aer:        np.ndarray,
    dry_thr:        float = 0.0,
    report_aer:     list  = None,
    dsw_method:     int   = 1,
    min_wet_storms: int   = 2,
) -> tuple:
    """Full DSW + HC reconstruction + global and per-AER-level metrics.

    Returns
    -------
    HC_recon : ndarray [m x N_AER]
    metrics  : dict  (mean_bias, mean_uncertainty, mean_rmse,
                      bias_aer<N> for each N in report_aer)
    """
    if report_aer is None:
        report_aer = []
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer, dry_thr,
                                    min_wet_storms, method=dsw_method)
    HC_recon   = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
    metrics    = {
        **_hc_residual_metrics(HC_recon, HC_bench),
        **_bias_at_aer_levels(HC_recon, HC_bench, tbl_aer, report_aer),
    }
    return HC_recon, metrics
