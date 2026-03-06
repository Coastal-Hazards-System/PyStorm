"""
backend/engines/weights/dsw.py
================================
Discrete Storm Weight (DSW) back-computation and JPM-OS hazard-curve
reconstruction.

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
    DSW_global[j] = nanmean across ACTIVE nodes of nodal DSW for storm j.

Step 3 — HC reconstruction (JPM-OS)
    At each node: sort storms by descending surge, cumsum global DSWs,
    interpolate surge vs cumulative AER onto tbl_aer.

Step 4 — Residual metrics
    resid = HC_recon - HC_bench
    Per-node: bias, uncertainty, rmse.  Scalar = nanmean across nodes.

Engine contract: arrays in, arrays/dict out.  No config, no I/O.

Public API
----------
  compute_global_dsw(Y_sub, HC_bench, tbl_aer, dry_thr, min_wet_storms)
      -> ndarray [k]
  reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
      -> ndarray [m x N]
  evaluate_hc_metrics(Y_sub, HC_bench, tbl_aer, dry_thr, min_wet_storms)
      -> dict
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _surge_to_aer(hc_surge, hc_aer, query):
    """
    Map surge values → AER by linear interpolation in log-AER space.

    hc_surge : [N_AER]  benchmark surge values at this node (the x-axis of HC)
    hc_aer   : [N_AER]  corresponding AER levels (events/year)
    query    : [q]      surge values to map to AER

    Returns [q] AER values (NaN for out-of-range surges).
    """
    valid = ~np.isnan(hc_surge)
    if valid.sum() < 2:
        return np.full(len(query), np.nan)
    s = hc_surge[valid]
    a = hc_aer[valid]
    _, uid = np.unique(s, return_index=True)
    fn = interp1d(s[uid], np.log(a[uid]),
                  kind="linear", bounds_error=False, fill_value=np.nan)
    return np.exp(fn(query))


def _jpm_integrate(resp, dsw, tbl_aer, dry_thr):
    """
    Reconstruct the hazard curve at a single node via JPM-OS integration.

    resp    : [k]      surge responses for the selected storms
    dsw     : [k]      global DSW per storm
    tbl_aer : [N_AER]  output AER levels
    dry_thr : float    surges <= this are treated as dry
    """
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
    resid     = HC_recon - HC_bench
    node_bias = np.nanmean(resid,             axis=1)
    node_unc  = np.nanstd( resid,             axis=1)
    node_rmse = np.sqrt(np.nanmean(resid**2,  axis=1))
    return {
        "mean_bias":        float(np.nanmean(node_bias)),
        "mean_uncertainty": float(np.nanmean(node_unc)),
        "mean_rmse":        float(np.nanmean(node_rmse)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_global_dsw(
    Y_sub:          np.ndarray,
    HC_bench:       np.ndarray,
    tbl_aer:        np.ndarray,
    dry_thr:        float = 0.0,
    min_wet_storms: int   = 2,
) -> np.ndarray:
    """
    Back-compute one global DSW per selected storm by averaging nodal DSWs
    across active nodes only.

    A node is active for averaging when:
      - at least `min_wet_storms` of the k selected surges exceed dry_thr, AND
      - the HC interpolation returns at least one finite AER value.

    This prevents dry-node bias: coastal meshes have large areas where most
    storms produce zero surge, and including those nodes in the nanmean dilutes
    the DSW signal and introduces systematic positive bias in HC reconstruction.

    Parameters
    ----------
    Y_sub          : [k x m]  surge responses for k selected storms at m nodes
    HC_bench       : [m x N]  benchmark hazard curve table
    tbl_aer        : [N]      AER levels (events/year)
    dry_thr        : float    surges <= this treated as dry (m)
    min_wet_storms : int      minimum wet storms required at a node for it to
                              contribute to the global DSW average (default 2)

    Returns
    -------
    DSW_global : [k]  one scalar weight per storm in original order
    """
    k, m = Y_sub.shape

    # Pre-compute wet-storm count per node to gate node inclusion
    wet_counts = np.sum((~np.isnan(Y_sub)) & (Y_sub > dry_thr), axis=0)  # [m]

    # Sort indices descending by surge at each node: sort_idx [k x m]
    sort_idx   = np.argsort(Y_sub, axis=0)[::-1]
    sort_idx_T = sort_idx.T                          # [m x k]

    # Inverse permutation: inv_perm[node, orig_storm] = rank at that node
    inv_perm = np.empty((m, k), dtype=int)
    row_idx  = np.arange(m)[:, np.newaxis]
    rank_idx = np.tile(np.arange(k), (m, 1))
    inv_perm[row_idx, sort_idx_T] = rank_idx

    # Accumulate nodal DSWs — only for active nodes
    # Use a running sum + count to compute nanmean without storing [m x k]
    dsw_sum   = np.zeros(k, dtype=np.float64)
    node_count = np.zeros(k, dtype=np.float64)   # per-storm active node count

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

        # Map valid DSWs back to original storm positions
        dsw_sorted = np.zeros(k, dtype=np.float64)   # zeros for dry/NaN storms
        valid_pos  = np.where(valid)[0]
        dsw_sorted[valid_pos] = dsw_valid

        # Map from sorted → original storm order and accumulate
        dsw_orig   = dsw_sorted[inv_perm[node, :]]
        active     = dsw_orig > 0                     # storms with non-zero DSW
        dsw_sum   += dsw_orig
        node_count += active.astype(np.float64)

    if n_clipped_total:
        warnings.warn(
            f"{n_clipped_total} negative nodal DSW(s) clipped to 0 across all nodes "
            "(caused by non-monotone benchmark HC segments).",
            RuntimeWarning, stacklevel=2,
        )

    # Global DSW: mean over active nodes per storm (NaN where no node contributed)
    with np.errstate(invalid="ignore"):
        DSW_global = np.where(node_count > 0, dsw_sum / node_count, np.nan)

    return DSW_global


def reconstruct_hc_global_dsw(
    Y_sub:      np.ndarray,
    DSW_global: np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
) -> np.ndarray:
    """
    Reconstruct the hazard curve at every node via JPM-OS integration.

    Returns
    -------
    HC_recon : [m x N_AER]
    """
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
) -> dict:
    """
    Run the full DSW pipeline and return scalar HC quality metrics.

    Returns dict with keys: mean_bias, mean_uncertainty, mean_rmse
    """
    DSW_global = compute_global_dsw(Y_sub, HC_bench, tbl_aer, dry_thr, min_wet_storms)
    HC_recon   = reconstruct_hc_global_dsw(Y_sub, DSW_global, tbl_aer, dry_thr)
    return _hc_residual_metrics(HC_recon, HC_bench)
