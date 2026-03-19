"""
backend/engines/weights/qbm.py
================================
Quantile Bias Mapping (QBM) — post-DSW bias correction.

Algorithm (per node)
--------------------
1. From selected storms and their global DSWs, build per-storm
   cumulative AER positions (sort descending surge, cumsum DSW).

2. For each red circle at cum_aer_j, look up the benchmark HC value
   at that EXACT AER (log-linear interpolation within the benchmark
   range only — NO extrapolation).  If cum_aer_j is outside the
   benchmark's AER range, bias is zero for that circle.

3. Raw bias at each overlapping storm = storm_surge - bench_surge.
   If there is no overlap at all, bias is zero for every circle.

4. Intermediate grid (user's choice via aer_mode):
     "631"      (default) — map per-storm bias to a dense 631-point
                  AER grid (10^1 … 10^-6, d=1/90 in log10 space).
     "standard" (22 AERs) — map per-storm bias to the 22 tbl_aer grid.

   Smoothing is applied on the intermediate grid.

5. Map intermediate-grid bias back to per-storm AER positions to
   correct the red circles (green circles).

6. Corrected green circles are mapped to the 22 tbl_aer grid
   for final output (regardless of intermediate grid choice).

Public API
----------
  build_aer_631()  -> ndarray [631]

  compute_qbm_bias(Y_sub, DSW_global, HC_bench, tbl_aer, ...)
      -> bias_22  [m x N_AER]

  correct_node_qbm(resp, DSW_global, HC_bench_node, bias_22_node,
                    tbl_aer, dry_thr, aer_mode)
      -> (cum_aer, surge_corrected)   — for plotting green circles
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 631-AER dense grid
# ---------------------------------------------------------------------------

def build_aer_631() -> np.ndarray:
    """
    Build the dense 631-point AER grid from 10^1 to 10^-6.

    Equivalent MATLAB:
        d=1/90; v=10.^(1:-d:0)'; plt_aef=v; x=10;
        for i=1:6
            plt_aef=[plt_aef;v(2:end)/x];
            x=x*10;
        end
    """
    d = 1.0 / 90.0
    exponents = np.arange(1.0, -d / 2, -d)  # 1, 1-d, 1-2d, ..., 0
    v = 10.0 ** exponents               # 91 values: 10 … 1
    parts = [v]
    x = 10.0
    for _ in range(6):
        parts.append(v[1:] / x)         # 90 values each
        x *= 10.0
    return np.concatenate(parts)         # 91 + 6*90 = 631


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _interp_log_linear(x_known, y_known, x_query, extrapolate=True):
    """
    Log-linear interpolation.

    Parameters
    ----------
    extrapolate : bool
        True  = nearest-neighbor extrapolation outside range.
        False = NaN outside range (no extrapolation).
    """
    log_x = np.log(x_known)
    order = np.argsort(log_x)
    log_x = log_x[order]
    y_sorted = y_known[order]

    if extrapolate:
        fill = (y_sorted[0], y_sorted[-1])
    else:
        fill = np.nan

    fn = interp1d(log_x, y_sorted,
                  kind="linear", bounds_error=False,
                  fill_value=fill)
    return fn(np.log(x_query))


def _gaussian_smooth(bias_raw: np.ndarray, win_frac: float = 0.10) -> np.ndarray:
    """Gaussian kernel smoothing in index space."""
    N = len(bias_raw)
    if N < 3:
        return bias_raw.copy()

    sigma = (win_frac * N) / 6.0
    sigma = max(sigma, 0.5)

    idx = np.arange(N, dtype=np.float64)
    bias_smooth = np.empty(N)

    for i in range(N):
        w = np.exp(-0.5 * ((idx - i) / sigma) ** 2)
        w /= w.sum()
        bias_smooth[i] = np.dot(w, bias_raw)

    return bias_smooth


def _ramp_endpoints(
    bias_smooth: np.ndarray,
    bias_raw: np.ndarray,
    ramp_frac: float = 0.03,
) -> np.ndarray:
    """Smoothstep (C1) ramp to exact raw values at both endpoints."""
    N = len(bias_smooth)
    m = max(3, round(ramp_frac * N))
    m = min(m, N // 2)

    out = bias_smooth.copy()
    t = np.linspace(0, 1, m)
    a = 3 * t**2 - 2 * t**3

    out[:m] = (1 - a) * bias_raw[0] + a * bias_smooth[:m]
    out[0] = bias_raw[0]

    out[-m:] = (1 - a) * bias_smooth[-m:] + a * bias_raw[-1]
    out[-1] = bias_raw[-1]

    return out


def _enforce_monotonicity(surge: np.ndarray) -> np.ndarray:
    """Enforce non-increasing surge (higher response at lower AER)."""
    out = surge.copy()
    for j in range(1, len(out)):
        if out[j] > out[j - 1]:
            out[j] = out[j - 1]
    return out


# ---------------------------------------------------------------------------
# Per-node storm-level bias computation (shared by both modes)
# ---------------------------------------------------------------------------

def _node_storm_bias(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """
    Compute raw bias at per-storm AER positions for a single node.

    Returns
    -------
    cum_aer_s    : cumulative AER at each storm (descending surge order)
    surge_s      : surge values (descending)
    bias_raw     : raw bias at overlapping positions (len = overlap.sum())
    overlap_mask : bool mask over cum_aer_s — True where benchmark covers
    None, None, None, None  when fewer than 2 valid storms
    """
    valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
    if valid.sum() < 2:
        return None, None, None, None

    desc = np.argsort(resp[valid])[::-1]
    surge_s = resp[valid][desc]
    cum_aer_s = np.cumsum(DSW_global[valid][desc])

    # Benchmark surge at each storm's cum_aer — NO extrapolation
    valid_bench = np.isfinite(HC_bench_node)
    if valid_bench.sum() < 2:
        return cum_aer_s, surge_s, None, None

    bench_surge_at_storm = _interp_log_linear(
        tbl_aer[valid_bench], HC_bench_node[valid_bench],
        cum_aer_s, extrapolate=False)

    overlap = np.isfinite(bench_surge_at_storm)
    if overlap.sum() < 2:
        return cum_aer_s, surge_s, None, None

    bias_raw = surge_s[overlap] - bench_surge_at_storm[overlap]
    return cum_aer_s, surge_s, bias_raw, overlap


def _bias_via_intermediate_grid(cum_aer_s, overlap, bias_raw,
                                inter_grid, win_frac, ramp_frac):
    """
    Map per-storm raw bias to an intermediate AER grid, smooth there,
    and return the smoothed bias on the intermediate grid.

    Returns
    -------
    bias_inter : ndarray [len(inter_grid)]
    """
    # Map raw per-storm bias to intermediate grid
    bias_inter = _interp_log_linear(
        cum_aer_s[overlap], bias_raw, inter_grid, extrapolate=True)

    # Smooth on the intermediate grid
    valid_n = np.isfinite(bias_inter)
    if valid_n.sum() >= 3:
        bias_raw_copy = bias_inter.copy()
        bias_inter[valid_n] = _gaussian_smooth(
            bias_inter[valid_n], win_frac)
        bias_inter[valid_n] = _ramp_endpoints(
            bias_inter[valid_n], bias_raw_copy[valid_n], ramp_frac)

    return bias_inter


# ---------------------------------------------------------------------------
# Public API: compute bias at 22 standard AERs (for storage)
# ---------------------------------------------------------------------------

_AER_631_CACHE = None


def _get_aer_631():
    global _AER_631_CACHE
    if _AER_631_CACHE is None:
        _AER_631_CACHE = build_aer_631()
    return _AER_631_CACHE


def compute_qbm_bias(
    Y_sub:      np.ndarray,
    DSW_global: np.ndarray,
    HC_bench:   np.ndarray,
    tbl_aer:    np.ndarray,
    dry_thr:    float = 0.0,
    win_frac:   float = 0.10,
    ramp_frac:  float = 0.03,
    aer_mode:   str   = "631",
) -> np.ndarray:
    """
    Compute QBM bias at the standard tbl_aer grid for every node.

    Both modes follow the same pipeline:
      per-storm bias → map to intermediate grid → smooth → map to 22 AERs

    Parameters
    ----------
    aer_mode : str
        "631"      (default) — intermediate grid is a dense 631-point
                    AER grid (10^1 … 10^-6, d=1/90 in log10 space).
        "standard" — intermediate grid is the 22 tbl_aer grid.

    Returns
    -------
    bias_tbl : [m x N_AER]  bias at the standard 22-AER grid per node
    """
    k, m = Y_sub.shape
    n_aer = len(tbl_aer)
    bias_tbl = np.zeros((m, n_aer), dtype=np.float64)

    if aer_mode == "631":
        inter_grid = _get_aer_631()
    else:
        inter_grid = tbl_aer

    for node in range(m):
        cum_aer_s, surge_s, bias_raw, overlap = _node_storm_bias(
            Y_sub[:, node], DSW_global, HC_bench[node, :], tbl_aer, dry_thr)

        if bias_raw is None:
            continue

        # Map per-storm bias to intermediate grid and smooth
        bias_inter = _bias_via_intermediate_grid(
            cum_aer_s, overlap, bias_raw, inter_grid, win_frac, ramp_frac)

        if aer_mode == "631":
            # Map from 631 intermediate grid to 22 standard AERs
            bias_node = _interp_log_linear(
                inter_grid, bias_inter, tbl_aer, extrapolate=True)
        else:
            # Already on the 22 standard AERs
            bias_node = bias_inter

        bias_tbl[node, :] = bias_node

    return bias_tbl


# ---------------------------------------------------------------------------
# Public API: correct a single node's storm responses for plotting
# ---------------------------------------------------------------------------

def correct_node_qbm(
    resp:           np.ndarray,
    DSW_global:     np.ndarray,
    HC_bench_node:  np.ndarray,
    bias_22_node:   np.ndarray,
    tbl_aer:        np.ndarray,
    dry_thr:        float = 0.0,
    aer_mode:       str   = "631",
    win_frac:       float = 0.10,
    ramp_frac:      float = 0.03,
):
    """
    Correct a single node's storm responses using QBM bias.

    Both modes: per-storm bias → map to intermediate grid → smooth →
    map back to per-storm positions → correct.

    Parameters
    ----------
    resp          : [k] storm responses at this node
    DSW_global    : [k] global DSW weights
    HC_bench_node : [N_AER] benchmark HC at this node
    bias_22_node  : [N_AER] stored bias at 22 AERs (unused for "631" mode,
                    used as fallback; "standard" mode uses this directly)
    tbl_aer       : [N_AER] standard AER levels
    dry_thr       : dry threshold
    aer_mode      : "631" or "standard"
    win_frac      : smoothing window fraction
    ramp_frac     : endpoint ramp fraction

    Returns
    -------
    cum_aer       : [n_valid] cumulative AER positions (descending surge)
    surge_corr    : [n_valid] corrected surge values (green circles)

    Returns (None, None) if correction is not possible.
    """
    cum_aer_s, surge_s, bias_raw, overlap = _node_storm_bias(
        resp, DSW_global, HC_bench_node, tbl_aer, dry_thr)

    if cum_aer_s is None:
        return None, None

    if bias_raw is None:
        # No overlap → no correction
        return cum_aer_s, surge_s.copy()

    if aer_mode == "631":
        inter_grid = _get_aer_631()
    else:
        inter_grid = tbl_aer

    # Map per-storm bias to intermediate grid and smooth
    bias_inter = _bias_via_intermediate_grid(
        cum_aer_s, overlap, bias_raw, inter_grid, win_frac, ramp_frac)

    # Map smoothed bias from intermediate grid back to per-storm positions
    bias_at_storm = _interp_log_linear(
        inter_grid, bias_inter, cum_aer_s, extrapolate=True)

    surge_corr = surge_s - bias_at_storm

    # Enforce monotonicity: non-increasing with increasing cum_aer
    surge_corr = _enforce_monotonicity(surge_corr)

    return cum_aer_s, surge_corr
