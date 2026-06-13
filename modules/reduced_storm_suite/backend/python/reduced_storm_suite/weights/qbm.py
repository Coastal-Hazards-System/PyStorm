"""Quantile Bias Mapping (QBM) -- post-DSW bias correction.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Two correction modes, selected by *qbm_mode*:

  "aer"      (default) -- AER-based correction.  Surge values are physical
             model outputs and remain untouched.  Only the cumulative AER
             positions are remapped via the inverse benchmark HC so that the
             corrected points land on the benchmark curve horizontally.

  "response" -- Response-based correction (legacy).  Surge values are shifted
             vertically to match the benchmark HC at each storm's cum-AER
             position.  Alters model outputs; retained for comparison only.

Both modes share the same smoothing infrastructure (intermediate AER grid,
Gaussian kernel, endpoint ramps).

Intermediate grid (user's choice via aer_mode):
  "631"      (default) -- dense 631-point AER grid (10^1 ... 10^-6).
  "standard" (22 AERs) -- the 22 tbl_aer grid.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 631-AER dense grid
# ---------------------------------------------------------------------------

def build_aer_631() -> np.ndarray:
    """Build the dense 631-point AER grid from 10^1 to 10^-6."""
    d = 1.0 / 90.0
    exponents = np.arange(1.0, -d / 2, -d)
    v = 10.0 ** exponents
    parts = [v]
    x = 10.0
    for _ in range(6):
        parts.append(v[1:] / x)
        x *= 10.0
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _interp_log_linear(x_known, y_known, x_query, extrapolate=True):
    """Log-linear interpolation."""
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


def _enforce_aer_monotonicity(aer: np.ndarray) -> np.ndarray:
    """Enforce non-decreasing corrected AER (cumulative AER must grow)."""
    out = aer.copy()
    for j in range(1, len(out)):
        if out[j] < out[j - 1]:
            out[j] = out[j - 1]
    return out


# ---------------------------------------------------------------------------
# Inverse benchmark lookup (AER mode)
# ---------------------------------------------------------------------------

def _bench_inverse_aer(HC_bench_node, tbl_aer, surge_query):
    """Inverse benchmark HC: given surge values, return the corresponding AER."""
    valid = np.isfinite(HC_bench_node) & (HC_bench_node > 0)
    if valid.sum() < 2:
        return np.full_like(surge_query, np.nan, dtype=np.float64)

    surge_bench = HC_bench_node[valid]
    log_aer_bench = np.log(tbl_aer[valid])

    order = np.argsort(surge_bench)
    surge_sorted = surge_bench[order]
    log_aer_sorted = log_aer_bench[order]

    fn = interp1d(surge_sorted, log_aer_sorted,
                  kind="linear", bounds_error=False,
                  fill_value=np.nan)
    result = fn(surge_query)
    with np.errstate(invalid="ignore"):
        return np.exp(result)


# ---------------------------------------------------------------------------
# Per-node storm-level bias
# ---------------------------------------------------------------------------

def _node_storm_bias_response(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """Raw response bias at per-storm AER positions for a single node."""
    valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
    if valid.sum() < 2:
        return None, None, None, None

    desc = np.argsort(resp[valid])[::-1]
    surge_s = resp[valid][desc]
    cum_aer_s = np.cumsum(DSW_global[valid][desc])

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


def _node_storm_bias_aer(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """Raw log-AER bias at per-storm AER positions for a single node."""
    valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
    if valid.sum() < 2:
        return None, None, None, None

    desc = np.argsort(resp[valid])[::-1]
    surge_s = resp[valid][desc]
    cum_aer_s = np.cumsum(DSW_global[valid][desc])

    aer_corrected = _bench_inverse_aer(HC_bench_node, tbl_aer, surge_s)

    overlap = np.isfinite(aer_corrected) & (aer_corrected > 0) & (cum_aer_s > 0)
    if overlap.sum() < 2:
        return cum_aer_s, surge_s, None, None

    bias_raw = np.log(cum_aer_s[overlap]) - np.log(aer_corrected[overlap])
    return cum_aer_s, surge_s, bias_raw, overlap


# ---------------------------------------------------------------------------
# Shared smoothing infrastructure
# ---------------------------------------------------------------------------

def _bias_via_intermediate_grid(cum_aer_s, overlap, bias_raw,
                                inter_grid, win_frac, ramp_frac):
    """Map per-storm raw bias to an intermediate AER grid, smooth there."""
    bias_inter = _interp_log_linear(
        cum_aer_s[overlap], bias_raw, inter_grid, extrapolate=True)

    valid_n = np.isfinite(bias_inter)
    if valid_n.sum() >= 3:
        bias_raw_copy = bias_inter.copy()
        bias_inter[valid_n] = _gaussian_smooth(
            bias_inter[valid_n], win_frac)
        bias_inter[valid_n] = _ramp_endpoints(
            bias_inter[valid_n], bias_raw_copy[valid_n], ramp_frac)

    return bias_inter


# ---------------------------------------------------------------------------
# Public API
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
    qbm_mode:   str   = "aer",
) -> np.ndarray:
    """Compute QBM bias at the standard tbl_aer grid for every node."""
    k, m = Y_sub.shape
    n_aer = len(tbl_aer)
    bias_tbl = np.zeros((m, n_aer), dtype=np.float64)

    if aer_mode == "631":
        inter_grid = _get_aer_631()
    else:
        inter_grid = tbl_aer

    bias_fn = (_node_storm_bias_aer if qbm_mode == "aer"
               else _node_storm_bias_response)

    for node in range(m):
        cum_aer_s, surge_s, bias_raw, overlap = bias_fn(
            Y_sub[:, node], DSW_global, HC_bench[node, :], tbl_aer, dry_thr)

        if bias_raw is None:
            continue

        bias_inter = _bias_via_intermediate_grid(
            cum_aer_s, overlap, bias_raw, inter_grid, win_frac, ramp_frac)

        if aer_mode == "631":
            bias_node = _interp_log_linear(
                inter_grid, bias_inter, tbl_aer, extrapolate=True)
        else:
            bias_node = bias_inter

        bias_tbl[node, :] = bias_node

    return bias_tbl


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
    qbm_mode:       str   = "aer",
):
    """Correct a single node's storm data using QBM bias."""
    bias_fn = (_node_storm_bias_aer if qbm_mode == "aer"
               else _node_storm_bias_response)

    cum_aer_s, surge_s, bias_raw, overlap = bias_fn(
        resp, DSW_global, HC_bench_node, tbl_aer, dry_thr)

    if cum_aer_s is None:
        return None, None

    if bias_raw is None:
        return cum_aer_s, surge_s.copy()

    if aer_mode == "631":
        inter_grid = _get_aer_631()
    else:
        inter_grid = tbl_aer

    bias_inter = _bias_via_intermediate_grid(
        cum_aer_s, overlap, bias_raw, inter_grid, win_frac, ramp_frac)

    bias_at_storm = _interp_log_linear(
        inter_grid, bias_inter, cum_aer_s, extrapolate=True)

    if qbm_mode == "aer":
        log_aer_corr = np.log(cum_aer_s) - bias_at_storm
        aer_corr = np.exp(log_aer_corr)
        aer_corr = _enforce_aer_monotonicity(aer_corr)
        return aer_corr, surge_s
    else:
        surge_corr = surge_s - bias_at_storm
        surge_corr = _enforce_monotonicity(surge_corr)
        return cum_aer_s, surge_corr
