"""
backend/engines/weights/qbm.py
================================
Quantile Bias Mapping (QBM) — post-DSW bias correction.

Two correction modes
--------------------
**qbm_mode="aer"** (default) — AER-based correction.
    The global-DSW bias is a horizontal displacement in (AER, surge) space.
    For each storm at each node, the corrected AER is obtained by inverting
    the benchmark hazard curve:  lambda_corrected = H_bench^{-1}(surge).
    Surge values are untouched; only cumulative AER positions are remapped.

**qbm_mode="response"** — Response-based correction (legacy).
    Computes b_j = surge_j - H_bench(lambda_j) and subtracts the bias from
    the surge values.  Produces corrected surge values that fall near the
    benchmark curve, but alters the physical model output and corrects along
    the wrong axis.  Retained for backward compatibility and comparison.

Intermediate AER grid (applies to response mode only)
------------------------------------------------------
    "631"      (default) — dense 631-point grid (10^1 … 10^-6, d=1/90).
    "standard" — 22 tbl_aer levels.

Public API
----------
  build_aer_631()  -> ndarray [631]

  compute_qbm_bias(Y_sub, DSW_global, HC_bench, tbl_aer, ..., qbm_mode)
      -> bias_tbl  [m x N_AER]
         qbm_mode="response": response bias (surge units)
         qbm_mode="aer":      log-AER delta (dimensionless, diagnostic)

  correct_node_qbm(resp, DSW_global, HC_bench_node, bias_22_node,
                    tbl_aer, ..., qbm_mode)
      -> (cum_aer, surge)
         qbm_mode="response": (cum_aer_original, surge_corrected)
         qbm_mode="aer":      (cum_aer_corrected, surge_original)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# C++ backend (optional)
# ---------------------------------------------------------------------------

try:
    from backend.engines.weights.cpp._dsw_cpp import (
        compute_qbm_bias_aer      as _cpp_compute_qbm_bias_aer,
        compute_qbm_bias_response as _cpp_compute_qbm_bias_response,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


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
    Log-linear interpolation (log in x, linear in y).

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


def _invert_hc_bench(hc_surge, tbl_aer, query_surge, extrapolate=False):
    """
    Invert the benchmark hazard curve: given surge values, return AER.

    Interpolates in (surge, log-AER) space — surge is the independent
    variable, log(AER) is the dependent variable.

    Parameters
    ----------
    hc_surge     : [N_AER] benchmark surge values at standard AER levels
    tbl_aer      : [N_AER] standard AER levels (descending)
    query_surge  : [k] surge values to look up
    extrapolate  : bool — if False (default), returns NaN outside
                   the benchmark surge range

    Returns
    -------
    aer_at_surge : [k] AER values corresponding to each query surge
    """
    valid = np.isfinite(hc_surge)
    if valid.sum() < 2:
        return np.full_like(query_surge, np.nan)

    surge_v = hc_surge[valid]
    log_aer_v = np.log(tbl_aer[valid])

    # Sort by surge (ascending) for interp1d
    order = np.argsort(surge_v)
    surge_sorted = surge_v[order]
    log_aer_sorted = log_aer_v[order]

    # Remove duplicate surge values (keep first occurrence)
    _, uniq_idx = np.unique(surge_sorted, return_index=True)
    surge_sorted = surge_sorted[uniq_idx]
    log_aer_sorted = log_aer_sorted[uniq_idx]

    if len(surge_sorted) < 2:
        return np.full_like(query_surge, np.nan)

    if extrapolate:
        fill = (log_aer_sorted[0], log_aer_sorted[-1])
    else:
        fill = np.nan

    fn = interp1d(surge_sorted, log_aer_sorted,
                  kind="linear", bounds_error=False,
                  fill_value=fill)
    return np.exp(fn(query_surge))


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


def _enforce_monotonicity_aer(aer: np.ndarray) -> np.ndarray:
    """Enforce non-decreasing cumulative AER (must increase with index)."""
    out = aer.copy()
    for j in range(1, len(out)):
        if out[j] < out[j - 1]:
            out[j] = out[j - 1]
    return out


# ---------------------------------------------------------------------------
# Per-node helpers: response-based bias (legacy)
# ---------------------------------------------------------------------------

def _node_storm_bias_response(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """
    Compute raw response bias at per-storm AER positions for a single node.

    Returns
    -------
    cum_aer_s    : cumulative AER at each storm (descending surge order)
    surge_s      : surge values (descending)
    bias_raw     : raw response bias at overlapping positions
    overlap_mask : bool mask over cum_aer_s
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
    """
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
# Per-node helpers: AER-based correction
# ---------------------------------------------------------------------------

def _node_aer_correction(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """
    Compute AER-based correction at a single node.

    For each storm (sorted descending by surge):
      - cum_aer_global = cumsum(DSW_global)   [the uncorrected x-position]
      - cum_aer_corrected = H_bench^{-1}(surge)  [the benchmark AER for that surge]

    Returns
    -------
    cum_aer_global    : [n_valid] global-DSW cumulative AER (uncorrected)
    cum_aer_corrected : [n_valid] benchmark-inverted AER (corrected)
    surge_s           : [n_valid] surge values (descending, untouched)
    overlap_mask      : [n_valid] bool — True where benchmark inversion is valid
    None x 4  when fewer than 2 valid storms
    """
    valid = (~np.isnan(resp)) & (~np.isnan(DSW_global)) & (resp > dry_thr)
    if valid.sum() < 2:
        return None, None, None, None

    desc = np.argsort(resp[valid])[::-1]
    surge_s = resp[valid][desc]
    cum_aer_global = np.cumsum(DSW_global[valid][desc])

    # Invert benchmark: surge → AER (no extrapolation)
    cum_aer_corrected = _invert_hc_bench(
        HC_bench_node, tbl_aer, surge_s, extrapolate=False)

    overlap = np.isfinite(cum_aer_corrected)

    # For storms outside the benchmark surge range, keep their global AER
    cum_aer_corrected[~overlap] = cum_aer_global[~overlap]

    # Enforce monotonicity: corrected AER must be non-decreasing
    cum_aer_corrected = _enforce_monotonicity_aer(cum_aer_corrected)

    return cum_aer_global, cum_aer_corrected, surge_s, overlap


# ---------------------------------------------------------------------------
# Cached 631-AER grid
# ---------------------------------------------------------------------------

_AER_631_CACHE = None


def _get_aer_631():
    global _AER_631_CACHE
    if _AER_631_CACHE is None:
        _AER_631_CACHE = build_aer_631()
    return _AER_631_CACHE


# ---------------------------------------------------------------------------
# Public API: compute bias at 22 standard AERs (for storage)
# ---------------------------------------------------------------------------

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
    """
    Compute QBM bias at the standard tbl_aer grid for every node.

    Parameters
    ----------
    aer_mode : str
        Intermediate grid resolution (applies to response mode only).
        "631"      (default) — dense 631-point AER grid.
        "standard" — 22 tbl_aer grid.

    qbm_mode : str
        "aer"      (default) — AER-based correction.  Stored values are
                    the log-AER delta: log(lambda_global) - log(lambda_corrected).
                    Positive delta means global DSW overestimates the AER.
        "response" — response-based correction (legacy).  Stored values
                    are surge bias in physical units (metres).

    Returns
    -------
    bias_tbl : [m x N_AER]  bias at the standard 22-AER grid per node
    """
    k, m = Y_sub.shape
    n_aer = len(tbl_aer)

    # ── C++ fast path ─────────────────────────────────────────────────────
    if _HAS_CPP:
        Y_c  = np.ascontiguousarray(Y_sub,      dtype=np.float64)
        D_c  = np.ascontiguousarray(DSW_global,  dtype=np.float64)
        HC_c = np.ascontiguousarray(HC_bench,    dtype=np.float64)
        A_c  = np.ascontiguousarray(tbl_aer,     dtype=np.float64)

        if qbm_mode == "aer":
            return np.asarray(_cpp_compute_qbm_bias_aer(
                Y_c, D_c, HC_c, A_c, dry_thr))
        else:
            inter = _get_aer_631() if aer_mode == "631" else tbl_aer
            G_c = np.ascontiguousarray(inter, dtype=np.float64)
            return np.asarray(_cpp_compute_qbm_bias_response(
                Y_c, D_c, HC_c, A_c, dry_thr, G_c, win_frac, ramp_frac))

    # ── Python fallback ───────────────────────────────────────────────────
    bias_tbl = np.zeros((m, n_aer), dtype=np.float64)

    if qbm_mode == "aer":
        # ── AER mode: store log-AER delta at 22 AERs (diagnostic) ──
        for node in range(m):
            result = _node_aer_correction(
                Y_sub[:, node], DSW_global, HC_bench[node, :],
                tbl_aer, dry_thr)
            cum_aer_global, cum_aer_corrected, surge_s, overlap = result

            if cum_aer_global is None:
                continue
            if overlap.sum() < 2:
                continue

            # Log-AER delta at per-storm positions (where overlap exists)
            log_delta = (np.log(cum_aer_global[overlap])
                         - np.log(cum_aer_corrected[overlap]))

            # Map to 22 standard AERs via log-linear interpolation
            delta_22 = _interp_log_linear(
                cum_aer_global[overlap], log_delta,
                tbl_aer, extrapolate=True)

            bias_tbl[node, :] = delta_22

        return bias_tbl

    # ── Response mode (legacy): per-storm response bias → intermediate → 22 ──
    if aer_mode == "631":
        inter_grid = _get_aer_631()
    else:
        inter_grid = tbl_aer

    for node in range(m):
        cum_aer_s, surge_s, bias_raw, overlap = _node_storm_bias_response(
            Y_sub[:, node], DSW_global, HC_bench[node, :], tbl_aer, dry_thr)

        if bias_raw is None:
            continue

        # Map per-storm bias to intermediate grid and smooth
        bias_inter = _bias_via_intermediate_grid(
            cum_aer_s, overlap, bias_raw, inter_grid, win_frac, ramp_frac)

        if aer_mode == "631":
            bias_node = _interp_log_linear(
                inter_grid, bias_inter, tbl_aer, extrapolate=True)
        else:
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
    qbm_mode:       str   = "aer",
    win_frac:       float = 0.10,
    ramp_frac:      float = 0.03,
):
    """
    Correct a single node's storm data using QBM.

    Parameters
    ----------
    qbm_mode : str
        "aer"      (default) — returns (cum_aer_corrected, surge_original).
                    Surge values are untouched; AER positions are remapped
                    by inverting the benchmark HC.
        "response" — returns (cum_aer_original, surge_corrected).
                    AER positions are untouched; surge values are shifted
                    to match the benchmark (legacy).

    Returns
    -------
    cum_aer    : [n_valid] AER positions
    surge      : [n_valid] surge values
    (None, None) if correction is not possible.
    """
    if qbm_mode == "aer":
        return _correct_node_aer(
            resp, DSW_global, HC_bench_node, tbl_aer, dry_thr)
    else:
        return _correct_node_response(
            resp, DSW_global, HC_bench_node, bias_22_node, tbl_aer,
            dry_thr, aer_mode, win_frac, ramp_frac)


def _correct_node_aer(resp, DSW_global, HC_bench_node, tbl_aer, dry_thr):
    """AER-based correction: remap AER positions, keep surge untouched."""
    result = _node_aer_correction(
        resp, DSW_global, HC_bench_node, tbl_aer, dry_thr)
    cum_aer_global, cum_aer_corrected, surge_s, overlap = result

    if cum_aer_global is None:
        return None, None

    return cum_aer_corrected, surge_s


def _correct_node_response(resp, DSW_global, HC_bench_node, bias_22_node,
                           tbl_aer, dry_thr, aer_mode, win_frac, ramp_frac):
    """Response-based correction (legacy): shift surge, keep AER."""
    cum_aer_s, surge_s, bias_raw, overlap = _node_storm_bias_response(
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

    surge_corr = surge_s - bias_at_storm
    surge_corr = _enforce_monotonicity(surge_corr)

    return cum_aer_s, surge_corr
