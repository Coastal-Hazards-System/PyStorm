"""marginals - Stage 2: per-CRL per-intensity marginal distributions.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Per CRL: the central-pressure deficit Dp is a (jitter-bootstrapped) Weibull, fit on
the >= dp_low body and the [min_dp, dp_low) tail, then truncated to each intensity
band; Rmax is lognormal; Vt is normal (HI/MI) or lognormal (LI); heading uses the
SCA DSRR directional distribution. Emits one tidy record per (intensity, parameter).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from joint_distribution_model.bootstrap import ecdf_boot
from joint_distribution_model.solver import (
    CPP_KERNEL_AVAILABLE, weibull_bootstrap_cpp,
)

DP_CAP = 148.0          # upper truncation for Dp (hPa)


# ---------------------------------------------------------------------------
# Distribution helpers (also reused by plots and any downstream sampler)
# ---------------------------------------------------------------------------

def weibull_cdf(x, scale, shape):
    """Weibull CDF F(x) = 1 - exp(-(x/A)^k), A=scale, k=shape."""
    return 1.0 - np.exp(-(np.asarray(x, float) / scale) ** shape)


def weibull_ppf(q, scale, shape):
    """Weibull quantile A*(-ln(1-q))^(1/k)."""
    return scale * (-np.log(1.0 - np.asarray(q, float))) ** (1.0 / shape)


def trunc_weibull_ppf(q, scale, shape, lo, hi):
    """Quantile of a Weibull truncated to [lo, hi]."""
    flo, fhi = weibull_cdf(lo, scale, shape), weibull_cdf(hi, scale, shape)
    return weibull_ppf(flo + np.asarray(q, float) * (fhi - flo), scale, shape)


def fit_weibull(x: np.ndarray) -> Tuple[float, float]:
    """2-parameter Weibull MLE (location fixed at 0); returns (scale A, shape k).

    Solves the shape score equation by Brent root-finding, which reproduces
    ``weibull_min.fit(x, floc=0)`` far faster (the bootstrap calls it many times).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return np.nan, np.nan
    lnx = np.log(x)
    mlnx = lnx.mean()

    def score(k):
        xk = x ** k
        return (xk * lnx).sum() / xk.sum() - 1.0 / k - mlnx

    try:
        k = brentq(score, 1e-3, 50.0, maxiter=200)
    except ValueError:
        return np.nan, np.nan
    scale = float(np.mean(x ** k) ** (1.0 / k))
    return scale, float(k)


def _weibull_mle_vec(samples: np.ndarray, max_iter: int = 60,
                     tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized two-parameter Weibull MLE over the rows of ``samples`` ([B, m]).

    Solves the shape score equation for all B replicates at once by a vectorized
    Newton iteration (the score and its derivative are elementwise over the replicate
    axis), then sets the scale. Replaces B separate scalar root-finds. Returns
    (scale A [B], shape k [B]).
    """
    X = np.asarray(samples, float)
    lnx = np.log(X)
    mlnx = lnx.mean(axis=1)
    k = np.ones(X.shape[0])
    for _ in range(max_iter):
        xk = X ** k[:, None]
        S0 = xk.sum(1)
        S1 = (xk * lnx).sum(1)
        S2 = (xk * lnx * lnx).sum(1)
        g = S1 / S0 - 1.0 / k - mlnx
        gp = (S2 * S0 - S1 * S1) / (S0 * S0) + 1.0 / (k * k)
        k_new = np.clip(k - g / gp, 1e-3, 50.0)
        if np.nanmax(np.abs(k_new - k)) < tol:
            k = k_new
            break
        k = k_new
    A = (X ** k[:, None]).mean(1) ** (1.0 / k)
    return A, k


def fit_weibull_boot(x: np.ndarray, n_boot: int, th: float, *,
                     rng=None, seed=None, use_cpp: bool = True):
    """Bootstrapped Weibull: mean (scale, shape) over jitter resamples, plus the
    per-replicate parameters (for confidence-limit bands).

    Uses the ``_jdm`` C++ kernel when it is built and a ``seed`` is given; otherwise
    the pure-NumPy path (jitter bootstrap + vectorized MLE). Both are statistically
    equivalent; the kernel is faster and releases the GIL.
    """
    if use_cpp and CPP_KERNEL_AVAILABLE and seed is not None:
        par = weibull_bootstrap_cpp(x, n_boot, th, seed)
    else:
        samples = ecdf_boot(x, n_boot, th, rng)        # [n_boot, Nstrm]
        A_b, k_b = _weibull_mle_vec(samples)
        par = np.column_stack([A_b, k_b])
    A = float(np.nanmean(par[:, 0]))
    k = float(np.nanmean(par[:, 1]))
    return A, k, par


def fit_lognorm(x: np.ndarray) -> Tuple[float, float]:
    """Lognormal MLE in log space: (mu, sigma) = mean/std (1/N) of log(x)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return np.nan, np.nan
    lx = np.log(x)
    return float(lx.mean()), float(lx.std(ddof=0))


def fit_norm(x: np.ndarray) -> Tuple[float, float]:
    """Normal fit (mean, std), sample standard deviation (ddof=1)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1))


# ---------------------------------------------------------------------------
# Per-CRL marginal fitting
# ---------------------------------------------------------------------------

def _rec(intensity, param, dist, p1, p2, lo, hi, n) -> dict:
    return {"intensity": intensity, "param": param, "dist": dist,
            "p1": p1, "p2": p2, "trunc_lo": lo, "trunc_hi": hi, "n": int(n)}


def fit_crl_marginals(bins: Dict[str, np.ndarray],
                      dsrr_mean: Dict[str, float], dsrr_stdv: Dict[str, float], *,
                      n_boot: int, rng, seed=None, min_dp: float, dp_low: float,
                      dp_med: float, dp_cap: float = DP_CAP) -> Tuple[List[dict], dict]:
    """Fit all marginals for one CRL. Returns (records, boot_extra).

    ``bins`` maps SCA bin name (all/high/med/low) to an [N, 4] array of adjusted
    [Hd, Dp, Rmax, Vt]. ``boot_extra`` carries the Dp Weibull parameters (and the
    body bootstrap replicates) for the diagnostic plots.
    """
    records: List[dict] = []
    dp_all = bins["all"][:, 1] if bins["all"].size else np.empty(0)
    body = dp_all[dp_all >= dp_low]                    # >= dp_low (Weibull body)
    tail = dp_all[(dp_all >= min_dp) & (dp_all < dp_low)]

    if body.size >= 3:
        A_body, k_body, boot_par = fit_weibull_boot(
            body, n_boot, th=dp_low, rng=rng, seed=seed)
    else:
        A_body, k_body, boot_par = np.nan, np.nan, None
    A_tail, k_tail = fit_weibull(tail) if tail.size >= 3 else (np.nan, np.nan)

    records.append(_rec("all", "Dp_body", "weibull", A_body, k_body, dp_low, dp_cap, body.size))
    records.append(_rec("all", "Dp_tail", "weibull", A_tail, k_tail, min_dp, dp_low, tail.size))

    # Per-intensity Dp = truncated Weibull (body for HI/MI, tail for LI).
    dp_src = {"high": (A_body, k_body, dp_med, dp_cap),
              "med":  (A_body, k_body, dp_low, dp_med),
              "low":  (A_tail, k_tail, min_dp, dp_low)}

    for b in ("high", "med", "low"):
        d = bins[b]
        n = d.shape[0]
        A, k, lo, hi = dp_src[b]
        records.append(_rec(b, "Dp", "weibull_trunc", A, k, lo, hi, n))

        rm = d[:, 2] if n else np.empty(0)
        mu_r, sig_r = fit_lognorm(rm)
        records.append(_rec(b, "Rmax", "lognorm", mu_r, sig_r, np.nan, np.nan,
                            np.isfinite(rm).sum()))

        vt = d[:, 3] if n else np.empty(0)
        if b in ("high", "med"):
            mu_v, sig_v = fit_norm(vt); vdist = "norm"
        else:
            mu_v, sig_v = fit_lognorm(vt); vdist = "lognorm"
        records.append(_rec(b, "Vt", vdist, mu_v, sig_v, np.nan, np.nan,
                            np.isfinite(vt).sum()))

        # Heading uses the SCA DSRR directional distribution (mean/stdv; cdf in npz).
        records.append(_rec(b, "Hd", "dsrr", dsrr_mean[b], dsrr_stdv[b], -180.0, 180.0, n))

    # Keep only the scalar Weibull parameters (the per-replicate array is large and
    # the diagnostic plots do not use it); the bin records carry everything else.
    boot_extra = {"A_body": A_body, "k_body": k_body,
                  "A_tail": A_tail, "k_tail": k_tail}
    return records, boot_extra
