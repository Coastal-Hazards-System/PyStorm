"""
backend/engines/optimization/bayesian_ab.py
============================================
Bayesian optimization for (alpha, beta) weight selection in the RTCS
joint-matrix formulation.

Uses a lightweight Gaussian Process (GP) surrogate with Expected Improvement
(EI) acquisition to find the optimal (alpha, beta) that minimises the
DSW-based HC reconstruction score (|mean_bias| + mean_rmse).

No external dependencies beyond numpy and scipy.

Public API
----------
  optimize_alpha_beta(objective, bounds, n_calls, n_initial, seed)
      -> OptimizeResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import minimize as sp_minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """Result of Bayesian alpha/beta optimization."""
    best_alpha: float
    best_beta:  float
    best_score: float
    X_sampled:  np.ndarray          # [n_calls x 2] in original space
    y_sampled:  np.ndarray          # [n_calls]
    all_rows:   list = field(default_factory=list)  # dicts for sweep CSV


# ---------------------------------------------------------------------------
# Gaussian Process (Matérn 5/2 kernel)
# ---------------------------------------------------------------------------

def _matern52_kernel(X1: np.ndarray, X2: np.ndarray,
                     length_scales: np.ndarray, signal_var: float) -> np.ndarray:
    """
    Matérn 5/2 kernel with ARD (automatic relevance determination).

    X1 : [n1 x d]
    X2 : [n2 x d]
    Returns [n1 x n2] covariance matrix.
    """
    # Scaled distances
    X1s = X1 / length_scales
    X2s = X2 / length_scales

    # Pairwise Euclidean distance
    sq = np.sum(X1s**2, axis=1, keepdims=True) + \
         np.sum(X2s**2, axis=1, keepdims=False) - \
         2.0 * X1s @ X2s.T
    sq = np.maximum(sq, 0.0)
    r = np.sqrt(sq)

    sqrt5r = np.sqrt(5.0) * r
    return signal_var * (1.0 + sqrt5r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5r)


class _GaussianProcess:
    """Minimal GP with Matérn 5/2 kernel for 2-D surrogate modelling."""

    def __init__(self, noise_var: float = 1e-6):
        self.noise_var = noise_var
        self._X = None
        self._y = None
        self._L = None          # Cholesky factor of K + noise*I
        self._alpha = None      # L \ (L^T \ y)
        self._length_scales = None
        self._signal_var = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP to observations (X, y).  X: [n x d], y: [n]."""
        self._X = np.array(X, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)
        n = len(y)

        # Estimate hyperparameters from data spread
        self._signal_var = max(np.var(y), 1e-8)
        self._length_scales = np.std(X, axis=0)
        self._length_scales = np.where(
            self._length_scales > 1e-8, self._length_scales, 1.0)

        K = _matern52_kernel(self._X, self._X,
                             self._length_scales, self._signal_var)
        K += self.noise_var * np.eye(n)

        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, self._y)

    def predict(self, X_new: np.ndarray):
        """Return (mean, std) at X_new [n_new x d]."""
        X_new = np.atleast_2d(X_new)
        K_s  = _matern52_kernel(X_new, self._X,
                                self._length_scales, self._signal_var)
        K_ss = _matern52_kernel(X_new, X_new,
                                self._length_scales, self._signal_var)

        mu = K_s @ self._alpha
        v  = cho_solve(self._L, K_s.T)
        var = np.diag(K_ss) - np.sum(K_s.T * v, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)


# ---------------------------------------------------------------------------
# Acquisition function: Expected Improvement
# ---------------------------------------------------------------------------

def _expected_improvement(X: np.ndarray, gp: _GaussianProcess,
                          y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    EI(x) = (y_best - mu) * Phi(z) + sigma * phi(z)
    where z = (y_best - mu - xi) / sigma.

    We are MINIMISING, so y_best = min(y_observed).
    """
    mu, sigma = gp.predict(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling (initial design)
# ---------------------------------------------------------------------------

def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n points in [0,1]^d via LHS."""
    samples = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_alpha_beta(
    objective,
    bounds: tuple = ((0.01, 50.0), (0.01, 2.0)),
    n_calls: int = 16,
    n_initial: int = 5,
    seed: int = 42,
    xi: float = 0.01,
) -> OptimizeResult:
    """
    Bayesian optimization of (alpha, beta) using a GP surrogate.

    Parameters
    ----------
    objective : callable(alpha, beta) -> dict
        Must return a dict with at least 'mean_bias' and 'mean_rmse' keys.
    bounds : ((alpha_lo, alpha_hi), (beta_lo, beta_hi))
        Search bounds in original space.  Optimisation runs in log10 space.
    n_calls : int
        Total number of objective evaluations (initial + BO iterations).
    n_initial : int
        Number of Latin Hypercube initial samples.
    seed : int
        Random seed for reproducibility.
    xi : float
        Exploration parameter for Expected Improvement.

    Returns
    -------
    OptimizeResult with best_alpha, best_beta, best_score, and full history.
    """
    rng = np.random.default_rng(seed)

    # Work in log10 space for both alpha and beta
    log_bounds = np.array([[np.log10(b[0]), np.log10(b[1])] for b in bounds])

    # Initial design: Latin Hypercube in log10 space
    X_log = _latin_hypercube(n_initial, 2, rng)
    for d in range(2):
        X_log[:, d] = log_bounds[d, 0] + X_log[:, d] * (log_bounds[d, 1] - log_bounds[d, 0])

    # Evaluate initial points
    X_all = []  # log10 space
    y_all = []
    rows  = []

    for i in range(n_initial):
        alpha = 10.0 ** X_log[i, 0]
        beta  = 10.0 ** X_log[i, 1]
        hc_m = objective(alpha, beta)
        score = abs(hc_m["mean_bias"]) + hc_m["mean_rmse"]
        X_all.append(X_log[i])
        y_all.append(score)
        rows.append({"alpha": alpha, "beta": beta, **hc_m, "score": score})

    # Bayesian optimization loop
    gp = _GaussianProcess(noise_var=1e-6)
    n_bo = n_calls - n_initial

    for _ in range(n_bo):
        X_arr = np.array(X_all)
        y_arr = np.array(y_all)
        gp.fit(X_arr, y_arr)

        y_best = y_arr.min()

        # Optimise acquisition function via multi-start L-BFGS-B
        best_ei = -np.inf
        best_x  = None

        # Random candidates + L-BFGS-B restarts
        n_restarts = 20
        X_cand = np.empty((n_restarts, 2))
        for d in range(2):
            X_cand[:, d] = rng.uniform(log_bounds[d, 0], log_bounds[d, 1],
                                        size=n_restarts)

        for x0 in X_cand:
            def neg_ei(x):
                return -_expected_improvement(
                    x.reshape(1, -1), gp, y_best, xi)[0]

            res = sp_minimize(
                neg_ei, x0,
                bounds=[(log_bounds[d, 0], log_bounds[d, 1]) for d in range(2)],
                method="L-BFGS-B")

            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x  = res.x

        # Evaluate the best acquisition point
        alpha = 10.0 ** best_x[0]
        beta  = 10.0 ** best_x[1]
        hc_m = objective(alpha, beta)
        score = abs(hc_m["mean_bias"]) + hc_m["mean_rmse"]
        X_all.append(best_x.copy())
        y_all.append(score)
        rows.append({"alpha": alpha, "beta": beta, **hc_m, "score": score})

    # Find best
    y_arr = np.array(y_all)
    best_idx = int(np.argmin(y_arr))
    best_alpha = 10.0 ** X_all[best_idx][0]
    best_beta  = 10.0 ** X_all[best_idx][1]

    # Build X_sampled in original space
    X_orig = np.array([[10.0 ** x[0], 10.0 ** x[1]] for x in X_all])

    return OptimizeResult(
        best_alpha=best_alpha,
        best_beta=best_beta,
        best_score=y_arr[best_idx],
        X_sampled=X_orig,
        y_sampled=y_arr,
        all_rows=rows,
    )
