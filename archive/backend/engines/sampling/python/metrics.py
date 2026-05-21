"""
backend/engines/sampling/metrics.py
=====================================
Space-filling quality metrics for a selected storm subset.

  Coverage     — fraction of Y-space k-means clusters represented by subset
  Discrepancy  — centered L2 discrepancy in standardised X-space
  Maximin      — minimum pairwise distance in joint Z-space

Engine contract: arrays in, scalars/dict out.  No config, no I/O.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc
from sklearn.cluster import KMeans


def compute_maximin(Z_sub: np.ndarray) -> float:
    return float(pdist(Z_sub).min()) if len(Z_sub) >= 2 else 0.0


def compute_coverage(
    Y_r_full:   np.ndarray,
    Y_r_sub:    np.ndarray,
    n_clusters: int,
    seed:       int,
) -> float:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(Y_r_full)
    return len(np.unique(km.predict(Y_r_sub))) / n_clusters


def compute_discrepancy(X_sub_scaled: np.ndarray) -> float:
    rng = np.ptp(X_sub_scaled, axis=0)
    X01 = (X_sub_scaled - X_sub_scaled.min(axis=0)) / (rng + 1e-12)
    return float(qmc.discrepancy(X01, method="CD"))


def evaluate_sf_metrics(
    Z_full:     np.ndarray,
    X_scaled:   np.ndarray,
    Y_r_full:   np.ndarray,
    indices:    np.ndarray,
    n_clusters: int,
    seed:       int,
) -> dict:
    """Return dict with keys: k, coverage, discrepancy, maximin."""
    return {
        "k":           len(indices),
        "coverage":    compute_coverage(Y_r_full, Y_r_full[indices], n_clusters, seed),
        "discrepancy": compute_discrepancy(X_scaled[indices]),
        "maximin":     compute_maximin(Z_full[indices]),
    }
