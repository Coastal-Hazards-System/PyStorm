"""Space-filling quality metrics for a selected storm subset.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

  Coverage     — fraction of Y-space k-means clusters represented by subset
  Discrepancy  — centered L2 discrepancy of the subset in X-space (full-data scaled)
  Maximin      — minimum pairwise distance in standardized X-space

Engine contract: arrays in, scalars/dict out.  No config, no I/O.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc
from sklearn.cluster import KMeans


def compute_maximin(X_sub_scaled: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance within the subset in standardized X-space."""
    return float(pdist(X_sub_scaled).min()) if len(X_sub_scaled) >= 2 else 0.0


def compute_coverage(
    Y_r_full:   np.ndarray,
    Y_r_sub:    np.ndarray,
    n_clusters: int,
    seed:       int,
) -> float:
    """Fraction of k-means clusters (fit on full dataset) represented by the subset."""
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(Y_r_full)
    return len(np.unique(km.predict(Y_r_sub))) / n_clusters


def compute_discrepancy(
    X_sub_scaled:  np.ndarray,
    X_full_scaled: np.ndarray,
) -> float:
    """Centered L2 discrepancy of the subset, [0,1]-scaled against full data column range."""
    col_min   = X_full_scaled.min(axis=0)
    col_range = X_full_scaled.max(axis=0) - col_min
    col_range = np.where(col_range == 0, 1.0, col_range)
    X01 = (X_sub_scaled - col_min) / col_range
    X01 = np.clip(X01, 0.0, 1.0)
    return float(qmc.discrepancy(X01, method="CD"))


def evaluate_sf_metrics(
    Z_full:        np.ndarray,
    X_scaled:      np.ndarray,
    Y_r_full:      np.ndarray,
    indices:       np.ndarray,
    n_clusters:    int,
    seed:          int,
) -> dict:
    """Compute all three space-filling metrics for a given subset.

    Returns dict with keys: k, coverage, discrepancy, maximin.
    Z_full is retained for API symmetry (unused for maximin/discrepancy).
    """
    X_sub = X_scaled[indices]
    return {
        "k":           len(indices),
        "coverage":    compute_coverage(Y_r_full, Y_r_full[indices], n_clusters, seed),
        "discrepancy": compute_discrepancy(X_sub, X_scaled),
        "maximin":     compute_maximin(X_sub),
    }
