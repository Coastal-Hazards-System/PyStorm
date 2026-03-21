"""
backend/engines/sampling/metrics.py
=====================================
Space-filling quality metrics for a selected storm subset.

  Coverage     — fraction of Y-space k-means clusters represented by subset
  Discrepancy  — centered L2 discrepancy of the subset in X-space
  Maximin      — minimum pairwise distance in standardised X-space

Engine contract: arrays in, scalars/dict out.  No config, no I/O.

Developed by: Norberto C. Nadal-Caraballo, PhD

Notes on metric definitions
---------------------------
Discrepancy
  Requires points in [0,1]^p.  We rescale using the FULL dataset's column
  min/max (passed in as X_full_scaled), not the subset's own range.
  Using the subset's range (np.ptp on X_sub) produces values >> 1 and is
  meaningless as a uniformity measure.

Maximin
  Computed in standardised X-space (p dimensions), not in the joint Z-space
  (p + r dimensions).  Z-space maximin is dominated by the PCA block and
  is not a useful input-space design metric.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc
from sklearn.cluster import KMeans


def compute_maximin(X_sub_scaled: np.ndarray) -> float:
    """
    Minimum pairwise Euclidean distance within the subset in standardised X-space.

    Parameters
    ----------
    X_sub_scaled : [k x p]  standardised X rows for the selected subset

    Returns
    -------
    float  (0.0 if fewer than 2 points)
    """
    return float(pdist(X_sub_scaled).min()) if len(X_sub_scaled) >= 2 else 0.0


def compute_coverage(
    Y_r_full:   np.ndarray,
    Y_r_sub:    np.ndarray,
    n_clusters: int,
    seed:       int,
) -> float:
    """
    Fraction of k-means clusters (fit on full dataset) represented by the subset.
    """
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(Y_r_full)
    return len(np.unique(km.predict(Y_r_sub))) / n_clusters


def compute_discrepancy(
    X_sub_scaled:  np.ndarray,
    X_full_scaled: np.ndarray,
) -> float:
    """
    Centered L2 discrepancy of the subset in X-space.

    Points are rescaled to [0,1]^p using the FULL dataset's column min/max
    so that the subset's uniformity is measured relative to the full design
    space, not its own bounding box.

    Parameters
    ----------
    X_sub_scaled  : [k x p]  standardised X for the selected subset
    X_full_scaled : [n x p]  standardised X for the full dataset
                             (provides the reference min/max for [0,1] rescaling)
    """
    col_min   = X_full_scaled.min(axis=0)
    col_range = X_full_scaled.max(axis=0) - col_min
    # Avoid division by zero for constant columns
    col_range = np.where(col_range == 0, 1.0, col_range)
    X01 = (X_sub_scaled - col_min) / col_range
    # Clip to [0,1] to handle any floating-point overshoot at the edges
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
    """
    Compute all three space-filling metrics for a given subset.

    Parameters
    ----------
    Z_full    : [n x (p+r)]  full joint feature matrix  (unused for maximin/disc,
                              retained for API consistency)
    X_scaled  : [n x p]      standardised X for the full dataset
    Y_r_full  : [n x r]      PCA scores for the full dataset
    indices   : [k]          row indices of the selected subset
    n_clusters: k-means cluster count for coverage
    seed      : RNG seed

    Returns
    -------
    dict with keys: k, coverage, discrepancy, maximin
    """
    X_sub = X_scaled[indices]
    return {
        "k":           len(indices),
        "coverage":    compute_coverage(Y_r_full, Y_r_full[indices], n_clusters, seed),
        "discrepancy": compute_discrepancy(X_sub, X_scaled),
        "maximin":     compute_maximin(X_sub),
    }
