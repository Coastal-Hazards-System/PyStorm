"""
backend/engines/sampling/kmedoids.py
======================================
k-medoids (PAM) subset selection.

Engine contract: arrays in, index array out.  No config, no I/O.

Two implementations:
  1. sklearn_extra.cluster.KMedoids  (preferred)
  2. _greedy_kmedoids                (built-in fallback — BUILD + SWAP)

select_kmedoids() dispatches automatically.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import cdist

try:
    from sklearn_extra.cluster import KMedoids as _SKLearnKMedoids
    _HAS_SKLEARN_EXTRA = True
except ImportError:
    _HAS_SKLEARN_EXTRA = False


def _greedy_kmedoids(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Greedy PAM  (BUILD: maximin init  +  SWAP).  O(k·n²)."""
    rng = np.random.default_rng(seed)
    n   = Z.shape[0]
    first    = int(rng.integers(n))
    selected = [first]
    min_d    = cdist(Z, Z[[first]]).ravel()
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, cdist(Z, Z[[nxt]]).ravel())
    selected = np.array(selected)

    D = cdist(Z, Z)
    improved = True
    while improved:
        improved     = False
        current_cost = D[:, selected].min(axis=1).sum()
        non_med      = np.setdiff1d(np.arange(n), selected)
        for i in range(len(selected)):
            for cand in non_med:
                trial = selected.copy()
                trial[i] = cand
                cost = D[:, trial].min(axis=1).sum()
                if cost < current_cost - 1e-10:
                    selected     = trial
                    current_cost = cost
                    improved     = True
                    break
            if improved:
                break
    return selected


def select_kmedoids(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Select k medoids from the rows of Z.

    Returns
    -------
    indices : [k]  row indices into Z
    """
    if _HAS_SKLEARN_EXTRA:
        km = _SKLearnKMedoids(n_clusters=k, metric="euclidean",
                              method="pam", init="k-medoids++",
                              random_state=seed)
        km.fit(Z)
        return km.medoid_indices_

    warnings.warn(
        "scikit-learn-extra not found — using built-in greedy PAM. "
        "pip install scikit-learn-extra",
        UserWarning, stacklevel=2,
    )
    return _greedy_kmedoids(Z, k, seed)
