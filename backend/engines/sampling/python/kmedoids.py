"""
backend/engines/sampling/kmedoids.py
======================================
k-medoids (PAM) subset selection.

Engine contract: arrays in, index array out.  No config, no I/O.

Two implementations:
  1. sklearn_extra.cluster.KMedoids  (preferred, unconstrained only)
  2. _greedy_kmedoids                (built-in fallback — BUILD + SWAP)

select_kmedoids() dispatches automatically.
When forced_indices are provided, the greedy implementation is always used
because sklearn_extra does not support fixed medoids.

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

try:
    from sklearn_extra.cluster import KMedoids as _SKLearnKMedoids
    _HAS_SKLEARN_EXTRA = True
except ImportError:
    _HAS_SKLEARN_EXTRA = False

try:
    from backend.engines.sampling.cpp._kmedoids_cpp import kmedoids_pam as _cpp_pam
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


def _greedy_kmedoids(
    Z:      np.ndarray,
    k:      int,
    seed:   int,
    forced: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Greedy PAM  (BUILD: maximin init  +  SWAP).  O(k·n²).

    Parameters
    ----------
    forced : optional array of row indices that must appear in the result.
        BUILD initialises with these and fills remaining slots greedily.
        SWAP never replaces a forced medoid.
    """
    n = Z.shape[0]

    # ── C++ fast path ────────────────────────────────────────────────────
    D = cdist(Z, Z)
    if _HAS_CPP:
        forced_arr = (np.asarray(forced, dtype=np.int32)
                      if forced is not None else np.array([], dtype=np.int32))
        return _cpp_pam(D, k, seed, forced_arr)

    # ── BUILD (Python fallback, uses precomputed D) ────────────────────────
    if forced is not None and len(forced) > 0:
        selected = list(forced)
        if len(selected) >= k:
            return np.array(selected[:k])
        min_d = D[:, forced].min(axis=1)
    else:
        rng      = np.random.default_rng(seed)
        first    = int(rng.integers(n))
        selected = [first]
        min_d    = D[:, first].copy()

    while len(selected) < k:
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, D[:, nxt])

    selected = np.array(selected)
    forced_set = set(forced.tolist() if forced is not None else [])

    # ── SWAP ───────────────────────────────────────────────────────────────
    improved = True
    while improved:
        improved     = False
        current_cost = D[:, selected].min(axis=1).sum()
        non_med      = np.setdiff1d(np.arange(n), selected)
        for i in range(len(selected)):
            if selected[i] in forced_set:
                continue                        # forced medoid — never swap out
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


def select_kmedoids(
    Z:              np.ndarray,
    k:              int,
    seed:           int,
    forced_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Select k medoids from the rows of Z.

    Parameters
    ----------
    forced_indices : optional array of row indices that must be included.
        When provided, sklearn_extra is bypassed and the greedy PAM is used
        so that forced medoids are never swapped out.

    Returns
    -------
    indices : [k]  row indices into Z
    """
    if forced_indices is not None and len(forced_indices) > 0:
        return _greedy_kmedoids(Z, k, seed, forced=np.asarray(forced_indices))

    if _HAS_SKLEARN_EXTRA:
        km = _SKLearnKMedoids(n_clusters=k, metric="euclidean",
                              method="pam", init="k-medoids++",
                              random_state=seed)
        km.fit(Z)
        return km.medoid_indices_

    return _greedy_kmedoids(Z, k, seed)
