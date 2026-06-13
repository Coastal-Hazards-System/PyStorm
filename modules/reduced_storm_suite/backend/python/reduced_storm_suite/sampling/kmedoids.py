"""k-medoids (PAM) subset selection.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Engine contract: arrays in, index array out.  No config, no I/O.

Dispatch order:
  1. C++ binding ``reduced_storm_suite._rss`` - preferred. Supports forced medoids.
     Installed into this package by ``backend/engines/cpp/build.py`` per
     CyHAN v2.2 §16.2 / §16.5.
  2. sklearn_extra.cluster.KMedoids - when no forced medoids requested.
  3. _greedy_kmedoids (built-in fallback - BUILD + SWAP).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

try:
    from .._rss import kmedoids_pam as _cpp_pam   # type: ignore[attr-defined]
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

try:
    from sklearn_extra.cluster import KMedoids as _SKLearnKMedoids
    _HAS_SKLEARN_EXTRA = True
except ImportError:
    _HAS_SKLEARN_EXTRA = False


def _greedy_kmedoids(
    Z:      np.ndarray,
    k:      int,
    seed:   int,
    forced: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Greedy PAM  (BUILD: maximin init  +  SWAP).  O(k·n²)."""
    n = Z.shape[0]
    D = cdist(Z, Z)

    if _HAS_CPP:
        forced_arr = (np.asarray(forced, dtype=np.int32)
                      if forced is not None else np.array([], dtype=np.int32))
        return _cpp_pam(D, k, seed, forced_arr)

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

    improved = True
    while improved:
        improved     = False
        current_cost = D[:, selected].min(axis=1).sum()
        non_med      = np.setdiff1d(np.arange(n), selected)
        for i in range(len(selected)):
            if selected[i] in forced_set:
                continue
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


def select_maximin(
    Z:              np.ndarray,
    k:              int,
    seed:           int,
    forced_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Greedy farthest-point (maximin) selection - pure BUILD, no SWAP."""
    n = Z.shape[0]
    D = cdist(Z, Z)

    if forced_indices is not None and len(forced_indices) > 0:
        forced = np.asarray(forced_indices, dtype=int)
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

    return np.array(selected)


def select_subset(
    Z:              np.ndarray,
    k:              int,
    seed:           int,
    forced_indices: Optional[np.ndarray] = None,
    method:         str = "kmedoids",
) -> np.ndarray:
    """Select k storms from the rows of Z by the chosen method.

    method
    ------
    "kmedoids"  PAM - minimizes total distance to the nearest medoid.
                Density-following (medoids concentrate in dense regions).
    "maximin"   Greedy farthest-point - space-filling, spreads the subset
                across the feature space and reaches the extremes/tail.
    """
    if method == "kmedoids":
        return select_kmedoids(Z, k, seed, forced_indices)
    if method == "maximin":
        return select_maximin(Z, k, seed, forced_indices)
    raise ValueError(
        f"unknown selection_method {method!r}; expected 'kmedoids' or 'maximin'")


def select_kmedoids(
    Z:              np.ndarray,
    k:              int,
    seed:           int,
    forced_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select k medoids from the rows of Z."""
    if forced_indices is not None and len(forced_indices) > 0:
        return _greedy_kmedoids(Z, k, seed, forced=np.asarray(forced_indices))

    if _HAS_SKLEARN_EXTRA:
        km = _SKLearnKMedoids(n_clusters=k, metric="euclidean",
                              method="pam", init="k-medoids++",
                              random_state=seed)
        km.fit(Z)
        return km.medoid_indices_

    return _greedy_kmedoids(Z, k, seed)
