"""bootstrap - parametric jitter bootstrap of a peak sample.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Resamples a sorted peak vector with replacement and adds Gaussian jitter scaled by
the local spacing between order statistics, rejecting any resample that falls below
the truncation threshold. Each bootstrap replicate is then refit (by the caller) to
propagate sampling uncertainty into the marginal-distribution parameters.
"""

from __future__ import annotations

import numpy as np


def ecdf_boot(pot: np.ndarray, n_sim: int, th: float, rng) -> np.ndarray:
    """Return an ``[n_sim, Nstrm]`` array of jittered, threshold-respecting resamples.

    ``pot`` is the peak sample (NaNs dropped); ``th`` the truncation floor. Each row
    is sorted in descending order.
    """
    pot = np.asarray(pot, dtype=float)
    pot = np.sort(pot[~np.isnan(pot)])[::-1]            # descending
    nstrm = pot.size
    if nstrm == 0:
        return np.empty((n_sim, 0))
    if nstrm == 1:
        return np.full((n_sim, 1), pot[0])
    dlt = np.abs(np.diff(pot))
    dlt = np.append(dlt, dlt[-1])                       # local spacing, length Nstrm

    boot = np.empty((n_sim, nstrm))
    for i in range(n_sim):
        while True:
            idx = rng.integers(0, nstrm, size=nstrm)    # resample with replacement
            y = pot[idx] + rng.standard_normal(nstrm) * dlt[idx]
            if np.all(y >= th):
                break
        boot[i] = np.sort(y)[::-1]
    return boot
