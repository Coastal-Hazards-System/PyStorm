"""copula - Stage 3: meta-Gaussian copula correlation.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The dependence among the JPM parameters [Hd, Dp, Rmax, Vt] is captured by a
meta-Gaussian copula. The rank dependence is measured by Kendall's tau, then mapped
to the Gaussian copula correlation rho by the standard relation rho = sin(pi*tau/2).
Pairwise on the historical sample, with any row containing a NaN dropped first.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def kendall_matrix(data: np.ndarray) -> np.ndarray:
    """Pairwise Kendall's tau matrix of the columns of ``data`` ([N, p]).

    Rows with any NaN are dropped first. The diagonal is 1; an
    undefined pair (e.g. a constant column) yields NaN off-diagonal.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-D [N, p].")
    data = data[~np.isnan(data).any(axis=1)]
    p = data.shape[1]
    tau = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            t = kendalltau(data[:, i], data[:, j]).statistic if data.shape[0] >= 2 else np.nan
            tau[i, j] = tau[j, i] = t
    return tau


def gaussian_rho(tau: np.ndarray) -> np.ndarray:
    """Gaussian-copula correlation from Kendall's tau: rho = sin(pi*tau/2)."""
    return np.sin(np.pi * np.asarray(tau, dtype=float) / 2.0)


def fit_copula(data: np.ndarray):
    """Return (tau, rho) for a parameter matrix ([N, p]); rho = sin(pi*tau/2)."""
    tau = kendall_matrix(data)
    return tau, gaussian_rho(tau)
