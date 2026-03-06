"""
backend/engines/sampling/pca.py
================================
PCA (POD) dimensionality reduction on the surge response matrix Y.

Engine contract: arrays in, arrays out.  No config, no I/O.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduce_output(
    Y: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, PCA]:
    """
    Compress the surge response matrix via PCA.

    Parameters
    ----------
    Y                  : [n_storms x m_nodes]
    variance_threshold : cumulative explained variance fraction to retain

    Returns
    -------
    Y_r : [n_storms x r]  PCA score matrix
    pca : fitted sklearn PCA object
    """
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    Y_r = pca.fit_transform(Y)
    return Y_r, pca
