"""
backend/engines/sampling/joint_matrix.py
==========================================
Joint input-output feature matrix  Z = [w * X~  |  Y_r~].

Each block is independently standardised (zero mean, unit variance) before
concatenation.  The scalar weight *w* controls the relative influence of the
TC parameter space (X) versus the hydrodynamic response space (Y_r).

Engine contract: arrays in, arrays out.  No config, no I/O.

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_joint_matrix(
    X:   np.ndarray,
    Y_r: np.ndarray,
    w:   float = 1.0,
) -> tuple[np.ndarray, StandardScaler, StandardScaler]:
    """
    Standardise X and Y_r independently then concatenate with weight:

        Z = [ w * X~  |  Y_r~ ]

    Parameters
    ----------
    w : float
        Relative weight of TC parameters (X) vs. response (Y_r).
        w > 1 emphasises X; w < 1 emphasises Y_r.

    Returns
    -------
    Z        : [n x (p+r)]
    scaler_X : fitted StandardScaler for X
    scaler_Y : fitted StandardScaler for Y_r
    """
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Z = np.hstack([w * scaler_X.fit_transform(X),
                   scaler_Y.fit_transform(Y_r)])
    return Z, scaler_X, scaler_Y
