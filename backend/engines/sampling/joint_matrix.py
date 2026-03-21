"""
backend/engines/sampling/joint_matrix.py
==========================================
Joint input-output feature matrix  Z = [alpha * X~  |  beta * Y_r~].

Each block is independently standardised (zero mean, unit variance) before
weighting to prevent scale-dependent dominance of either space.

Engine contract: arrays in, arrays out.  No config, no I/O.

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_joint_matrix(
    X:     np.ndarray,
    Y_r:   np.ndarray,
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> tuple[np.ndarray, StandardScaler, StandardScaler]:
    """
    Standardise X and Y_r independently then concatenate with weights:

        Z = [ alpha * X~  |  beta * Y_r~ ]

    Returns
    -------
    Z        : [n x (p+r)]
    scaler_X : fitted StandardScaler for X
    scaler_Y : fitted StandardScaler for Y_r
    """
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Z = np.hstack([alpha * scaler_X.fit_transform(X),
                   beta  * scaler_Y.fit_transform(Y_r)])
    return Z, scaler_X, scaler_Y
