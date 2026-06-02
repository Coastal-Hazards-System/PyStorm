"""Joint input-output feature matrix  Z = [alpha * X~  |  beta * Y_r~].

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Engine contract: arrays in, arrays out.  No config, no I/O.
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
    """Standardize X and Y_r independently then concatenate with weights:

        Z = [ alpha * X~  |  beta * Y_r~ ]
    """
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Z = np.hstack([alpha * scaler_X.fit_transform(X),
                   beta  * scaler_Y.fit_transform(Y_r)])
    return Z, scaler_X, scaler_Y
