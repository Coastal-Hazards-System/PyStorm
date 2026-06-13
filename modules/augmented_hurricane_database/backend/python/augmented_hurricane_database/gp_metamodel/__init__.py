"""gp_metamodel - Gaussian-process metamodels for HURDAT data imputation.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Python re-implementation of the GP metamodel of Taflanidis et al. (universal
kriging with an anisotropic power-exponential kernel) as used by the CHS
HURDAT central-pressure and radius-of-maximum-wind imputation codes. Trains the
metamodels on the observed rows and predicts the missing ones.
"""

from .gp import GPModel, fit_gp
from .features import CP_BASE, RMAX_MAX, RMAX_MIN, motion_known
from .impute import (
    ImputeReport,
    impute_all,
    impute_central_pressure,
    impute_rmax,
)

__all__ = [
    "GPModel",
    "fit_gp",
    "CP_BASE",
    "RMAX_MIN",
    "RMAX_MAX",
    "motion_known",
    "ImputeReport",
    "impute_all",
    "impute_central_pressure",
    "impute_rmax",
]
