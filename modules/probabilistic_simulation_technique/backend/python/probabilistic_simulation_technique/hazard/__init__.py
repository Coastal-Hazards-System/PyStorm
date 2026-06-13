"""hazard - hazard-curve assembly: GPD ensemble, empirical tail, AER grids.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from .curve import (
    make_aer_grids,
    fit_gpd_ensemble,
    assemble_hazard_curve,
    interpolate_to_table,
)

__all__ = [
    "make_aer_grids",
    "fit_gpd_ensemble",
    "assemble_hazard_curve",
    "interpolate_to_table",
]
