"""tc_climatological_analysis - CRL-based tropical-cyclone storm recurrence rates.

For each CHS Coastal Reference Location (CRL), selects the tropical cyclones from
the augmented HURDAT2 best-track (the augmented_hurricane_database output) within
a cutoff distance, then computes the omnidirectional storm recurrence rate (SRR,
storms/km/year) and the directional rate (DSRR, storms/deg/year) with the
Gaussian Kernel Function, both annually and per calendar month, for All/Low/Med/
High intensity bins. Optionally maps the selected TCs per CRL.
"""

from tc_climatological_analysis.config import TCAConfig, BASINS
from tc_climatological_analysis.crls import load_crls
from tc_climatological_analysis.hurdat_source import (
    locate_augmented_hurdat,
    load_augmented_hurdat,
)
from tc_climatological_analysis.selection import select_storms, gaussian_weights
from tc_climatological_analysis.gkf import (
    compute_rates,
    azimuth_diff,
    heading_zero_degree_adj,
    HEADINGS,
    MONTHS,
)
from tc_climatological_analysis.orchestrator import (
    TCAOrchestrator,
    TCAResult,
    BasinResult,
)
from tc_climatological_analysis import plots, basemap, writer

__all__ = [
    "TCAConfig",
    "BASINS",
    "load_crls",
    "locate_augmented_hurdat",
    "load_augmented_hurdat",
    "select_storms",
    "gaussian_weights",
    "compute_rates",
    "azimuth_diff",
    "heading_zero_degree_adj",
    "HEADINGS",
    "MONTHS",
    "TCAOrchestrator",
    "TCAResult",
    "BasinResult",
    "plots",
    "basemap",
    "writer",
]
