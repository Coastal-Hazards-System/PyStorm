"""life_cycle_simulation - Monte-Carlo synthetic TC life cycles from the SCA SRR.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

For a chosen Coastal Reference Location (CRL) and radius of influence, draws a
synthetic catalog of tropical cyclones over a requested number of years and
realizations: a Poisson number of TCs per year (rate lambda = SRR * 2 * radius),
each assigned an intensity stratum from the Low/Med/High SRR ratios and a calendar
day of closest approach from the seasonal (daily or monthly) SRR shape. Driven by
the storm_climatology_analysis SRR tables.
"""

from life_cycle_simulation.config import LCSConfig, STRATA, PLOT_KEYS
from life_cycle_simulation.srr_source import (
    CRLSrr,
    load_srr_table,
    locate_daily_companion,
    load_daily_table,
    build_crl_srr,
)
from life_cycle_simulation.simulator import (
    SimOutput,
    simulate,
    poisson_rate,
    stratum_probs,
    draw_counts,
    add_sequencing,
)
from life_cycle_simulation.orchestrator import (
    LCSOrchestrator,
    LCSResult,
    CRLResult,
)
from life_cycle_simulation.calibration import (
    crl_annual_counts,
    calibrate_correlation,
    calibrate_correlation_regional,
    within_season_latent,
    within_season_rho_estimate,
)
from life_cycle_simulation.plots import render_suite
from life_cycle_simulation import writer, plots, calendar365, calibration

__all__ = [
    "LCSConfig",
    "STRATA",
    "PLOT_KEYS",
    "render_suite",
    "CRLSrr",
    "load_srr_table",
    "locate_daily_companion",
    "load_daily_table",
    "build_crl_srr",
    "SimOutput",
    "simulate",
    "poisson_rate",
    "stratum_probs",
    "draw_counts",
    "add_sequencing",
    "crl_annual_counts",
    "calibrate_correlation",
    "calibrate_correlation_regional",
    "within_season_latent",
    "within_season_rho_estimate",
    "calibration",
    "LCSOrchestrator",
    "LCSResult",
    "CRLResult",
    "writer",
    "plots",
    "calendar365",
]
