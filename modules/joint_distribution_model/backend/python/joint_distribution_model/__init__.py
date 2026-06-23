"""joint_distribution_model - per-CRL JPM joint distribution of TC parameters.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

For each CRL and intensity bin, characterizes the joint distribution of the
tropical-cyclone parameters [Heading, central-pressure deficit Dp, Rmax, forward
translation Vt]: a distance-weighted, intensity-binned adjustment of the SCA
selection, marginal distributions (Dp truncated-Weibull with a jitter bootstrap,
Rmax lognormal, Vt normal/lognormal, heading from the SCA DSRR), and a meta-Gaussian
copula (Kendall tau -> Gaussian rho). Consumes the storm_climatology_analysis
outputs. Fitting only; a synthetic-parameter sampler belongs downstream (LCS).
"""

from joint_distribution_model.config import (
    JDMConfig, BASINS, INTENSITY_BINS, INTENSITY_LABEL, PARAM_NAMES,
)
from joint_distribution_model.adjust import (
    distance_weighted_adj, distance_weighted_adj_heading,
    heading_zero_degree_adj, adjust_crl,
)
from joint_distribution_model.bootstrap import ecdf_boot
from joint_distribution_model.marginals import (
    fit_weibull, fit_weibull_boot, fit_lognorm, fit_norm, fit_crl_marginals,
    weibull_cdf, weibull_ppf, trunc_weibull_ppf,
)
from joint_distribution_model.copula import kendall_matrix, gaussian_rho, fit_copula
from joint_distribution_model.orchestrator import (
    JDMOrchestrator, JDMResult, BasinResult,
)
from joint_distribution_model import sca_source, writer, plots

__all__ = [
    "JDMConfig", "BASINS", "INTENSITY_BINS", "INTENSITY_LABEL", "PARAM_NAMES",
    "distance_weighted_adj", "distance_weighted_adj_heading",
    "heading_zero_degree_adj", "adjust_crl", "ecdf_boot",
    "fit_weibull", "fit_weibull_boot", "fit_lognorm", "fit_norm",
    "fit_crl_marginals", "weibull_cdf", "weibull_ppf", "trunc_weibull_ppf",
    "kendall_matrix", "gaussian_rho", "fit_copula",
    "JDMOrchestrator", "JDMResult", "BasinResult",
    "sca_source", "writer", "plots",
]
