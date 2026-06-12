"""preprocessing - upstream NTR-pipeline engines for the POT module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Clean, independently usable engines that build the non-tidal-residual (NTR)
time series the POT extractor consumes:

  download_noaa_wl_data   fetch NOAA water level / tide prediction CSVs
  detrend_time_series     remove the linear sea-level trend from water level
  estimate_ntr            NTR = detrended water level - interpolated tide

The stage orchestration that wires these into the download -> detrend -> ntr
chain lives in ``preprocessing.orchestrator`` (driven by the module's main
orchestrator); these functions themselves do no cross-stage sequencing.
"""

from .detrend       import detrend_time_series, fill_missing_time_steps
from .ntr           import estimate_ntr
from .noaa_download import download_noaa_wl_data

__all__ = [
    "download_noaa_wl_data",
    "detrend_time_series",
    "fill_missing_time_steps",
    "estimate_ntr",
]
