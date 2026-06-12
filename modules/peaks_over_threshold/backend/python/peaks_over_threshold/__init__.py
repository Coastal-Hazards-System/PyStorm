"""peaks_over_threshold — POT orchestration package (CyHAN v2.1 §5.3 expanded form).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The orchestrator entry point per §5.3 is the ``main_peaks_over_threshold``
module one level up (``backend/python/main_peaks_over_threshold.py``); this
package is its expanded, multi-file realization. Submodules:

  config        pydantic POTConfig + PreprocessConfig
  orchestrator  POTOrchestrator workflow runner
  solver        thin _pot binding wrapper (CyHAN v2.1 §4.1 binding role)
  sampling      iterative threshold search (C++ kernel + Python fallback)
  segmentation  hydrograph / peak-gap event segmenters (pure-Python fallback)
  preprocessing NOAA download, detrending, NTR estimation (upstream chain)
  postproc      time-series + peaks diagnostic plotter
  io            CSV reader for input time series, writers for module outputs
"""

from .config       import POTConfig, PreprocessConfig
from .orchestrator import POTOrchestrator, POTResult
from .preprocessing import (
    download_noaa_wl_data, detrend_time_series, estimate_ntr,
)
from .preprocessing.orchestrator import PreprocessOrchestrator, PreprocessResult
from .solver       import CPP_KERNEL_AVAILABLE

__all__ = [
    "POTConfig",
    "PreprocessConfig",
    "POTOrchestrator",
    "POTResult",
    "PreprocessOrchestrator",
    "PreprocessResult",
    "download_noaa_wl_data",
    "detrend_time_series",
    "estimate_ntr",
    "CPP_KERNEL_AVAILABLE",
]
