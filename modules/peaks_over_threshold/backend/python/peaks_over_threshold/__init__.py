"""peaks_over_threshold — POT orchestration package (CyHAN v2.0 §5.3 expanded form).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The orchestrator entry point per §5.3 is the ``main_peaks_over_threshold``
module one level up (``backend/python/main_peaks_over_threshold.py``); this
package is its expanded, multi-file realization. Submodules:

  config        pydantic POTConfig
  orchestrator  POTOrchestrator workflow runner
  solver        thin _pot binding wrapper (CyHAN v2.0 §4.1 binding role)
  sampling      iterative threshold search (C++ kernel + Python fallback)
  segmentation  hydrograph / peak-gap event segmenters (pure-Python fallback)
  postproc      time-series + peaks diagnostic plotter
  io            CSV reader for input time series, writer for POT peaks
"""

from .config       import POTConfig
from .orchestrator import POTOrchestrator, POTResult
from .solver       import CPP_KERNEL_AVAILABLE

__all__ = [
    "POTConfig",
    "POTOrchestrator",
    "POTResult",
    "CPP_KERNEL_AVAILABLE",
]
