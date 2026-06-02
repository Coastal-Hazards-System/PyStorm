"""probabilistic_simulation_technique — PST orchestration package (CyHAN v2.0 §5.3 expanded form).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The orchestrator entry point per §5.3 is the ``main_probabilistic_simulation_technique``
module one level up (``backend/python/main_probabilistic_simulation_technique.py``);
this package is its expanded, multi-file realization. Submodules:

  config        pydantic configuration models
  orchestrator  PSTOrchestrator workflow runner
  solver        thin _pst binding wrapper (CyHAN v2.0 §4.1 binding role)
  sampling      bootstrap kernel + GPD-threshold search
  hazard        ensemble fit + tail splice + AEF-table interpolation
  postproc      hazard-curve plotting
  io            POT CSV reader + PST result writers
"""

from .config       import PSTConfig, BootstrapConfig
from .orchestrator import PSTOrchestrator, PSTResult
from .solver       import CPP_KERNEL_AVAILABLE

__all__ = [
    "PSTConfig",
    "BootstrapConfig",
    "PSTOrchestrator",
    "PSTResult",
    "CPP_KERNEL_AVAILABLE",
]
