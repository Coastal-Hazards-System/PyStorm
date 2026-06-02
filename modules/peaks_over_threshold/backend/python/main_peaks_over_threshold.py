"""main_peaks_over_threshold — orchestrator entry (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_peaks_over_threshold.py`` at the module root imports ``run`` from this
file with the operator-edited configuration and invokes it. No user-facing
options live here.

The substantive workflow has been expanded into a dedicated package
``peaks_over_threshold/`` per §5.3 ("Begin as a single file and expand into a
``backend/python/<name>/`` package as complexity warrants, preserving its
import entry point").

Public API
----------
  run(config)  ->  POTResult     execute one POT extraction
"""

from peaks_over_threshold.config       import POTConfig
from peaks_over_threshold.orchestrator import POTOrchestrator, POTResult


def run(config) -> POTResult:
    """Execute one POT extraction.

    Parameters
    ----------
    config : dict or POTConfig
        Job configuration. Dicts are validated through ``POTConfig`` before
        use; ``POTConfig`` instances pass through unchanged.

    Returns
    -------
    POTResult
        In-memory bundle of the converged threshold, the selected peaks
        DataFrame, and the diagnostic fields produced by the kernel.
    """
    if not isinstance(config, POTConfig):
        config = POTConfig(**config)
    return POTOrchestrator(config).run()
