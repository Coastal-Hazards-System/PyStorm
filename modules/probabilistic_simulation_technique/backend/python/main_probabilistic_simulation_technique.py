"""main_probabilistic_simulation_technique — orchestrator entry (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_probabilistic_simulation_technique.py`` at the module root imports
``run`` from this file with the operator-edited configuration and invokes it.
No user-facing options live here; no orchestration logic lives in the launcher.

The substantive workflow has been expanded into a dedicated package per
§5.3 ("Begin as a single file and expand into a ``backend/python/<name>/``
package as complexity warrants, preserving its import entry point"); see
``probabilistic_simulation_technique/`` for the per-stage modules.

Public API
----------
  run(config)  ->  PSTResult     execute one PST job
"""

from probabilistic_simulation_technique.config       import PSTConfig
from probabilistic_simulation_technique.orchestrator import (
    PSTOrchestrator, PSTResult,
)


def run(config) -> PSTResult:
    """Execute one PST job.

    Parameters
    ----------
    config : dict or PSTConfig
        Job configuration. Dictionaries are validated through ``PSTConfig``
        before use; ``PSTConfig`` instances are passed through unchanged.

    Returns
    -------
    PSTResult
        In-memory bundle of the produced hazard-curve tables, the ensemble,
        and the chosen GPD threshold.
    """
    if not isinstance(config, PSTConfig):
        config = PSTConfig(**config)
    return PSTOrchestrator(config).run()
