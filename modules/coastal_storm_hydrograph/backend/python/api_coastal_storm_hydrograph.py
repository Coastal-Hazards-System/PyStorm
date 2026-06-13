"""api_coastal_storm_hydrograph - orchestrator entry (CyHAN v2.2 5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into an ``CSHConfig``, and runs
the analysis. No user options live here.
"""

from __future__ import annotations

from coastal_storm_hydrograph.config import CSHConfig
from coastal_storm_hydrograph.orchestrator import CSHOrchestrator, CSHResult


def run(config: dict) -> CSHResult:
    cfg = CSHConfig(**config)
    return CSHOrchestrator(cfg).run()
