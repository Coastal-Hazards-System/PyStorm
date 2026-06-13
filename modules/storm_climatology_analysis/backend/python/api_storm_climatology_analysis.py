"""api_storm_climatology_analysis - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into a ``SCAConfig``, and runs
the analysis. No user options live here.
"""

from __future__ import annotations

from storm_climatology_analysis.config import SCAConfig
from storm_climatology_analysis.orchestrator import SCAOrchestrator, SCAResult


def run(config: dict) -> SCAResult:
    cfg = SCAConfig(**config)
    return SCAOrchestrator(cfg).run()
