"""main_tc_climatological_analysis - orchestrator entry (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into a ``TCAConfig``, and runs
the analysis. No user options live here.
"""

from __future__ import annotations

from tc_climatological_analysis.config import TCAConfig
from tc_climatological_analysis.orchestrator import TCAOrchestrator, TCAResult


def run(config: dict) -> TCAResult:
    cfg = TCAConfig(**config)
    return TCAOrchestrator(cfg).run()
