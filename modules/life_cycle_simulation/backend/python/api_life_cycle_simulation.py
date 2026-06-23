"""api_life_cycle_simulation - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into an ``LCSConfig``, and runs
the life-cycle Monte-Carlo. No user options live here.
"""

from __future__ import annotations

from life_cycle_simulation.config import LCSConfig
from life_cycle_simulation.orchestrator import LCSOrchestrator, LCSResult


def run(config) -> LCSResult:
    cfg = config if isinstance(config, LCSConfig) else LCSConfig(**config)
    return LCSOrchestrator(cfg).run()
