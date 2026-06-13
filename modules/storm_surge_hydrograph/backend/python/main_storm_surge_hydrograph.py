"""main_storm_surge_hydrograph - orchestrator entry (CyHAN v2.1 5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into an ``SSHConfig``, and runs
the analysis. No user options live here.
"""

from __future__ import annotations

from storm_surge_hydrograph.config import SSHConfig
from storm_surge_hydrograph.orchestrator import SSHOrchestrator, SSHResult


def run(config: dict) -> SSHResult:
    cfg = SSHConfig(**config)
    return SSHOrchestrator(cfg).run()
