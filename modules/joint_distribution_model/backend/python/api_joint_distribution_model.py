"""api_joint_distribution_model - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Receives the launcher's option dict, validates it into a ``JDMConfig``, and runs the
joint-distribution fitting. No user options live here.
"""

from __future__ import annotations

from joint_distribution_model.config import JDMConfig
from joint_distribution_model.orchestrator import JDMOrchestrator, JDMResult


def run(config) -> JDMResult:
    cfg = config if isinstance(config, JDMConfig) else JDMConfig(**config)
    return JDMOrchestrator(cfg).run()
