"""main_augmented_hurricane_database - orchestrator entry (CyHAN v2.1 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_augmented_hurricane_database.py`` at the module root imports ``run`` from
this file with the operator-edited configuration and invokes it. No user-facing
options live here.

Public API
----------
  run(config)  ->  AHDResult
"""

from __future__ import annotations

from typing import Union

from augmented_hurricane_database.config import AHDConfig
from augmented_hurricane_database.orchestrator import AHDOrchestrator, AHDResult


def run(config: Union[dict, AHDConfig]) -> AHDResult:
    """Build the HURDAT-like CSV(s) for the configured basins.

    Parameters
    ----------
    config : dict | AHDConfig
        Operator configuration (see AHDConfig). A plain dict is validated into
        an AHDConfig; extra keys are ignored.

    Returns
    -------
    AHDResult
        Per-basin source file, output paths, and storm/row counts.
    """
    cfg = config if isinstance(config, AHDConfig) else AHDConfig(**config)
    return AHDOrchestrator(cfg).run()
