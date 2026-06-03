"""main_peaks_over_threshold — orchestrator entry (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_peaks_over_threshold.py`` at the module root imports ``run`` from this
file with the operator-edited configuration and invokes it. No user-facing
options live here — only the stage-dispatch orchestration.

A run executes the stages listed in ``config["stages"]`` (canonical order
``download -> detrend -> ntr -> pot``):

  * The PRIMARY use is POT-only (``stages == ["pot"]``): POT runs directly on
    the user-provided ``input_csv``. This is the default when ``stages`` is
    omitted, preserving the original single-purpose behaviour.
  * The SECONDARY use is the upstream NTR pipeline: any of ``download``,
    ``detrend``, ``ntr`` build the POT input from NOAA data first; when ``pot``
    is also present the chain feeds straight into extraction.

Public API
----------
  run(config)  ->  POTResult | PipelineResult
"""

from dataclasses import dataclass, field
from typing      import Optional, Union

from peaks_over_threshold.config       import POTConfig, PreprocessConfig
from peaks_over_threshold.orchestrator import POTOrchestrator, POTResult
from peaks_over_threshold.preprocessing.orchestrator import (
    PreprocessOrchestrator, PreprocessResult,
)


@dataclass
class PipelineResult:
    """Bundle returned when preprocessing stages run (with or without POT).

    ``pot`` maps each POT target tag to its result — e.g. ``{"dwl": ...,
    "ntr": ...}`` when the chain extracts peaks from both the detrended water
    level and the non-tidal residual.
    """
    preprocess: PreprocessResult
    pot:        dict = field(default_factory=dict)


def run(config) -> Union[POTResult, PipelineResult]:
    """Execute the requested stages.

    Parameters
    ----------
    config : dict | POTConfig | PreprocessConfig
        Job configuration. A ``POTConfig`` (or a dict with no preprocessing
        stages) runs POT only. A dict whose ``stages`` include any of
        ``download``/``detrend``/``ntr`` runs the upstream pipeline first.

    Returns
    -------
    POTResult        when POT-only.
    PipelineResult   when any preprocessing stage runs.
    """
    # POTConfig instance → POT-only (backward compatible).
    if isinstance(config, POTConfig):
        return POTOrchestrator(config).run()

    stages = _resolve_stages(config)

    # PRIMARY: POT-only on the user-provided path.
    if stages == ["pot"]:
        return POTOrchestrator(_as_pot_config(config)).run()

    # SECONDARY: run the preprocessing chain, then optionally POT.
    pre_cfg = config if isinstance(config, PreprocessConfig) else PreprocessConfig(**config)
    pre = PreprocessOrchestrator(pre_cfg).run()

    pot_results: dict = {}
    if "pot" in stages:
        # Extract peaks from every processed series the chain produced: the
        # detrended water level (dwl_*_pot.csv) and the NTR (ntr_*_pot.csv).
        # Each target's output filename derives from its input CSV stem.
        targets = []
        if "detrend" in stages:
            targets.append(("dwl", pre.detrended_csv, "Water Level"))
        if "ntr" in stages:
            targets.append(("ntr", pre.ntr_csv, "NTR"))
        if not targets:                      # e.g. stages=["download","pot"]
            cfg_input = config.get("input_csv") if isinstance(config, dict) else None
            targets.append(("input", cfg_input, config.get("value_col", "NTR")
                            if isinstance(config, dict) else "NTR"))

        for tag, csv, value_col in targets:
            pot_cfg = _as_pot_config(config, overrides={
                "input_csv":    csv,
                "datetime_col": pre_cfg.datetime_col,
                "value_col":    value_col,
            })
            pot_results[tag] = POTOrchestrator(pot_cfg).run()

    return PipelineResult(preprocess=pre, pot=pot_results)


# ── helpers ────────────────────────────────────────────────────────────────
def _resolve_stages(config) -> list:
    """Stages list in canonical order; default ``["pot"]``."""
    if isinstance(config, PreprocessConfig):
        return list(config.stages)
    stages = config.get("stages") if isinstance(config, dict) else None
    if not stages:
        return ["pot"]
    if isinstance(stages, str):
        stages = [stages]
    order = ("download", "detrend", "ntr", "pot")
    return [s for s in order if s in set(stages)]


def _as_pot_config(config, overrides: Optional[dict] = None) -> POTConfig:
    """Build a POTConfig from a dict/POTConfig, applying optional overrides."""
    if isinstance(config, POTConfig):
        return config if not overrides else config.model_copy(update=overrides)
    data = dict(config)            # POTConfig ignores extra (preprocessing) keys
    if overrides:
        data.update(overrides)
    return POTConfig(**data)
