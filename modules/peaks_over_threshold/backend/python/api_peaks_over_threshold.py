"""api_peaks_over_threshold - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_peaks_over_threshold.py`` at the module root imports ``run`` from this
file with the operator-edited configuration and invokes it. No user-facing
options live here - only the stage-dispatch orchestration.

A run executes the stages listed in ``config["stages"]`` (canonical order
``download -> detrend -> ntr -> pot``):

  * The PRIMARY use is POT-only (``stages == ["pot"]``): POT runs directly on
    the user-provided ``input_csv``. This is the default when ``stages`` is
    omitted, preserving the original single-purpose behaviour.
  * The SECONDARY use is the upstream NTR pipeline: any of ``download``,
    ``detrend``, ``ntr`` build the POT input from NOAA data first; when ``pot``
    is also present the chain feeds straight into extraction. The chain runs
    once per station in ``config["station_ids"]`` (one or many), with the
    per-station data folders derived from ``config["data_dir"]``.

Public API
----------
  run(config)  ->  POTResult | PipelineResult | dict[str, PipelineResult]
"""

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional, Union

from peaks_over_threshold.config       import POTConfig, PreprocessConfig
from peaks_over_threshold.orchestrator import POTOrchestrator, POTResult
from peaks_over_threshold.preprocessing.orchestrator import (
    PreprocessOrchestrator, PreprocessResult,
)


@dataclass
class PipelineResult:
    """Bundle returned when preprocessing stages run (with or without POT).

    ``pot`` maps each POT target tag to its result - e.g. ``{"dwl": ...,
    "ntr": ...}`` when the chain extracts peaks from both the detrended water
    level and the non-tidal residual.
    """
    preprocess: PreprocessResult
    pot:        dict = field(default_factory=dict)


def run(config) -> Union[POTResult, PipelineResult, dict]:
    """Execute the requested stages.

    Parameters
    ----------
    config : dict | POTConfig | PreprocessConfig
        Job configuration. A ``POTConfig`` (or a dict with no preprocessing
        stages) runs POT only. A dict whose ``stages`` include any of
        ``download``/``detrend``/``ntr`` runs the upstream pipeline first.

    Returns
    -------
    POTResult
        A single POT-only extraction (a ``POTConfig``, or a dict resolving to
        one input series).
    PipelineResult
        A single-station / no-station run with preprocessing stages; bundles the
        preprocess result and a ``pot`` map ``{target: POTResult}``.
    dict[str, POTResult | PipelineResult]
        A batch over ``station_ids`` (keyed by station) or over ``input_csvs``
        (keyed by file stem). A batch of size one collapses to the single
        result above.
    """
    # POTConfig instance → POT-only (backward compatible).
    if isinstance(config, POTConfig):
        return POTOrchestrator(config).run()

    stages = _resolve_stages(config)

    # PRIMARY: POT-only. One path (input_csv) or many (input_csvs, for CLI
    # batch processing over explicit absolute paths).
    if stages == ["pot"]:
        paths = config.get("input_csvs") if isinstance(config, dict) else None
        if paths:
            results: dict = {}
            for p in paths:
                csv = Path(p)
                if not csv.is_file():
                    raise FileNotFoundError(f"POT input not found: {csv}")
                results[csv.stem] = POTOrchestrator(
                    _as_pot_config(config, overrides={"input_csv": csv})).run()
            return results if len(results) != 1 else next(iter(results.values()))
        return POTOrchestrator(_as_pot_config(config)).run()

    # SECONDARY: run the chain once per requested station.
    stations = _resolve_stations(config)
    data_dir = _data_dir(config)
    if not stations:
        return _run_chain(config, stages)        # dirs already in config

    results: dict = {}
    for idx, st in enumerate(stations):
        if len(stations) > 1:
            print(f"\n[POT] ===================== station {st} =====================")
        results[st] = _run_chain(
            _station_config(config, st, data_dir, idx, len(stations)), stages)
    return results if len(results) != 1 else next(iter(results.values()))


def _run_chain(config, stages) -> PipelineResult:
    """Run the preprocessing chain (+ optional POT) for one station's config."""
    pre_cfg = config if isinstance(config, PreprocessConfig) else PreprocessConfig(**config)
    pre = PreprocessOrchestrator(pre_cfg).run()

    pot_results: dict = {}
    if "pot" in stages:
        # Extract peaks from every processed series the chain produced: the
        # detrended water level (dwl_*_pot.csv) and the NTR (ntr_*_pot.csv).
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
def _resolve_stations(config) -> list:
    """Stations to process: ``station_ids`` (list/str) or legacy ``station_id``."""
    if not isinstance(config, dict):
        return []
    s = config.get("station_ids", config.get("station_id"))
    if s is None:
        return []
    return [str(s)] if isinstance(s, str) else [str(x) for x in s]


def _data_dir(config) -> Optional[Path]:
    """Base data dir for per-station folder derivation, if provided."""
    if isinstance(config, dict) and config.get("data_dir"):
        return Path(config["data_dir"])
    return None


# Config values that may be given per station (a list parallel to station_ids)
# or as a single value applied to every station.
_PER_STATION_KEYS = ("ntde_start", "ntde_end", "detrend_slope")


def _station_config(config: dict, station: str, data_dir: Optional[Path],
                    idx: int, n_stations: int) -> dict:
    """Per-station config: derived data folders + per-station resolved values.

    Resolves the tidal epoch (``ntde_start``/``ntde_end``) and the optional
    sea-level slope override (``detrend_slope``) for this station - each may be
    a single value used for all stations or a list parallel to ``station_ids``.
    """
    cfg = dict(config)
    cfg["station_id"] = station
    for key in _PER_STATION_KEYS:
        if key in config:
            cfg[key] = _per_station(config, key, idx, n_stations)
    if data_dir is not None:
        cfg["raw_dir"]       = data_dir / "inputs" / "raw" / station
        cfg["processed_dir"] = data_dir / "inputs" / "processed" / station
        cfg["output_dir"]    = data_dir / "outputs" / station       # data files, per station
        cfg["plots_dir"]     = data_dir / "outputs" / "plots"        # plots, shared
    return cfg


def _per_station(config: dict, key: str, idx: int, n_stations: int):
    """Resolve a possibly-per-station config value for the station at ``idx``.

    ``config[key]`` may be a single value - used for every station, including
    in batch - or a list/tuple parallel to ``station_ids`` giving one value per
    station. A list whose length does not match the station count is a
    configuration error.
    """
    v = config.get(key)
    if isinstance(v, (list, tuple)):
        if len(v) != n_stations:
            raise ValueError(
                f"{key} has {len(v)} entries but there are {n_stations} "
                f"station(s). Provide a single {key} (used for all stations) "
                f"or exactly one per station (parallel to station_ids).")
        return v[idx]
    return v


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
