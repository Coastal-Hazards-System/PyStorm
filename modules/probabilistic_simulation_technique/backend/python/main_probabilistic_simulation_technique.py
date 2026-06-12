"""main_probabilistic_simulation_technique — orchestrator entry (CyHAN v2.1 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_probabilistic_simulation_technique.py`` at the module root imports
``run`` from this file with the operator-edited configuration and invokes it.
No user-facing options live here; no orchestration logic lives in the launcher.

Input resolution
----------------
``run`` resolves the POT input(s) from the config's ``input_mode`` before
handing each to ``PSTOrchestrator``:

  "path"    — one POT CSV at ``input_csv``.
  "station" — POT file(s) from the peaks_over_threshold module's outputs for
              ``station_id``, selected by ``targets`` ("dwl"/"ntr"/"both"):
              ``<pot_outputs_dir>/<station>/<target>_<station>_pot.csv``.

The substantive workflow lives in the ``probabilistic_simulation_technique/``
package (per §5.3 expansion).

Public API
----------
  run(config)  ->  PSTResult | dict[str, PSTResult]
"""

from pathlib import Path
from typing  import Dict, List, Optional, Tuple, Union

from probabilistic_simulation_technique.config       import PSTConfig
from probabilistic_simulation_technique.orchestrator import (
    PSTOrchestrator, PSTResult,
)

_VALID_TARGETS = ("dwl", "ntr")
# Keys consumed only by input resolution; not part of PSTConfig.
_RESOLUTION_KEYS = ("input_mode", "pot_outputs_dir", "station_id",
                    "station_ids", "input_csvs", "targets", "target_ylabels")


def run(config) -> Union[PSTResult, Dict[str, PSTResult]]:
    """Execute one or more PST jobs.

    Parameters
    ----------
    config : dict or PSTConfig
        A ``PSTConfig`` (or a dict with no ``input_mode``) runs PST once on
        ``input_csv``. A dict with ``input_mode="station"`` resolves one POT
        file per requested target and runs PST on each.

    Returns
    -------
    PSTResult                when a single input is resolved.
    dict[str, PSTResult]     keyed by target tag when several run (e.g. both
                             "dwl" and "ntr").
    """
    if isinstance(config, PSTConfig):
        return PSTOrchestrator(config).run()

    inputs   = _resolve_inputs(config)
    base_out = config.get("output_dir")
    results: Dict[str, PSTResult] = {}
    for tag, csv, station in inputs:
        overrides = {"input_csv": csv}
        # Data files go to data/outputs/<station>/ per station; plots stay in
        # the shared plots_dir (data/outputs/plots) for every run.
        if station and base_out is not None:
            overrides["output_dir"] = Path(base_out) / station
        # Record length is resolved in the orchestrator (auto = n_pot /
        # events_per_year) — no water-level file needed here.
        ylabel = _ylabel_for(config, tag)
        if ylabel is not None:
            overrides["y_axis_label"] = ylabel
        print(f"[PST] === target '{tag}': {csv.name} ===")
        results[tag] = PSTOrchestrator(_as_pst_config(config, overrides)).run()

    return next(iter(results.values())) if len(results) == 1 else results


# ── input resolution ────────────────────────────────────────────────────────
def _resolve_inputs(config: dict) -> List[Tuple[str, Path, Optional[str]]]:
    """Resolve POT input(s) as ``[(tag, path, station), ...]`` per ``input_mode``.

    ``station`` is the station id in station mode (drives the per-station output
    folder) or ``None`` in path mode.
    """
    mode = str(config.get("input_mode", "path")).lower().strip()

    if mode == "path":
        # One path (input_csv) or many (input_csvs) — the latter enables CLI
        # batch processing over explicit absolute paths.
        paths = list(config.get("input_csvs") or [])
        if not paths and config.get("input_csv") is not None:
            paths = [config["input_csv"]]
        if not paths:
            raise ValueError("path mode requires 'input_csv' or 'input_csvs'")
        resolved: List[Tuple[str, Path, Optional[str]]] = []
        for p in paths:
            csv = Path(p)
            if not csv.is_file():
                raise FileNotFoundError(f"PST input not found: {csv}")
            resolved.append((csv.stem, csv, None))
        return resolved

    if mode == "station":
        base     = Path(config["pot_outputs_dir"])
        stations = _resolve_stations(config)
        if not stations:
            raise ValueError("station mode requires 'station_ids' (or 'station_id')")
        targets  = _parse_targets(config.get("targets", "both"))
        resolved: List[Tuple[str, Path, Optional[str]]] = []
        for station in stations:                 # batch: one or many stations
            for t in targets:                    # × each requested series
                csv = base / station / f"{t}_{station}_pot.csv"
                if not csv.is_file():
                    raise FileNotFoundError(
                        f"PST station-mode input not found:\n  {csv}\n"
                        f"Run the POT chain for station {station} first so it "
                        f"produces {t}_{station}_pot.csv (stages include "
                        f"'{t}'-producing step + 'pot')."
                    )
                resolved.append((f"{t}_{station}", csv, station))
        return resolved

    raise ValueError(f"input_mode must be 'station' or 'path'; got {mode!r}")


def _resolve_stations(config: dict) -> List[str]:
    """Stations to process: ``station_ids`` (list/str) or legacy ``station_id``."""
    s = config.get("station_ids", config.get("station_id"))
    if s is None:
        return []
    return [str(s)] if isinstance(s, str) else [str(x) for x in s]


def _parse_targets(targets) -> List[str]:
    """Normalize ``targets`` to a canonical ``['dwl', 'ntr']``-ordered list."""
    if isinstance(targets, (list, tuple)):
        items = [str(x).lower().strip() for x in targets]
    else:
        t = str(targets).lower().strip()
        items = list(_VALID_TARGETS) if t == "both" else [t]

    bad = [x for x in items if x not in _VALID_TARGETS]
    if bad:
        raise ValueError(
            f"targets must be 'dwl', 'ntr', or 'both'; got {bad}")
    return [t for t in _VALID_TARGETS if t in set(items)]   # canonical, deduped


def _ylabel_for(config: dict, tag: str):
    """Per-target y-axis label (station mode) falling back to ``y_axis_label``."""
    labels = config.get("target_ylabels") or {}
    for key in (tag, tag.split("_", 1)[0]):    # match "dwl_8518750" or "dwl"
        if key in labels:
            return labels[key]
    return config.get("y_axis_label")


def _as_pst_config(config: dict, overrides: dict) -> PSTConfig:
    """Build a PSTConfig from the launcher dict + per-run overrides."""
    data = {k: v for k, v in config.items() if k not in _RESOLUTION_KEYS}
    data.update(overrides)
    return PSTConfig(**data)
