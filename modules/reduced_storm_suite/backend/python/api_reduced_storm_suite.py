"""api_reduced_storm_suite - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_reduced_storm_suite.py`` at the module root holds the operator-edited
option block (the per-dataset registries, the bbox window, and the tuning
config), parses the CLI, and hands a single ``config`` dict to ``run`` here.

This entry owns everything procedural: (a) option resolution, (b) dataset
resolution and batch iteration over study areas (per CyHAN §5.3, dataset
resolution and batch marshaling are orchestration responsibilities, not the
launcher's), (c) path wiring and the auto-bootstrap of ``tc_data.h5``, (d) the
optional bbox filter, (e) per-mode output sub-directory routing, and (f) the
mode dispatch into ``workflows.rss_selection`` or
``workflows.growth_evaluation``. The substantive workflow itself is expanded
into the ``reduced_storm_suite`` package.

Public API
----------
  run(config)  ->  RSSResult | dict[str, RSSResult]
      The single module entry point (matches every PyStorm module's
      ``run(config) -> <Module>Result`` contract). ``config`` carries the
      operator option block plus the orchestration controls: ``root``,
      ``dataset``/``datasets``, ``mode``, ``scope``, ``storm_type``,
      ``raw_files_by_dataset``, ``units_by_dataset``, ``preprocess_metadata``,
      ``track_file_patterns``, ``bbox``; every other key is selection tuning
      passed through to the workflow. Per CyHAN §5.3 dataset resolution and the
      batch loop over study areas live here, in the orchestrator. Returns a
      single ``RSSResult`` for one dataset, or a ``{dataset: RSSResult}``
      mapping when batching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing  import Any, Dict, List, Mapping, Optional, Union


# Fallback TROP naming when a dataset is absent from track_file_patterns.
_DEFAULT_TRACK_FILE_PATTERN = "LACPR2_JPM{:04d}_TROP.txt"

# Config keys that steer orchestration (consumed by run / _launch_one). Every
# other key in the config is selection tuning, passed through to the workflow.
# (storm_type is intentionally NOT here: it is validated by run() but also flows
# through to the workflow, preserving the pre-alignment behavior.)
_ORCH_KEYS = frozenset({
    "root", "dataset", "datasets", "mode", "scope",
    "raw_files_by_dataset", "units_by_dataset", "preprocess_metadata",
    "track_file_patterns", "bbox",
})


@dataclass
class RSSResult:
    """Outcome of one RSS selection (a single dataset)."""
    dataset:    str
    mode:       str
    scope:      str
    output_dir: str
    result:     Any        # the workflow's native return, e.g. (indices, metrics)


def _check_storm_type(config: Mapping[str, Any]) -> None:
    """Validate the storm-type mode. tc = tropical cyclone (implemented); etc is a
    placeholder (the same k-medoids / DSW / HC selection would run on an ETC
    synthetic-storm set, from an ETC storm source)."""
    st = str(config.get("storm_type", "tc")).lower()
    if st not in ("tc", "etc"):
        raise ValueError(f"storm_type must be 'tc' or 'etc'; got {st!r}")
    if st == "etc":
        raise NotImplementedError(
            "storm_type='etc' (extratropical-cyclone suite) is a placeholder and "
            "not yet implemented; use storm_type='tc'.")


def _run_selection(
    config:      Dict[str, Any],
    *,
    bbox_config: Optional[Mapping[str, Any]] = None,
    mode:        str                         = "fixed",
    scope:       str                         = "local",
) -> Any:
    """Execute one RSS selection for an already-wired config (paths resolved).

    Internal per-dataset step used by ``run`` -> ``_launch_one``. Routes the
    output sub-directory by scope/mode, applies the bbox filter for local scope
    (regional forces ``bbox_config`` to None), and dispatches to the fixed-k or
    growth-sweep workflow. A shallow copy of ``config`` is made before mutation.
    Returns the workflow's native result (see
    ``reduced_storm_suite.workflows.rss_selection.run_rss_selection`` /
    ``...growth_evaluation.run_growth_evaluation``).
    """
    if mode not in ("fixed", "optimal"):
        raise ValueError(f"mode must be 'fixed' or 'optimal'; got {mode!r}")
    if scope not in ("local", "regional"):
        raise ValueError(f"scope must be 'local' or 'regional'; got {scope!r}")
    _check_storm_type(config)

    cfg = dict(config)
    # Per-scope / per-mode output subdirectory so no two runs (local vs
    # regional, fixed vs optimal) overwrite each other's results.
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / scope / mode)

    if scope == "regional":
        bbox_config = None

    if bbox_config is not None:
        _apply_bbox(cfg, bbox_config)

    if mode == "fixed":
        from reduced_storm_suite.workflows.rss_selection import run_rss_selection
        return run_rss_selection(cfg)

    from reduced_storm_suite.workflows.growth_evaluation import run_growth_evaluation
    return run_growth_evaluation(cfg)


def _apply_bbox(cfg: Dict[str, Any], bbox_config: Mapping[str, Any]) -> None:
    """Run the geographic bounding-box filter; write the diagnostic map."""
    from reduced_storm_suite.geo.bbox_filter import apply_bbox_filter
    from reduced_storm_suite.geo.track_map   import plot_bbox_map

    result = apply_bbox_filter(
        bbox_config,
        cfg["h5_path"],
        cfg["output_dir"],
    )

    cfg["bbox_node_col_indices"] = result["node_col_indices"]
    cfg["bbox_storm_indices"]    = result["storm_indices"]

    plot_bbox_map(
        bbox             = bbox_config["bbox"],
        all_node_lats    = result["all_node_lats"],
        all_node_lons    = result["all_node_lons"],
        bbox_node_lats   = result["bbox_node_lats"],
        bbox_node_lons   = result["bbox_node_lons"],
        tracks           = result["tracks"],
        storm_indices_near = result["storm_indices"],
        medoid_lat       = result["medoid_lat"],
        medoid_lon       = result["medoid_lon"],
        max_dist_km      = bbox_config.get("max_track_dist_km", 200),
        output_dir       = cfg["output_dir"],
    )


# ===========================================================================
# Launcher-side logic  (path wiring, bootstrap, CLI) - moved out of the
# operator-facing run_reduced_storm_suite.py so the launcher holds only
# user options. All functions take the launcher's declarative data as
# explicit arguments; none read module globals.
# ===========================================================================

def _raw_path(
    root: Path, dataset: str, raw_files: Mapping[str, Any], name: str,
    *, required: bool = True,
) -> Optional[str]:
    """Resolve raw_files[name] to an absolute path under raw/<dataset>/.

    required=True raises if the entry is None; required=False returns None.
    Either way, a non-None entry naming a missing file always raises, so
    typos can never silently pass.
    """
    if name not in raw_files:
        raise KeyError(
            f"raw_files has no entry '{name}'. Known keys: "
            f"{sorted(raw_files.keys())}")
    fname = raw_files[name]
    if fname is None:
        if required:
            raise ValueError(f"raw_files['{name}'] must be set (got None).")
        return None
    p = Path(root) / "data" / "inputs" / "raw" / dataset / fname
    if not p.is_file():
        raise FileNotFoundError(
            f"raw_files['{name}'] = '{fname}' not found at {p}. "
            f"Check the filename and that the file exists in {p.parent}.")
    return str(p)


def _build_preprocess_config(
    root: Path, dataset: str, raw_files: Mapping[str, Any],
    preprocess_metadata: Mapping[str, Any],
    track_file_patterns: Mapping[str, str], h5_path: str,
) -> Dict[str, Any]:
    """Assemble the Preprocessor config from the required raw paths and the
    preprocess metadata. Per-Y-row storm IDs are derived from the TROP
    filenames when the X source carries none (skipped if the folder is
    absent/empty, so a regional-only dataset still builds)."""
    return {
        "output_path":            h5_path,
        "X_source":               _raw_path(root, dataset, raw_files, "x_param_table"),
        "Y_source":               _raw_path(root, dataset, raw_files, "y_surge"),
        "Y_node_filter_source":   _raw_path(root, dataset, raw_files, "nodeID"),
        "HC_source":              _raw_path(root, dataset, raw_files, "hc_benchmark"),
        "storm_id_track_dir":     str(Path(root) / "data" / "inputs" / "raw"
                                      / "itcs_tropfiles" / dataset),
        "storm_id_track_pattern": track_file_patterns.get(
                                      dataset, _DEFAULT_TRACK_FILE_PATTERN),
        **preprocess_metadata,
    }


def _ensure_h5_exists(
    root: Path, dataset: str, raw_files: Mapping[str, Any],
    preprocess_metadata: Mapping[str, Any],
    track_file_patterns: Mapping[str, str], h5_path: str,
) -> None:
    """Build tc_data.h5 in-process if it's missing for the active dataset."""
    h5 = Path(str(h5_path))
    if h5.is_file():
        return

    print(f"\n[bootstrap] {h5.name} not found for dataset '{dataset}'.")
    print(f"[bootstrap] Building it via the preprocessor ...\n")

    from reduced_storm_suite.workflows.ingest import Preprocessor
    Preprocessor(_build_preprocess_config(
        root, dataset, raw_files, preprocess_metadata,
        track_file_patterns, h5_path)).run()

    if not h5.is_file():
        raise SystemExit(
            f"\n[bootstrap] Preprocessor finished but {h5} is still missing.")
    print(f"\n[bootstrap] Done - {h5.name} ready.\n")


def _build_bbox_config(
    root: Path, dataset: str, raw_files: Mapping[str, Any],
    track_file_patterns: Mapping[str, str], bbox: Mapping[str, Any],
) -> Dict[str, Any]:
    """Complete the operator's bbox option block with resolved paths: the
    node-coord source, the per-storm track folder, and the dataset's TROP
    filename pattern. Only invoked for a local-scope run."""
    full = dict(bbox)
    full["node_coord_source"]  = _raw_path(root, dataset, raw_files, "nodeID")
    full["track_dir"]          = str(Path(root) / "data" / "inputs" / "raw"
                                     / "itcs_tropfiles" / dataset)
    full["track_file_pattern"] = track_file_patterns.get(
                                     dataset, _DEFAULT_TRACK_FILE_PATTERN)
    return full


def _resolve_dataset(
    ds:                   str,
    raw_files_by_dataset: Mapping[str, Any],
    units_by_dataset:     Mapping[str, str],
    preprocess_metadata:  Mapping[str, Any],
) -> tuple:
    """Per-dataset (raw_files, preprocess_metadata) from the launcher registries.

    The CLI batches by dataset KEY (not file path) because RSS needs the
    dataset's raw filenames and vertical datum, which live in the launcher's
    registries. The datum is injected into a copy of the metadata template so it
    tracks the dataset.
    """
    try:
        raw = raw_files_by_dataset[ds]
    except KeyError:
        raise SystemExit(
            f"--dataset {ds!r} has no entry in RAW_FILES_BY_DATASET; "
            f"available: {sorted(raw_files_by_dataset)}")
    try:
        units = units_by_dataset[ds]
    except KeyError:
        raise SystemExit(
            f"--dataset {ds!r} has no entry in UNITS_BY_DATASET; "
            f"available: {sorted(units_by_dataset)}")
    meta = dict(preprocess_metadata)
    meta["Y_units"]  = units      # datum tracks the dataset
    meta["HC_units"] = units
    return raw, meta


def run(config: Dict[str, Any]) -> Union[RSSResult, Dict[str, RSSResult]]:
    """Single RSS entry point (see the module docstring for the config schema).

    Validates the orchestration controls, resolves the dataset(s) from the
    launcher registries, and runs the selection once per dataset (the batch loop
    and dataset resolution stay here in the orchestrator, per CyHAN §5.3).
    Returns one ``RSSResult`` for a single dataset, or a ``{dataset: RSSResult}``
    mapping for a batch. The caller's dict is not modified.
    """
    cfg = dict(config)

    # Validate controls first, before touching the registries, so a bad mode /
    # scope / storm_type is reported without needing the rest of the config.
    mode  = cfg.get("mode",  "fixed")
    scope = cfg.get("scope", "local")
    if mode not in ("fixed", "optimal"):
        raise ValueError(f"mode must be 'fixed' or 'optimal'; got {mode!r}")
    if scope not in ("local", "regional"):
        raise ValueError(f"scope must be 'local' or 'regional'; got {scope!r}")
    _check_storm_type(cfg)   # raises for the etc placeholder before any data work

    try:
        root                 = Path(cfg["root"])
        raw_files_by_dataset = cfg["raw_files_by_dataset"]
        units_by_dataset     = cfg["units_by_dataset"]
    except KeyError as exc:
        raise KeyError(f"run() config is missing required key {exc}") from None
    preprocess_metadata = cfg.get("preprocess_metadata", {})
    track_file_patterns = cfg.get("track_file_patterns", {})
    bbox                = cfg.get("bbox", {})

    datasets: List[str] = cfg.get("datasets") or [cfg["dataset"]]

    # Selection tuning = everything that is not an orchestration control; it is
    # exactly what the per-dataset workflow consumes.
    tuning = {k: v for k, v in cfg.items() if k not in _ORCH_KEYS}

    results: Dict[str, RSSResult] = {}
    for ds in datasets:
        if len(datasets) > 1:
            print(f"\n{'#' * 64}\n#  dataset: {ds}\n{'#' * 64}")
        raw_files, preprocess_meta = _resolve_dataset(
            ds, raw_files_by_dataset, units_by_dataset, preprocess_metadata)
        results[ds] = _launch_one(
            root, ds, mode, scope, raw_files, preprocess_meta,
            track_file_patterns, bbox, tuning)

    return results[datasets[0]] if len(datasets) == 1 else results


def _launch_one(
    root:                Path,
    dataset:             str,
    mode:                str,
    scope:               str,
    raw_files:           Mapping[str, Any],
    preprocess_metadata: Mapping[str, Any],
    track_file_patterns: Mapping[str, str],
    bbox:                Mapping[str, Any],
    config:              Dict[str, Any],
) -> RSSResult:
    """Run one RSS selection for a single resolved dataset. Resolves per-dataset
    paths, auto-bootstraps the store, builds the bbox config for local scope,
    then dispatches to ``_run_selection``. Called once per dataset by ``run``.
    """
    cfg = dict(config)
    # Per-dataset paths live here (derived from the standard layout), not in
    # the operator's options block.
    cfg["h5_path"]    = str(Path(root) / "data" / "inputs" / "processed"
                            / dataset / "tc_data.h5")
    cfg["output_dir"] = str(Path(root) / "data" / "outputs" / dataset)
    # The launcher option key is 'pre_selected_storms'; the backend key is
    # 'pre_selected_csv'. Bridge them here. Optional - None means none.
    cfg["pre_selected_csv"] = _raw_path(
        root, dataset, raw_files, "pre_selected_storms", required=False)

    # Auto-bootstrap tc_data.h5 if missing (no-op when present).
    _ensure_h5_exists(root, dataset, raw_files, preprocess_metadata,
                      track_file_patterns, cfg["h5_path"])

    # Build the bbox dict only when actually needed; regional skips it so a
    # dataset with no bbox / track files can still complete a regional run.
    bbox_config = (_build_bbox_config(root, dataset, raw_files,
                                      track_file_patterns, bbox)
                   if scope == "local" else None)

    result = _run_selection(cfg, bbox_config=bbox_config, mode=mode, scope=scope)
    out_dir = str(Path(root) / "data" / "outputs" / dataset / scope / mode)
    return RSSResult(dataset=dataset, mode=mode, scope=scope,
                     output_dir=out_dir, result=result)
