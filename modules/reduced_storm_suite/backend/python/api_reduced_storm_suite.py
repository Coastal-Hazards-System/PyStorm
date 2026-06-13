"""api_reduced_storm_suite - orchestrator entry (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_reduced_storm_suite.py`` at the module root holds only the operator-edited
option block (the per-dataset registries, the bbox window, and the tuning
config) and hands it to ``launch_batch`` here.

This entry owns everything procedural: (a) CLI / option resolution, (b) dataset
resolution and batch iteration over study areas (per CyHAN §5.3, dataset
resolution and batch marshaling are orchestration responsibilities, not the
launcher's), (c) path wiring and the auto-bootstrap of ``tc_data.h5``, (d) the
optional bbox filter, (e) per-mode output sub-directory routing, and (f) the
mode dispatch into ``workflows.rss_selection`` or
``workflows.growth_evaluation``. The substantive workflow itself is expanded
into the ``reduced_storm_suite`` package.

Public API
----------
  launch_batch(root, default_dataset, default_mode, default_scope,
               raw_files_by_dataset, units_by_dataset, preprocess_metadata,
               track_file_patterns, bbox, config)  -> int
      Batch entry the launcher calls. Parses the CLI (--dataset/--mode/--scope),
      resolves each dataset's raw files + vertical datum from the registries,
      and runs ``launch`` once per dataset. Returns an aggregate exit code.
  launch(root, dataset, mode, scope, raw_files, preprocess_metadata,
         track_file_patterns, bbox, config)  -> int
      Per-dataset entry: resolves paths, bootstraps the store, builds the bbox
      config, and dispatches to ``run``.
  run(config, bbox_config, mode, scope)  ->  workflow result
      Lower-level orchestration step (assumes config is already wired).
"""

import argparse
from pathlib import Path
from typing  import Any, Dict, Mapping, Optional


# Fallback TROP naming when a dataset is absent from track_file_patterns.
_DEFAULT_TRACK_FILE_PATTERN = "LACPR2_JPM{:04d}_TROP.txt"


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


def run(
    config:      Dict[str, Any],
    bbox_config: Optional[Mapping[str, Any]] = None,
    mode:        str                         = "fixed",
    scope:       str                         = "local",
) -> Any:
    """Execute one RSS selection.

    Parameters
    ----------
    config : dict
        Operator configuration (paths, k parameters, ...). A shallow copy is
        made before mutation; the caller's dict is not modified.
    bbox_config : Mapping or None
        Geographic bounding-box configuration. When provided AND scope is
        "local", the bbox filter is applied before selection and a diagnostic
        map is rendered. Ignored when scope is "regional".
    mode : {"fixed", "optimal"}
        Selection algorithm to run.
    scope : {"local", "regional"}
        Geographic scope. "local" applies bbox_config; "regional" forces a
        basin-wide run (all nodes, all storms) regardless of bbox_config.

    Returns
    -------
    Whatever the dispatched workflow returns (see
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


def _parse_args(
    default_dataset: str, default_mode: str, default_scope: str,
    available_datasets: Mapping[str, Any],
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_reduced_storm_suite.py",
        description="Run the Reduced Storm Suite headless. With no --dataset it "
                    "uses the DATASET option in the launcher; pass one or more "
                    "registered dataset keys to batch over study areas.")
    p.add_argument(
        "--dataset", nargs="+", metavar="KEY", default=None,
        help=f"Registered dataset key(s) to run (batch). Available: "
             f"{sorted(available_datasets)}. Default: {default_dataset!r}.")
    p.add_argument(
        "--mode", choices=["fixed", "optimal"], default=default_mode,
        help=f"Selection mode (default: {default_mode}, set by MODE constant).")
    p.add_argument(
        "--scope", choices=["local", "regional"], default=default_scope,
        help=f"Geographic scope (default: {default_scope}, set by SCOPE "
             f"constant). 'regional' ignores BBOX and uses all nodes/storms.")
    p.add_argument(
        "--storm-type", choices=["tc", "etc"], default=None,
        help="Storm type (tc = tropical, implemented; etc = placeholder). "
             "Default: the STORM_TYPE constant.")
    return p.parse_args()


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


def launch_batch(
    root:                 Path,
    default_dataset:      str,
    default_mode:         str,
    default_scope:        str,
    raw_files_by_dataset: Mapping[str, Any],
    units_by_dataset:     Mapping[str, str],
    preprocess_metadata:  Mapping[str, Any],
    track_file_patterns:  Mapping[str, str],
    bbox:                 Mapping[str, Any],
    config:               Dict[str, Any],
) -> int:
    """Operator entry point. Parses the CLI, resolves the dataset(s) to run from
    the launcher registries, and dispatches ``launch`` once per dataset.

    With no --dataset the single ``default_dataset`` runs; one or more --dataset
    keys batch over study areas. Per CyHAN §5.3 the dataset resolution and the
    batch loop are orchestration responsibilities and live here, not in the
    launcher. Returns the last non-zero per-dataset exit code (0 if all ran).
    """
    args = _parse_args(default_dataset, default_mode, default_scope,
                       raw_files_by_dataset)
    datasets = args.dataset or [default_dataset]

    cfg = dict(config)
    if args.storm_type:
        cfg["storm_type"] = args.storm_type
    _check_storm_type(cfg)   # raise early for the etc placeholder, before any data work

    rc = 0
    for ds in datasets:
        raw_files, preprocess_meta = _resolve_dataset(
            ds, raw_files_by_dataset, units_by_dataset, preprocess_metadata)
        if len(datasets) > 1:
            print(f"\n{'#' * 64}\n#  dataset: {ds}\n{'#' * 64}")
        rc = launch(
            root                = root,
            dataset             = ds,
            mode                = args.mode,
            scope               = args.scope,
            raw_files           = raw_files,
            preprocess_metadata = preprocess_meta,
            track_file_patterns = track_file_patterns,
            bbox                = bbox,
            config              = cfg,
        ) or rc
    return rc


def launch(
    root:                Path,
    dataset:             str,
    mode:                str,
    scope:               str,
    raw_files:           Mapping[str, Any],
    preprocess_metadata: Mapping[str, Any],
    track_file_patterns: Mapping[str, str],
    bbox:                Mapping[str, Any],
    config:              Dict[str, Any],
) -> int:
    """Run one RSS selection for a single resolved dataset with already-resolved
    ``mode`` and ``scope``. Resolves per-dataset paths, auto-bootstraps the
    store, builds the bbox config for local scope, then dispatches to ``run``.
    Called once per dataset by ``launch_batch``.
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

    run(config=cfg, bbox_config=bbox_config, mode=mode, scope=scope)
    return 0
