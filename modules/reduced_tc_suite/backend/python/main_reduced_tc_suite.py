"""main_reduced_tc_suite — orchestrator entry (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Non-user-facing realization of the Python Orchestration role (§4.2).
``run_reduced_tc_suite.py`` at the module root imports ``run`` from this
file with the operator-edited configuration, the optional bounding-box
configuration, and the selected mode (``"fixed"`` or ``"optimal"``).

The substantive workflow is already expanded into the
``reduced_tc_suite`` package — this entry composes (a) CLI / option
resolution and path wiring, (b) the auto-bootstrap of ``tc_data.h5``,
(c) the optional bbox filter, (d) the per-mode output sub-directory
routing, and (e) the mode dispatch into ``workflows.rtcs_selection`` or
``workflows.growth_evaluation``. None of that logic lives in the launcher;
``run_reduced_tc_suite.py`` holds only the operator-edited options and a
single call to ``launch``.

Public API
----------
  launch(root, dataset, default_mode, default_scope, raw_files,
         preprocess_metadata, track_file_patterns, bbox, config)  -> int
      Full entry point the launcher calls: resolves CLI args + paths,
      bootstraps the store, builds the bbox config, and dispatches to run().
  run(config, bbox_config, mode, scope)  ->  workflow result
      Lower-level orchestration step (assumes config is already wired).
"""

import argparse
from pathlib import Path
from typing  import Any, Dict, Mapping, Optional


# Fallback TROP naming when a dataset is absent from track_file_patterns.
_DEFAULT_TRACK_FILE_PATTERN = "LACPR2_JPM{:04d}_TROP.txt"


def run(
    config:      Dict[str, Any],
    bbox_config: Optional[Mapping[str, Any]] = None,
    mode:        str                         = "fixed",
    scope:       str                         = "local",
) -> Any:
    """Execute one RTCS selection.

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
    ``reduced_tc_suite.workflows.rtcs_selection.run_rtcs_selection`` /
    ``...growth_evaluation.run_growth_evaluation``).
    """
    if mode not in ("fixed", "optimal"):
        raise ValueError(f"mode must be 'fixed' or 'optimal'; got {mode!r}")
    if scope not in ("local", "regional"):
        raise ValueError(f"scope must be 'local' or 'regional'; got {scope!r}")

    cfg = dict(config)
    # Per-scope / per-mode output subdirectory so no two runs (local vs
    # regional, fixed vs optimal) overwrite each other's results.
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / scope / mode)

    if scope == "regional":
        bbox_config = None

    if bbox_config is not None:
        _apply_bbox(cfg, bbox_config)

    if mode == "fixed":
        from reduced_tc_suite.workflows.rtcs_selection import run_rtcs_selection
        return run_rtcs_selection(cfg)

    from reduced_tc_suite.workflows.growth_evaluation import run_growth_evaluation
    return run_growth_evaluation(cfg)


def _apply_bbox(cfg: Dict[str, Any], bbox_config: Mapping[str, Any]) -> None:
    """Run the geographic bounding-box filter; write the diagnostic map."""
    from reduced_tc_suite.geo.bbox_filter import apply_bbox_filter
    from reduced_tc_suite.geo.track_map   import plot_bbox_map

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
# Launcher-side logic  (path wiring, bootstrap, CLI) — moved out of the
# operator-facing run_reduced_tc_suite.py so the launcher holds only
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

    from reduced_tc_suite.workflows.ingest import Preprocessor
    Preprocessor(_build_preprocess_config(
        root, dataset, raw_files, preprocess_metadata,
        track_file_patterns, h5_path)).run()

    if not h5.is_file():
        raise SystemExit(
            f"\n[bootstrap] Preprocessor finished but {h5} is still missing.")
    print(f"\n[bootstrap] Done — {h5.name} ready.\n")


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


def _parse_args(default_mode: str, default_scope: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_reduced_tc_suite.py")
    p.add_argument(
        "--mode", choices=["fixed", "optimal"], default=default_mode,
        help=f"Selection mode (default: {default_mode}, set by MODE constant).")
    p.add_argument(
        "--scope", choices=["local", "regional"], default=default_scope,
        help=f"Geographic scope (default: {default_scope}, set by SCOPE "
             f"constant). 'regional' ignores BBOX and uses all nodes/storms.")
    return p.parse_args()


def launch(
    root:                Path,
    dataset:             str,
    default_mode:        str,
    default_scope:       str,
    raw_files:           Mapping[str, Any],
    preprocess_metadata: Mapping[str, Any],
    track_file_patterns: Mapping[str, str],
    bbox:                Mapping[str, Any],
    config:              Dict[str, Any],
) -> int:
    """Operator entry point. Resolves CLI overrides and per-dataset paths,
    auto-bootstraps the store, builds the bbox config for local scope, then
    dispatches to ``run``. All inputs are the launcher's declarative options.
    """
    args = _parse_args(default_mode, default_scope)

    cfg = dict(config)
    # Per-dataset paths live here (derived from the standard layout), not in
    # the operator's options block.
    cfg["h5_path"]    = str(Path(root) / "data" / "inputs" / "processed"
                            / dataset / "tc_data.h5")
    cfg["output_dir"] = str(Path(root) / "data" / "outputs" / dataset)
    # The launcher option key is 'pre_selected_storms'; the backend key is
    # 'pre_selected_csv'. Bridge them here. Optional — None means none.
    cfg["pre_selected_csv"] = _raw_path(
        root, dataset, raw_files, "pre_selected_storms", required=False)

    # Auto-bootstrap tc_data.h5 if missing (no-op when present).
    _ensure_h5_exists(root, dataset, raw_files, preprocess_metadata,
                      track_file_patterns, cfg["h5_path"])

    # Build the bbox dict only when actually needed; regional skips it so a
    # dataset with no bbox / track files can still complete a regional run.
    bbox_config = (_build_bbox_config(root, dataset, raw_files,
                                      track_file_patterns, bbox)
                   if args.scope == "local" else None)

    run(config=cfg, bbox_config=bbox_config, mode=args.mode, scope=args.scope)
    return 0
