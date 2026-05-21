"""storm_selection — canonical module launcher (CyHAN v1.1 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Selects a Reduced Tropical Cyclone Suite (RTCS) from a tc_data.h5 store.
Two modes are supported:

  fixed    : Select a fixed number of storms (k_additional on top of any
             pre-selected forced storms).  Runs once.
  optimal  : Sweep k from k_min to k_max evaluating DSW + HC at each step,
             then pick the smallest k whose global RMSE is at or below
             rmse_threshold (or argmin RMSE if no k meets it).

Usage
-----
    python scripts/run_storm_selection.py                 # uses MODE constant
    python scripts/run_storm_selection.py --mode fixed    # CLI override
    python scripts/run_storm_selection.py --mode optimal

Input  : data/tc_data.h5  (produced by scripts/preprocess.py)
Outputs: data/processed/outputs/<mode>/    — fixed/ or optimal/ subdirectory
                                             keyed by MODE so the two modes
                                             never overwrite each other.

All user-editable options live in the USER OPTIONS block below.  Values from
storm_selection.config.defaults.RTCS_SELECTION_DEFAULTS are inherited; only
override what you need.
"""

import argparse
import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================

# ---------------------------------------------------------------------------
# MODE  — selection algorithm
#   "fixed"   = pick a fixed k (single run, full diagnostic suite)
#   "optimal" = sweep k from k_min to k_max; pick smallest k meeting RMSE tolerance
# Override at runtime with the --mode CLI flag.
# ---------------------------------------------------------------------------
MODE = "optimal"


# ---------------------------------------------------------------------------
# BOUNDING BOX  — set to None to disable geographic filtering
# ---------------------------------------------------------------------------
BBOX_CONFIG = {
    "bbox": {
        "lat_min": 28.5,
        "lat_max": 31.0,
        "lon_min": -92.0,
        "lon_max": -88.5,
    },

    "node_coord_source":   str(_MODULE_ROOT / "data/raw/lpv/CHS-LA_nodeID.mat"),
    "node_coord_variable": "nodeID",
    "node_id_col": 0,
    "lat_col":     2,
    "lon_col":     3,

    "track_dir":          str(_MODULE_ROOT / "data/raw/itcs_tropfiles/chs-la"),
    "track_file_pattern": "LACPR2_JPM{:04d}_TROP.txt",

    "max_track_dist_km": 200,
}
# BBOX_CONFIG = None   # ← uncomment to disable bbox filtering


# ---------------------------------------------------------------------------
# CONFIG  — only override what you need; omit the rest
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":    str(_MODULE_ROOT / "data/processed/tc_data.h5"),
    "output_dir": str(_MODULE_ROOT / "data/processed/outputs"),

    # ── fixed-mode parameters ─────────────────────────────────────────────
    "k_additional": 200,

    # Sub-RTCS (optional second-stage subset).  See defaults.py for modes.
    "k_sub_rtcs":     50,
    "sub_rtcs_mode": "within_maximin",

    # ── shared parameters ─────────────────────────────────────────────────
    "node_stride": 100,

    "splom_params": [6, 7, 8, 9],
    "splom_labels": ["Dp (hPa)", "Rm (km)", "Vt (km/h)", "Hd (deg)"],
    "splom_title":  "TC Parameters",

    "pre_selected_csv": str(_MODULE_ROOT / "data/raw/lpv/LACS_100_TC_Subset.csv"),

    "dsw_method": 3,

    # ── optimal-mode parameters (only used when MODE == "optimal") ────────
    # Sweep k from k_min to k_max (step k_step), then pick the smallest k
    # whose global HC RMSE is at or below rmse_threshold (units: m).
    # If no k meets the tolerance, the k with the lowest RMSE is selected
    # and a warning is emitted.
    "k_min":             20,
    "k_max":            300,
    "k_step":             5,
    "rmse_threshold":  0.10,
    "bias_report_rp": [10, 100, 1000],
}

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================


def _apply_bbox(cfg):
    """Run the geographic bounding-box filter and write the diagnostic map."""
    from storm_selection.geo.bbox_filter import apply_bbox_filter
    from storm_selection.geo.track_map import plot_bbox_map

    result = apply_bbox_filter(
        BBOX_CONFIG,
        cfg["h5_path"],
        cfg["output_dir"],
    )

    cfg["bbox_node_col_indices"] = result["node_col_indices"]
    cfg["bbox_storm_indices"]    = result["storm_indices"]

    plot_bbox_map(
        bbox=BBOX_CONFIG["bbox"],
        all_node_lats=result["all_node_lats"],
        all_node_lons=result["all_node_lons"],
        bbox_node_lats=result["bbox_node_lats"],
        bbox_node_lons=result["bbox_node_lons"],
        tracks=result["tracks"],
        storm_indices_near=result["storm_indices"],
        medoid_lat=result["medoid_lat"],
        medoid_lon=result["medoid_lon"],
        max_dist_km=BBOX_CONFIG.get("max_track_dist_km", 200),
        output_dir=cfg["output_dir"],
    )


def main():
    parser = argparse.ArgumentParser(prog="run_storm_selection.py")
    parser.add_argument(
        "--mode", choices=["fixed", "optimal"], default=MODE,
        help=f"Selection mode (default: {MODE}, set by MODE constant in the script).",
    )
    args = parser.parse_args()

    cfg = dict(CONFIG)
    # Per-mode output subdirectory so fixed/optimal runs don't overwrite
    # each other's results.
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / args.mode)

    if BBOX_CONFIG is not None:
        _apply_bbox(cfg)

    if args.mode == "fixed":
        from storm_selection.workflows.rtcs_selection import run_rtcs_selection
        return run_rtcs_selection(cfg)

    from storm_selection.workflows.growth_evaluation import run_growth_evaluation
    return run_growth_evaluation(cfg)


if __name__ == "__main__":
    main()
