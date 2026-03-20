"""
cli/run_rtcs_fixed.py
======================
RTCS Selection (fixed k) — Select a Representative TC Subset (RTCS) with a
fixed number of storms that best fills both the TC parameter space (X) and
the hydrodynamic response space (Y).

Usage
-----
  1. Set CONFIG overrides below (or leave empty to use all defaults).
  2. Optionally set BBOX_CONFIG to restrict to a geographic region.
  3. Run:  python cli/run_rtcs_fixed.py

Input:   data/processed/tc_data.h5  (produced by cli/preprocess.py)
Outputs: data/processed/outputs/    (selected_storms.csv, selection_metrics.csv,
                                     pca_yspace_initial.png, pca_yspace_final.png,
                                     bbox_filter_map.png)

All default values are defined in config/defaults.py.
Only keys you want to change need to appear in CONFIG below.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.orch.python.workflows.rtcs_selection import run_rtcs_selection

# ---------------------------------------------------------------------------
# BOUNDING BOX  — set to None to disable geographic filtering
# ---------------------------------------------------------------------------
BBOX_CONFIG = {
    # Four corners of the region of interest (lat/lon in decimal degrees)
    "bbox": {
        "lat_min": 28.5,
        "lat_max": 31.0,
        "lon_min": -92.0,
        "lon_max": -88.5,
    },

    # Node coordinate source (the probQ .mat file from preprocessing)
    "node_coord_source":   str(_ROOT / "data/raw/lpv/CHS-LA_nodeID.mat"),
    "node_coord_variable": "nodeID",
    "node_id_col": 0,        # column with main node IDs (must match IDs stored in HDF5)
    "lat_col":     2,        # column with latitude
    "lon_col":     3,        # column with longitude

    # TC track directory and file naming
    "track_dir":          str(_ROOT / "data/raw/itcs_tropfiles/chs-la"),
    "track_file_pattern": "LACPR2_JPM{:04d}_TROP.txt",

    # Maximum radial distance (km) from bbox centroid for storm inclusion
    "max_track_dist_km": 200,
}
# BBOX_CONFIG = None   # ← uncomment to disable bbox filtering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK  — only override what you need; omit the rest
# ---------------------------------------------------------------------------
CONFIG = {
    "h5_path":    str(_ROOT / "data/processed/tc_data.h5"),
    "output_dir": str(_ROOT / "data/processed/outputs"),

    # Additional storms to select ON TOP of pre-selected (total = pre + additional)
    "k_additional": 200,

    # Node subsampling (None = use all nodes)
    "node_stride": 100,

    # TC parameter SPLOM — 0-based column indices in X (Param_MT)
    "splom_params": [6, 7, 8, 9],
    "splom_labels": ["Dp (hPa)", "Rm (km)", "Vt (km/h)", "Hd (deg)"],
    "splom_title":  "TC Parameters",

    # Pre-selected storms to build upon (shown as blue in SPLOM)
    "pre_selected_csv": str(_ROOT / "data/raw/lpv/LACS_100_TC_Subset.csv"),
    # "pre_selected_indices": [0, 5, 42],   # alternative: explicit list

    # Joint matrix weights
    # "alpha_default": 1.0,
    # "beta_default":  1.0,

    # DSW aggregation method — how nodal DSWs are averaged into a global weight:
    #   1 = Simple mean: every node counts equally (classic JPM-OS).
    #   2 = Surge-weighted: each node's DSW contribution to storm j is weighted
    #       by storm j's surge at that node.  Storm-specific — a storm's weight
    #       is dominated by nodes where it actually produces large surge.
    #   3 = Variance-weighted: each node gets a fixed weight = variance of surge
    #       across all storms.  Nodes with high response variability (where storm
    #       ranking matters most for HC reconstruction) dominate the average.
    "dsw_method": 3,
}
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Apply bounding-box filter (if configured) ──────────────────────
    if BBOX_CONFIG is not None:
        from backend.geo.bbox_filter import apply_bbox_filter
        from backend.geo.track_map import plot_bbox_map

        result = apply_bbox_filter(
            BBOX_CONFIG,
            CONFIG["h5_path"],
            CONFIG["output_dir"],
        )

        # Inject filtered indices into CONFIG for the workflow
        CONFIG["bbox_node_col_indices"] = result["node_col_indices"]
        CONFIG["bbox_storm_indices"]    = result["storm_indices"]

        # Generate diagnostic map
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
            output_dir=CONFIG["output_dir"],
        )

    indices, metrics = run_rtcs_selection(CONFIG)
