"""
cli/run_rtcs_optimal.py
========================
RTCS Selection (optimal k) — Sweep subset size from k_initial to k_max,
computing DSW weights and reconstructing hazard curves at each step to find
the optimal number of storms.  Quantifies bias relative
to benchmark HCs at the full AER table and at reporting return periods.

Usage
-----
  1. Run cli/preprocess.py first to produce data/processed/tc_data.h5.
  2. Set CONFIG overrides below.
  3. Optionally set BBOX_CONFIG to restrict to a geographic region.
  4. Run:  python cli/run_rtcs_optimal.py

Input:   data/processed/tc_data.h5  (must contain /HC benchmark curves)
Outputs: data/processed/outputs/    (selected_storms.csv, growth_history.csv,
                                     growth_evaluation.png, bbox_filter_map.png)

All default values are defined in config/defaults.py.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.orch.python.workflows.growth_evaluation import run_growth_evaluation

# ---------------------------------------------------------------------------
# BOUNDING BOX  — set to None to disable geographic filtering
# ---------------------------------------------------------------------------
BBOX_CONFIG = {
    # Four corners of the region of interest (lat/lon in decimal degrees)
    "bbox": {
        "lat_min": 29.0,
        "lat_max": 30.5,
        "lon_min": -91.0,
        "lon_max": -89.0,
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

    # Node subsampling (1 = use all nodes, N = keep every Nth node)
    "node_stride": 1,

    # Growth loop bounds
    "k_initial": 20,
    "k_max":    100,
    # "k_step":    5,

    # Return periods (years) to report bias at
    "bias_report_rp": [10, 100, 1000],

    # Pre-selected storms to build upon (optional — pick one or neither)
    # "pre_selected_csv":     str(_ROOT / "data/processed/outputs/selected_storms.csv"),
    # "pre_selected_indices": [0, 5, 42],   # explicit list of original indices

    # Optional RMSE stopping threshold (m); None = run to k_max
    # "rmse_threshold": 0.20,

    # Joint matrix weights
    # "alpha_default": 1.0,
    # "beta_default":  1.0,
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

    indices, history_df = run_growth_evaluation(CONFIG)
