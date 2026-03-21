"""
cli/run_rtcs_fixed.py
======================
RTCS Selection (fixed k) — Select a Representative TC Subset (RTCS) with a
fixed number of storms that best fills both the TC parameter space (X) and
the hydrodynamic response space (Y).

Pipeline
--------
  1. (Optional) Geographic bounding-box filter: restricts nodes and storms
     to a region of interest based on node coordinates and TC track proximity.
  2. Data loading: reads storm parameters (X), surge responses (Y), and
     benchmark hazard curves (HC) from the preprocessed HDF5 store.
  3. PCA dimensionality reduction on Y (retaining 95% variance by default).
  4. Bayesian optimization of the joint matrix weight w by minimizing HC
     reconstruction score over continuous log-space using a GP surrogate
     with Expected Improvement acquisition.
  5. K-medoids clustering on the joint matrix Z = [w*X~ | Y_r~] to select
     k storms maximizing spread in both parameter and response space.
  6. DSW back-computation and JPM hazard curve reconstruction at all nodes.
  7. Quantile Bias Mapping (QBM) post-correction of AER positions.

The DSW, HC reconstruction, and QBM engines use C++ with multi-threaded
parallelism and node-major memory layout for efficient operation at full
node count (>1M nodes).

Usage
-----
  1. Run cli/preprocess.py first to produce data/processed/tc_data.h5.
  2. Set CONFIG overrides below (or leave empty to use all defaults).
  3. Optionally set BBOX_CONFIG to restrict to a geographic region.
  4. Run:  python cli/run_rtcs_fixed.py

Input:   data/processed/tc_data.h5  (produced by cli/preprocess.py)
Outputs: data/processed/outputs/    (selected_storms.csv, selection_metrics.csv,
                                     pca_yspace_initial.png, pca_yspace_final.png,
                                     bbox_filter_map.png, hc_comparison.png,
                                     hc_comparison_qbm.png, qbm_bias.h5)

All default values are defined in config/defaults.py.
Only keys you want to change need to appear in CONFIG below.

Developed by: Norberto C. Nadal-Caraballo, PhD
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

    # Node coordinate source
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

    # Node subsampling (1 = use all nodes, N = keep every Nth node)
    "node_stride": 1,

    # TC parameter SPLOM — 0-based column indices in X (Param_MT)
    "splom_params": [6, 7, 8, 9],
    "splom_labels": ["Dp (hPa)", "Rm (km)", "Vt (km/h)", "Hd (deg)"],
    "splom_title":  "TC Parameters",

    # Pre-selected storms to build upon (shown as blue in SPLOM)
    "pre_selected_csv": str(_ROOT / "data/raw/lpv/LACS_100_TC_Subset.csv"),
    # "pre_selected_indices": [0, 5, 42],   # alternative: explicit list

    # Joint matrix weight  (w > 1 emphasises X; w < 1 emphasises Y_r)
    # "w_default": 10.0,

    # DSW aggregation method — how nodal DSWs are averaged into a global weight:
    #   1 = Simple mean: every node counts equally (classic JPM).
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
