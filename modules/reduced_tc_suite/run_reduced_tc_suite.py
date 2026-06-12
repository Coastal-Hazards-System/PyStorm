"""run_reduced_tc_suite — RTCS launcher (CyHAN v2.1 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry point for the Reduced TC Suite (formerly
``storm_selection``). The operator edits values in the USER OPTIONS block
below and runs the script. This file holds ONLY declarative options — every
code element (path wiring, store bootstrap, bbox assembly, CLI parsing,
dispatch) lives in ``main_reduced_tc_suite``; the launcher simply calls
``main_reduced_tc_suite.launch_batch`` with the option block per §5.3.

================================================================================
WHAT RTCS PRODUCES
================================================================================
RTCS selects a small, REPRESENTATIVE subset of synthetic tropical-cyclone storms
(a Reduced Tropical-Cyclone Suite, RTCS) from a large JPM storm set, so the
subset reproduces the full suite's coastal hazard with far fewer ADCIRC runs.
It reads the storms' TC parameters (X) and ADCIRC peak-surge fields (Y), builds
a joint feature space, picks k storms, derives their JPM-OS weights, and
verifies the reduced suite against benchmark hazard curves.

How it works (one pass)
  1. Build the joint feature matrix  Z = [ α·X̃ | β·Ỹ_r ]: standardize the TC
     parameters (X̃) and the PCA-reduced surge response (Ỹ_r), weighted by α / β.
  2. Select k storms in Z — "kmedoids" (representative) or "maximin"
     (space-filling); any pre-selected storms are always retained.
  3. Derive the discrete storm weights (DSW / QBM) so the subset reproduces the
     population hazard, and score hazard-curve reconstruction vs the benchmark
     (global and per-mean-return-interval RMSE / bias).
  4. (optimal mode) sweep k and pick the smallest that meets the HC RMSE
     tolerance; (AB_SWEEP) search α / β for the best HC reconstruction.
  Emits selection CSVs, diagnostic plots (SPLOM, PCA y-space, HC, QBM), and the
  QBM weight store.

Selection modes
---------------
  fixed    — single run with k = (pre-selected count) + k_additional. Emits
             the full diagnostic suite (SPLOM, PCA y-space, HC verification,
             QBM bias surface, optional sub-RTCS).
  optimal  — sweep k from k_min to k_max in steps of k_step. Pick the
             smallest k whose global HC RMSE is ≤ rmse_threshold; if no k
             meets the tolerance, pick argmin RMSE and emit a warning.

Geographic scopes
-----------------
  local    — apply BBOX: keep only nodes inside the bounding box AND only
             storms whose tracks pass within max_track_dist_km of the bbox
             medoid. Reads node_coord_source and the per-storm TROP track
             files; renders a bbox diagnostic map.
  regional — basin-wide run. BBOX is ignored — every node and every storm
             in tc_data.h5 is used. The bbox node-coord file and track files
             are not touched.

Bootstrapping
-------------
If the processed ``tc_data.h5`` store is missing for the active DATASET,
the launcher invokes the in-process Preprocessor automatically, using
RAW_FILES_BY_DATASET to locate the raw source `.mat` files and PREPROCESS_METADATA
to decode them.

Y-array cleanup
---------------
``read_store`` converts ADCIRC dry-node sentinels (-99999.0) to NaN on
load so every downstream consumer sees a uniform missing-data marker.
PCA further drops always-NaN nodes and zero-fills the rest (configurable
via ``pca_dry_strategy`` in CONFIG / defaults.py).

Run (headless / CLI)
--------------------
Headless by design — figures (SPLOM, PCA y-space, HC, QBM) are written to disk
(no window opens), so this runs unchanged over SSH, in a container, or cron.

  1. Install dependencies once:
         pip install -r requirements.txt
  2. Edit the USER OPTIONS block below (DATASET, RAW_FILES_BY_DATASET, CONFIG, …).
  3. Run from the module directory (uses the DATASET / MODE / SCOPE constants):
         python run_reduced_tc_suite.py
     ...override mode/scope on the command line for ad-hoc runs:
         python run_reduced_tc_suite.py --mode fixed   --scope local
         python run_reduced_tc_suite.py --mode optimal --scope regional
     ...batch over one or more REGISTERED dataset keys (no editing needed):
         python run_reduced_tc_suite.py --dataset chs-na chs-la
         python run_reduced_tc_suite.py --dataset chs-tx --mode optimal --scope regional
     ...or from the repository root:
         python modules/reduced_tc_suite/run_reduced_tc_suite.py

Unlike the POT/PST launchers (which take input-file PATHS), this one batches by
DATASET KEY — each key must exist in RAW_FILES_BY_DATASET below, because RTCS
needs the dataset's raw filenames + metadata + units, not a single file. If the
processed tc_data.h5 store is missing it is built automatically from
RAW_FILES_BY_DATASET (see Bootstrapping above). ``--help`` lists all options.

Inputs
------
  data/inputs/raw/<DATASET>/...           — raw `.mat` / `.csv` source files
  data/inputs/processed/<DATASET>/        — generated `tc_data.h5` store

Outputs
-------
  data/outputs/<DATASET>/<scope>/<mode>/  — CSVs, plots, QBM HDF5; keyed by
                                            dataset, scope, and mode so no
                                            two runs overwrite each other.
"""

import os
import sys
from pathlib import Path

# Guarantee headless rendering (no display needed) unless the operator overrides.
os.environ.setdefault("MPLBACKEND", "Agg")


# ── Module-root path anchoring (CyHAN v2.1 §A.5) ──────────────────────────
# All paths below are constructed relative to ROOT so the launcher can be
# invoked from any working directory (e.g. an IDE run-config, a Jenkins job,
# or a parent directory `python modules/.../run_*.py`).
ROOT = Path(__file__).resolve().parent       # run_<name>.py lives at module root
_BACKEND_PY = ROOT / "backend" / "python"
_COMMON_PY  = ROOT.parents[1] / "common" / "python"   # shared CyHAN common library (§5.2)
for _p in (_BACKEND_PY, _COMMON_PY):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_cpp_extension() -> None:
    """Build the _rtcs C++ kernel once if it isn't already compiled.

    Must run before the package is imported (kmedoids probes for _rtcs at import
    time). A failed build is non-fatal — the pure-Python fallback runs.
    """
    pkg = _BACKEND_PY / "reduced_tc_suite"
    if any(p.suffix in (".pyd", ".so", ".dylib") for p in pkg.glob("_rtcs*")):
        return
    build = ROOT / "backend" / "engines" / "cpp" / "build.py"
    if not build.is_file():
        return
    print("[run] C++ kernel _rtcs not built — compiling once "
          "(falls back to pure Python if this fails) ...")
    import subprocess
    try:
        subprocess.run([sys.executable, str(build)], check=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[run] _rtcs build failed: {exc}. Using pure-Python fallback.")


# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================

# ── Dataset folder ───────────────────────────────────────────────────────
# Case-sensitive folder name used to route all per-dataset paths:
#   data/inputs/raw/<DATASET>/         — raw .mat / .csv source files
#   data/inputs/processed/<DATASET>/   — preprocessed tc_data.h5 store
#   data/outputs/<DATASET>/            — selection results, plots, QBM HDF5
# To switch study areas, change ONLY this string to a key that exists in
# RAW_FILES_BY_DATASET below — the matching raw filenames are selected
# automatically. (PREPROCESS_METADATA may still need adjusting if the new
# dataset uses different MATLAB conventions.)
DATASET = "chs-na"

# ── Mode ─────────────────────────────────────────────────────────────────
# fixed   — single run at k = (# pre-selected) + k_additional; emits the full
#           diagnostic suite (SPLOM, PCA y-space, HC verification, QBM,
#           optional sub-RTCS).
# optimal — sweep k from k_min..k_max step k_step and pick the smallest k
#           with global HC RMSE ≤ rmse_threshold (else argmin RMSE).
# Overridable from the command line: --mode {fixed,optimal}.
MODE = "fixed"

# ── Scope ────────────────────────────────────────────────────────────────
# local    — apply BBOX: drop nodes outside the bbox, drop storms whose
#            tracks never come within max_track_dist_km of the bbox medoid.
#            Loads node_coord_source + the per-storm TROP track files,
#            renders a bbox diagnostic map.
# regional — basin-wide; BBOX is ignored entirely (no node-coord file or
#            track files are loaded). Use this for whole-domain selection
#            or when the dataset has no bbox calibration yet.
# Overridable from the command line: --scope {local,regional}.
SCOPE = "regional"

# ── Raw filenames (per dataset) ──────────────────────────────────────────
# One block per study area, keyed by the DATASET name above. Switching
# regions means changing ONLY the DATASET line — the launcher looks up
# RAW_FILES_BY_DATASET[DATASET], so the filenames can never drift out of
# sync with the dataset. To add a region: append its block here, then point
# DATASET at the new key.
#
# These names are read only when the processed tc_data.h5 store is missing
# and the preprocessor must rebuild it from raw .mat files in
# data/inputs/raw/<DATASET>/. Keys (do NOT rename):
#   x_param_table       — TC parameter table (MATLAB matrix; one row per storm)
#   y_surge             — ADCIRC peak surge fields (storms × nodes)
#   nodeID              — node ID + lat/lon table; used by BOTH the
#                         preprocessor (to subset Y columns) AND the bbox
#                         filter (to map node IDs back to coordinates)
#   hc_benchmark        — benchmark hazard curves (nodes × AER levels)
#   pre_selected_storms — OPTIONAL CSV listing storms that must always be
#                         included in the RTCS; set to None to skip.
# A missing file or typo raises a clear FileNotFoundError at startup.
RAW_FILES_BY_DATASET = {
    "chs-na": {
        "x_param_table":       "CHS-NA_ITCS_Param_MasterTable.mat",
        "y_surge":             "CHS-NA_SSL_TC_slc0_tide0_wave1_m_MSL.mat",
        "nodeID":              "CHS-NA_nodeID_v4.mat",
        "hc_benchmark":        "CHS-NA_HC_tbl_ATCS_SSL_slc0_BE_22.mat",
        "pre_selected_storms": None,                          # None = skip
    },
    "chs-la": {
        "x_param_table":       "CHS-LA_ITCS_Param_MasterTable.mat",
        "y_surge":             "CHS-LA_SSL_TC_slc0_tide0_wave1_m_NAVD88.mat",
        "nodeID":              "CHS-LA_nodeID.mat",
        "hc_benchmark":        "CHS-LA-24_HC_tbl_TC_SSL_slc0_BE_22.mat",
        "pre_selected_storms": None,            # or "pre_selected_storms_100.csv"
    },
}

# ── Vertical datum / units, keyed by DATASET ─────────────────────────────
# Unit string written to the /Y and /HC attributes of tc_data.h5 (one value
# per dataset — surge and HC share the same datum). CHS-NA is referenced to
# MSL; CHS-LA to NAVD88. Add an entry alongside the RAW_FILES_BY_DATASET
# block above whenever you onboard a dataset.
UNITS_BY_DATASET = {
    "chs-na": "m MSL",
    "chs-la": "m NAVD88",
}


# ── Preprocess metadata ─────────────────────────────────────────────────
# Per-dataset details that govern how each raw .mat file is decoded.
# Only consulted when the preprocessor runs (i.e. when tc_data.h5 is built
# or rebuilt). Adjust when a new dataset uses different MATLAB variable
# names or transpose orientation. The vertical datum (Y_units / HC_units)
# is NOT set here — it comes from UNITS_BY_DATASET above so it tracks DATASET.
PREPROCESS_METADATA = {
    # X — TC parameter table (storms × params, float64)
    "X_variable":             "Param_MT",   # MATLAB variable name inside the .mat
    "X_param_names":          None,         # list[str] override, or None → ["X0","X1",...]
    "X_storm_id_col":         None,         # column index holding storm IDs, or None
    "X_columns":              None,         # column subset to keep, or None = all
    "X_transpose":            False,        # set True if .mat ships X as (params × storms)

    # Y — ADCIRC peak surge fields (storms × nodes, stored as float32)
    "Y_variable":             "Resp",       # MATLAB variable name
    "Y_node_ids":             None,         # explicit node-ID list, or None → use node filter
    "Y_units":                None,         # injected per-dataset by the orchestrator from UNITS_BY_DATASET
    "Y_transpose":            True,         # CHS-* .mats ship Y as (nodes × storms) → transpose

    # Node filter — subsets Y columns to the kept ADCIRC mesh nodes
    "Y_node_filter_variable": "nodeID",     # MATLAB variable name in the nodeID .mat
    "Y_node_filter_col":      1,            # column index of the ADCIRC node ID (1-based IDs)
    "Y_node_id_col":          0,            # column index of the sequential / main node ID

    # HC — benchmark hazard curves (nodes × AER levels, float64)
    "HC_variable":            "BE_22",      # MATLAB variable name
    "HC_units":               None,         # injected per-dataset by the orchestrator from UNITS_BY_DATASET
    "HC_transpose":           False,        # set True if .mat ships HC as (AER × nodes)
    "HC_aer_levels":          None,         # explicit AER vector override, or None → defaults

    "validate":               True,         # run validate_store after writing tc_data.h5
}

# ── Per-storm TROP filename pattern, keyed by DATASET ───────────────────
# One file per storm, 1-indexed; the pattern's {:04d} is replaced with the
# storm number. Add an entry whenever you onboard a dataset whose TROP files
# follow a different naming convention. Datasets absent from this map fall
# back to a built-in default (LACPR2 naming). Used by BOTH the preprocessor
# (to derive the store's /storm_ids) and the local-scope bbox track filter.
#
# Conventions observed under data/inputs/raw/itcs_tropfiles/<DATASET>/:
#   chs-na   NACCS_JPM####_TROP.txt   (NACCS suite; 1050 TCs, IDs 1..1050)
#   chs-la   LACPR2_JPM####_TROP.txt  (645 TCs, IDs 1..645)
#   chs-pr   PR_JPM####_TROP.txt      (300 TCs, IDs 1..300)
#   chs-tx   TC_JPM####.TROP          (660 TCs, IDs 1..660)
#   chs-gom  SACCS_JPM####_TROP.txt   (subset of SACS 1700-TC suite; 1085 TCs)
#   chs-sa   SACCS_JPM####_TROP.txt   (subset of SACS 1700-TC suite; 1060 TCs)
#
# IMPORTANT — file number == SACS/NACCS *master-suite storm ID*, NOT a Y-row
# position. By dataset convention, Y row i is the i-th storm in ascending ID
# order. For contiguous suites (na/la/pr/tx) that is just 1..N. The SACS
# subsets (gom, sa) draw NON-contiguous master IDs from the 1700-TC suite
# (e.g. sa starts at 0065), so a Y row maps to its file by master ID.
# These per-row IDs are captured ONCE at preprocess time (see
# storm_id_track_dir in _build_preprocess_config) and written to the store's
# /storm_ids, so every workflow — and the selection outputs — label storms by
# their true master ID. apply_bbox_filter prefers the store's storm_ids and
# only re-derives from filenames as a fallback for legacy ID-less stores.
# The lone requirement for gom/sa is that the track folder hold exactly one
# TROP file per Y-row storm (file count == Y rows). Regional never loads tracks.
TRACK_FILE_PATTERNS = {
    "chs-gom": "SACCS_JPM{:04d}_TROP.txt",
    "chs-la":  "LACPR2_JPM{:04d}_TROP.txt",
    "chs-na":  "NACCS_JPM{:04d}_TROP.txt",
    "chs-pr":  "PR_JPM{:04d}_TROP.txt",
    "chs-sa":  "SACCS_JPM{:04d}_TROP.txt",
    "chs-tx":  "TC_JPM{:04d}.TROP",
}

# ── Bounding box  (only used when SCOPE == "local") ─────────────────────
# Geographic window for a "local" run:
#   1. Keep only mesh nodes whose (lat, lon) falls inside the bbox.
#   2. Keep only storms whose ITCS track passes within max_track_dist_km of
#      the geographic medoid of the kept nodes.
# Only the geographic window and node-coord decode options live here; the
# launcher (main_reduced_tc_suite._build_bbox_config) completes the block
# with resolved paths — node_coord_source, track_dir, track_file_pattern.
#
# When switching to a non-CHS-LA dataset in local mode you MUST update the
# bbox bounds (currently calibrated for Coastal Louisiana) and add the new
# dataset's TROP naming convention to TRACK_FILE_PATTERNS above. Ignored by
# regional runs, so a dataset with no nodeID / track files still runs regionally.
BBOX = {
    # Geographic window (decimal degrees, WGS84).
    "bbox": {
        "lat_min": 28.5,
        "lat_max": 31.0,
        "lon_min": -92.0,
        "lon_max": -88.5,
    },

    # Decode of the node-ID → (lat, lon) table (RAW_FILES["nodeID"]).
    "node_coord_variable": "nodeID",   # MATLAB variable name in that file
    "node_id_col": 0,                  # column with sequential / main ID
    "lat_col":     2,                  # column with node latitude  (deg)
    "lon_col":     3,                  # column with node longitude (deg)

    # A storm is kept if any track point lies within this distance of the
    # bbox-node medoid (great-circle, km).
    "max_track_dist_km": 200,
}

# ── Main configuration ──────────────────────────────────────────────────
# Backend selection-engine tuning knobs. Keys consumed by
# reduced_tc_suite.workflows.{rtcs_selection, growth_evaluation}.
# Defaults for every key live in config/defaults.py — entries below override
# only what's relevant for this study area. Per-dataset paths (h5_path,
# output_dir, pre_selected_csv) are NOT set here — the launcher derives them
# from DATASET and injects them, so this block stays purely tuning options.
CONFIG = {
    # ── fixed-mode subset sizing ──────────────────────────────────────
    # Final k = len(pre_selected) + k_additional. Pre-selected storms are
    # always retained; the k_additional extras are chosen by k-medoids on
    # the joint X/Y matrix.
    "k_additional": 500, # 200,

    # ── Sub-RTCS (optional second-stage subset) ───────────────────────
    # If k_sub_rtcs > 0, a smaller subset is selected from the initial RTCS.
    # Modes (see defaults.py for the full list):
    #   within          — PAM k-medoids inside the initial RTCS
    #   within_maximin  — greedy farthest-point inside the initial RTCS
    #   additional      — re-run main selection at k = forced + k_sub_rtcs
    "k_sub_rtcs":     0, # 50,
    "sub_rtcs_mode": "within_maximin",

    # ── Node subsampling ──────────────────────────────────────────────
    # Y has m ~= 2.4M nodes for CHS-LA. Stride decimates by taking every
    # Nth node (e.g., 100 → ~24,500 nodes). Lower stride = finer resolution,
    # higher memory and runtime. Set to None (or 1) to use every node.
    "node_stride": 100,

    # ── Reading the PCA y-space plots (pca_yspace_*.png) ──────────────
    # Each storm is projected onto the first two principal components of the
    # surge matrix Y. How to interpret the axes and the recurring fan shape:
    #   PC1 (x-axis) — TOTAL INUNDATION MAGNITUDE: how much surge a storm
    #       produces across the whole domain. It is the dominant axis because
    #       most of Y's variance is "how wet overall," not spatial pattern.
    #       Weak storms sit at one extreme (here the most-negative end),
    #       the strongest at the other.
    #   PC2 (y-axis) — the leading SPATIAL CONTRAST among storms of similar
    #       magnitude (e.g., surge focused in one sub-region vs. another).
    #   Apex (the dense vertex, ~PC1 most-negative / PC2 near the cluster
    #       center) — the pile of LOW-INUNDATION storms. They wet very few
    #       nodes, so after dry-fill their feature vectors are nearly identical
    #       and collapse onto one point; the fan opens toward stronger, more-
    #       inundating storms. This wedge is intrinsic to zero-inflated basin-
    #       wide data and barely moves between dry strategies (they only change
    #       node selection / fill). The wedge concentrates most storms in the
    #       apex, so a density-following selector (kmedoids) piles its picks
    #       there too. To spread the subset across PC1-PC2 instead, set
    #       selection_method = "maximin" below (space-filling selection).
    #
    # ── PCA dry-node handling ─────────────────────────────────────────
    # How NaN (dry-node) markers in Y are resolved before PCA. For basin-wide
    # regional runs, most nodes are dry for most storms, so the default
    # "drop_always_dry" lets zero-padding dominate PC1 (the classic wedge /
    # magnitude-axis artifact). Switch modes here to reshape the Y-space:
    #   "drop_always_dry" — drop 100%-dry nodes, zero-fill the rest (default)
    #   "zero"            — replace every NaN with 0.0 (keeps node count)
    #   "node_mean"       — impute NaN with each node's mean wet value
    #   "wet_only"        — keep only always-wet nodes (aggressive; may drop
    #                       nearly all nodes in a basin-wide run)
    #   "wet_ratio_floor" — keep nodes wet for ≥ pca_min_wet_fraction of
    #                       storms, then zero-fill the rest (middle ground)
    "pca_dry_strategy":     "drop_always_dry",
    # Wet-fraction floor for "wet_ratio_floor" (0–1); ignored by other modes.
    # e.g. 0.2 keeps nodes wet for at least 20% of storms.
    "pca_min_wet_fraction": 0.05,

    # ── Subset selection method ───────────────────────────────────────
    # How the k storms are chosen from the joint X / Y-space:
    #   "kmedoids" — PAM (default). Picks a REPRESENTATIVE subset, but is
    #                density-following: with the zero-inflated wedge it piles
    #                picks into the weak-storm apex and under-samples the tail.
    #   "maximin"  — farthest-point. SPACE-FILLING: spreads picks uniformly
    #                across PC1-PC2 and reaches the intense-storm tail. Use
    #                this when the kmedoids picks cluster in the apex.
    "selection_method": "kmedoids",

    # ── TC-parameter SPLOM (scatter-plot matrix) ──────────────────────
    # Visualises which X parameters span well in the selected RTCS.
    # splom_params holds 0-based column indices into X; splom_labels is the
    # display label for each. The two must be the same length.
    "splom_params": [6, 7, 8, 9],
    "splom_labels": ["Dp (hPa)", "Rm (km)", "Vt (km/h)", "Hd (deg)"],
    "splom_title":  "TC Parameters",

    # ── DSW aggregation method ────────────────────────────────────────
    # How per-node DSWs are averaged into the global DSW set:
    #   1 = simple mean (classic JPM-OS, equal node weight)
    #   2 = surge-weighted (per-storm-per-node)
    #   3 = variance-weighted (fixed per-node weight = surge variance)
    "dsw_method": 3,

    # ── α/β weighting ─────────────────────────────────────────────────
    # The joint feature matrix is  Z = [ α·X̃ | β·Ỹ_r ]  (X̃, Ỹ_r each
    # standardised to unit variance first). α weights the TC-parameter
    # space, β the surge-response space. The α/β search ON/OFF switch
    # (AB_SWEEP) is at the END of the user options below; when the search
    # is OFF these fixed weights are used as-is.
    #
    # IMPORTANT — α = β does NOT make X and Y count equally. Selection uses
    # Euclidean distance, which sums over ALL columns, and each standardised
    # column carries ~1 unit of variance. So each block's pull on the
    # distance scales with how many columns it has:
    #     X block ≈ α² × 4                  (4 = number of x_select_columns)
    #     Y block ≈ β² × (# retained PCs)   (~30 at 95% variance on this data)
    # With α = β = 1 that ratio is ≈ 4 : 30, so the surge-response space
    # outweighs the TC parameters ~7-8×. To make X and Y count EQUALLY
    # overall, shrink β to offset the column count: β ≈ α × (4 / #PCs).
    # (Letting AB_SWEEP find the balance empirically avoids guessing.)
    #
    # WHICH β TO USE depends on the deliverable — the two goals disagree:
    #   - Accurate hazard curves: the AB_SWEEP optimum minimises HC
    #     reconstruction RMSE and tends to pick a SMALL β (it favours
    #     covering TC-parameter space). That suite may score BETTER on HC
    #     RMSE even though its storms look clustered in the PCA y-space plot.
    #   - Representative / space-filling suite: a larger β (e.g. 1.0) gives
    #     the surge-response space real weight, spreading the selection
    #     evenly across the PCA y-space. Best judged by the y-space plot, not
    #     by HC RMSE.
    # The sweep optimises ONLY the first goal (it has no coverage term), so a
    # low-β winner can still cluster the picks. Check alpha_beta_sweep.csv to
    # see how much HC RMSE you trade for the spread a larger β buys.
    "alpha_default":   10.0,     # α  (TC-parameter weight)
    "beta_default":    1.0,      # β  (surge-response weight)

    # ── α/β sweep performance knobs ───────────────────────────────────
    # When the sweep is ON, it runs k-medoids + DSW evaluation once per
    # grid point (16 by default — see alpha_beta_grid in defaults.py).
    #
    # ab_search_workers:
    #   None — auto (≈ cpu_count, capped at grid size)
    #   1    — sequential (no process pool overhead; good for debugging)
    #   N    — exactly N worker processes
    # ab_search_node_sample:
    #   None — score using every node in Y (slowest, exact)
    #   int  — score using a fixed random subset of that many nodes
    #          (e.g. 3000); ~5–10× faster with negligible accuracy loss
    #          since the optimum (α, β) is robust to node sampling.
    "ab_search_workers":     None,
    "ab_search_node_sample": 3000,

    # ── optimal-mode parameters (only used when mode == "optimal") ────
    # Sweep k from k_min to k_max in steps of k_step; pick the smallest k
    # whose global HC RMSE ≤ rmse_threshold (units: m). If no k meets the
    # tolerance, argmin RMSE is selected and a warning is emitted.
    # bias_report_aer = AER hazard levels to log nodal bias at, labelled by MRI
    #                   year N (so bias_aer1000 is the AER = 1/1000 level).
    "k_min":             20,
    "k_max":            300,
    "k_step":             5,
    "rmse_threshold":  0.10,
    "bias_report_aer": [10, 100, 1000],
}

# ── α/β search ON/OFF ──────────────────────────────────────────────────────
# True  → search AB_GRID for the optimal (α, β) weights (needs HC_bench);
#         this overrides the fixed alpha_default / beta_default in CONFIG.
# False → skip the search; use the fixed alpha_default / beta_default weights.
AB_SWEEP = False

# Grid searched when AB_SWEEP is True (edit the points as you like).
AB_GRID = [
    (0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0),
    (5.0, 1.0), (10.0, 1.0), (20.0, 1.0),
    (0.1, 0.1), (1.0, 0.1), (5.0, 0.1), (10.0, 0.1), (20.0, 0.1),
    (0.1, 0.5), (1.0, 0.5), (5.0, 0.5), (10.0, 0.5),
]

CONFIG["alpha_beta_grid"] = AB_GRID if AB_SWEEP else None

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================
#
# All procedural logic (CLI parsing, dataset resolution, batch iteration, path
# wiring, store bootstrap, bbox assembly, dispatch) lives in
# main_reduced_tc_suite.launch_batch. This file only hands it the operator
# option block above.

if __name__ == "__main__":
    _ensure_cpp_extension()   # build _rtcs on first run if needed

    # The orchestrator entry (main_reduced_tc_suite) lives in backend/python,
    # added to sys.path above at runtime. Resolve it dynamically so there is no
    # static import for the IDE to flag as unresolved.
    from importlib import import_module
    launch_batch = import_module("main_reduced_tc_suite").launch_batch

    raise SystemExit(launch_batch(
        root                 = ROOT,
        default_dataset      = DATASET,
        default_mode         = MODE,
        default_scope        = SCOPE,
        raw_files_by_dataset = RAW_FILES_BY_DATASET,
        units_by_dataset     = UNITS_BY_DATASET,
        preprocess_metadata  = PREPROCESS_METADATA,
        track_file_patterns  = TRACK_FILE_PATTERNS,
        bbox                 = BBOX,
        config               = dict(CONFIG),
    ))
