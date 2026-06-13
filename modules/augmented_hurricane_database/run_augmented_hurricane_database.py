"""run_augmented_hurricane_database - AHD launcher (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Augmented Hurricane Database (AHD) module. The
operator edits the USER OPTIONS block below and runs the script. No
orchestration logic lives here - the launcher hands the option block to
``api_augmented_hurricane_database.run`` per §5.3.

================================================================================
WHAT AHD PRODUCES
================================================================================
AHD turns NHC's HURDAT2 best-track archive into a tidy, HURDAT-like CSV - one
file per basin - with two motion columns derived on top of the raw record:

  * trans_kmh   - translation (forward) speed of the storm center, km/h
  * heading_deg - heading (forward azimuth), degrees, in (-180, 180]

These are computed from consecutive 6-hourly fixes by WGS-84 geodesic
distance/azimuth (the same convention as the legacy MATLAB house format,
CHS_TC_HURDAT_Atlantic.m). Unit conventions: maximum sustained wind is converted
knots -> km/h, wind-radii and Rmax nautical miles -> km, minimum central pressure
stays in hPa, and HURDAT2 sentinels (-99 / -999) become NaN. A stationary fix
(rounded speed 0 km/h) gets NaN speed and heading.

Output columns (per row = one storm time-snapshot):
  tc_no, snap_no, year, nhc_id, basin, name, ymd, hhmm, time_utc, lat, lon,
  landflag, status, vmax_kmh, pmin_hpa, trans_kmh, heading_deg,
  radii{34,50,64}_{ne,se,sw,nw}_km, rmax_km

================================================================================
METHOD
================================================================================
  1. Resolve the source per basin.
     With DOWNLOAD = True the newest dated file is discovered from the NHC directory
     (https://www.nhc.noaa.gov/data/hurdat/) and fetched into data/inputs/ - ranked
     by record end-year then date-stamp.
     With DOWNLOAD = False the newest matching hurdat2-*.txt already on disk is
     used (data/inputs/, or beside the module). ATLANTIC_FILE / PACIFIC_FILE pin
     an explicit file and skip discovery.
  2. Parse each storm header + its track rows into time snapshots; map the
     record-identifier and status codes; convert units; flag sentinels as NaN.
  3. Derive trans_kmh and heading_deg per snapshot from consecutive geodesic
     fixes (first fix forward-filled from the second).
  4. (Optional, APPEND_EBTRK_RMAX) Backfill missing rmax_km from the Extended
     Best Track (EBTRK) dataset, matched per storm on (nhc_id, synoptic time).
     Atlantic uses the AL file, Pacific the EP and CP files; HURDAT-provided Rmax
     is left untouched.
  5. (Optional, IMPUTE_GPM) Fill values still missing - pmin_hpa then rmax_km -
     with Gaussian-process metamodels (Taflanidis et al.) self-trained on the
     observed rows. Observed values are kept.
  6. Write one HURDAT-like CSV per basin to data/outputs/.

Run (headless / CLI)
--------------------
  1. Install dependencies once:
         pip install -r requirements.txt
  2. Edit the USER OPTIONS block below (BASIN, DOWNLOAD, ...).
  3. Run from the module directory:
         python run_augmented_hurricane_database.py
     ...or from the repository root:
         python modules/augmented_hurricane_database/run_augmented_hurricane_database.py

  CLI overrides (no file edits needed):
         python run_augmented_hurricane_database.py --basin atlantic
         python run_augmented_hurricane_database.py --basin both --no-download
     ``--help`` lists options.
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Basin selection ─────────────────────────────────────────────────────────
# Which ocean basin(s) to build. One of:
#   "atlantic" - North Atlantic (HURDAT2 Atlantic file)
#   "pacific"  - Northeast and North-Central Pacific (HURDAT2 "nepac" file,
#                which contains both East Pacific and Central Pacific storms)
#   "both"     - build the Atlantic first, then the Pacific (two output files)
BASIN = "both"

# ── Source policy ────────────────────────────────────────────────────────────
# DOWNLOAD = True  → fetch the newest dated file per basin from NHC into
#                    data/inputs/ (ranked by record end-year, then date-stamp).
# DOWNLOAD = False → use the newest hurdat2-*.txt already on disk (data/inputs/,
#                    or beside the module). No network access.
DOWNLOAD  = True
# OVERWRITE controls what happens when the dated file is ALREADY on disk.
#   False (default): reuse the local copy, skip the network. NHC filenames carry a
#                    content date-stamp, so a same-named file is byte-identical;
#                    re-downloading it gains nothing.
#   True           : fetch a fresh copy and overwrite the local file. Use this only
#                    to repair a corrupted or partial download. (When a NEWER file
#                    is published, it has a DIFFERENT name and is fetched anyway,
#                    regardless of this flag.)
OVERWRITE = True

# Pin an explicit local HURDAT2 file per basin. This bypasses BOTH discovery and
# download - the named file is used as-is, with no network access even when
# DOWNLOAD = True (and it overrides ATLANTIC_URL / PACIFIC_URL below). Give one
# file per basin: ATLANTIC_FILE for the Atlantic, PACIFIC_FILE for the Pacific
# ("nepac"). The path is absolute, or relative to data/inputs/. Leave as None to
# resolve automatically (discover-and-download the newest when DOWNLOAD = True,
# otherwise the newest matching file already on disk).
ATLANTIC_FILE = None     # e.g. "hurdat2-1851-2025-02272026.txt"
PACIFIC_FILE  = None     # e.g. "hurdat2-nepac-1949-2025-02272026.txt"

# Per-basin HURDAT2 download URL override. Leave as None to auto-discover the
# newest dated file from the NHC directory. Set a full file URL to fetch that
# exact file instead (e.g. a mirror, a proxy, or a file NHC moved). Honored only
# when DOWNLOAD = True; an ATLANTIC_FILE / PACIFIC_FILE local pin still wins.
ATLANTIC_URL = None      # e.g. "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
PACIFIC_URL  = None      # e.g. "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-02272026.txt"

# ── EBTRK Rmax backfill ──────────────────────────────────────────────────────
# HURDAT2 records the radius of maximum wind (rmax_km) only for recent seasons.
# The Extended Best Track (EBTRK) dataset carries Rmax back to the late 1980s.
# With APPEND_EBTRK_RMAX = True the missing rmax_km values are filled from EBTRK,
# matched per storm on (nhc_id, synoptic time); HURDAT-provided values are kept.
# The Atlantic uses the AL file; the Pacific (HURDAT nepac, which contains both
# East Pacific and Central Pacific storms) uses both the EP and CP files. When
# DOWNLOAD = True the newest file is discovered from the CIRA listing and fetched
# (ranked by record end-year, then publish stamp); otherwise the newest matching
# local copy is used (data/inputs/, or the legacy ebtrk/ folder).
APPEND_EBTRK_RMAX = True
# Pin explicit local EBTRK file(s). EBTRK ships as THREE separate files, one per
# cyclone-id basin: AL (Atlantic), EP (East Pacific), CP (Central Pacific). An
# Atlantic build needs AL; a Pacific build (HURDAT "nepac" = EP + CP storms)
# needs both EP and CP. Setting EBTRK_FILE bypasses BOTH discovery and download
# for these files - they are used as-is, with no network access even when
# DOWNLOAD = True (and it overrides the EBTRK_*_URL options below).
#
# Give a single path or a LIST of paths (absolute, or relative to data/inputs/):
#     EBTRK_FILE = "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt"
#     EBTRK_FILE = ["EBTRK_EP_..._02-Sep-2022.txt",          # Pacific (EP + CP)
#                   "EBTRK_CP_..._02-Sep-2022-1.txt"]
# Whatever you list is pooled and matched to HURDAT storms by cyclone id, so the
# order does not matter and listing files a basin does not need is harmless (they
# simply match nothing). The SAME list is used for every basin in the run, so for
# BASIN = "both" just list all the files you want available (e.g. all three:
# AL + EP + CP) and each basin will pick up its matches.
#
# Leave as None for normal use: the module discovers and downloads the correct,
# newest file(s) per basin automatically. Use EBTRK_FILE only to force specific
# local copies (an offline run, or a frozen vintage for reproducibility).
EBTRK_FILE        = None
# Per-file EBTRK download URL overrides, one per cyclone-id code (AL / EP / CP).
# Leave as None to auto-discover the newest file from the CIRA listing. Set a full
# file URL to fetch that exact file instead. Honored only when DOWNLOAD = True;
# an EBTRK_FILE local pin still wins over these.
EBTRK_AL_URL      = None   # Atlantic
EBTRK_EP_URL      = None   # East Pacific
EBTRK_CP_URL      = None   # Central Pacific

# ── GP-metamodel imputation (pmin & Rmax) ────────────────────────────────────
# Fill values still missing after the steps above using Gaussian-process
# metamodels, self-trained on the observed rows: central pressure first, then
# Rmax (which uses the completed pressure). Runs per basin.
#
# This is a Python re-implementation of the GP metamodel of Taflanidis et al.
# (universal kriging: polynomial/physical trend, anisotropic power-exponential
# kernel, nugget, hyperparameters by MLE) and the CHS HURDAT imputation drivers.
# See backend/python/augmented_hurricane_database/gp_metamodel/README.md for the
# full MATLAB-to-Python map, the improvements, and the validation.
#
# Improvements over the original MATLAB (all default-on; set the three quality
# flags to False to reproduce the MATLAB to r=0.98 on central pressure):
#   * Vecchia/NNGP: predict from all training fixes, not just the support set
#   * physical mean: wind-pressure trend (Cp), latitude/intensity trend (Rmax)
#   * log-Rmax: fit Rmax in log space (positive, roughly lognormal)
#   * speed: analytic-gradient MLE, OpenMP C++ kernel, parallel model fits
# Validation of the recommended (default) config vs the MATLAB, on the IDENTICAL
# HURDAT file and EBTRK-augmented Rmax set, both scored by leave-one-out on the
# deployed predictor (the MATLAB's metric); full tables in the README and white
# paper. The recommended config BEATS the MATLAB on all four models:
#   Cp6 0.936 R2 (MATLAB 0.932)   Cp3 0.920 (MATLAB 0.918)
#   Rm7 0.607    (MATLAB 0.603)   Rm4 0.447 (MATLAB 0.401)
# Two levers drive this: a deep calibration (N_CAL / N_LHS) and a large enough
# support set (Rmax needed 8000, not 3000). Rmax follows the MATLAB workflow: it is
# trained on observed pressure (before Cp imputation) and predicts missing Rmax
# with the completed pressure. Do not compare a Python hold-out number with a
# MATLAB LOOCV number (different protocols).
#
# IMPUTE_GPM is the master on/off switch for this whole step:
#   False (default): skip GP imputation entirely. Rows that are still missing
#                    pmin_hpa or rmax_km after the EBTRK backfill stay blank
#                    (NaN) in the output CSV. No models are trained.
#   True           : train the four GP models on the observed rows and fill the
#                    missing values (central pressure first, then Rmax). This is
#                    the most computationally expensive step; the model cache
#                    below makes repeat runs fast. All the GPM_* options that
#                    follow apply only when this is True.
# The GPM_* defaults below are the recommended configuration.
IMPUTE_GPM      = True
# Trained-model cache. Each fitted model is saved under MODEL_DIR as a compressed
# .npz keyed by basin, model, and a signature of the settings and training data.
# RETRAIN = False reuses a matching cached model and skips training (fast repeat
# runs); RETRAIN = True retrains regardless and overwrites. A settings or
# data-vintage change yields a new signature, so a stale model is never reused.
MODEL_DIR    = DATA / "models"   # folder where trained models are cached (one .npz per basin and model)
GPM_RETRAIN  = False             # False: reuse a matching cached model if one exists, else train and
                                 #        cache it (so the first run always trains)
                                 # True: always retrain and overwrite the cache
# Quality/speed upgrades over the original MATLAB (all on by default). To closely
# reproduce the original MATLAB GP metamodel (constant-mean kriging, raw response,
# support-set prediction), set GPM_PHYSICAL_MEAN and GPM_LOG_RMAX False and set the
# four *_VECCHIA flags False.
GPM_PHYSICAL_MEAN = True    # wind-pressure (Cp) / lat·deficit (Rmax) kriging trend, not constant
GPM_LOG_RMAX      = True    # fit the Rmax models in log space (Rmax is ~lognormal)
GPM_PARALLEL      = True    # train the two models per target concurrently (speed only)
# Solver per model: True = nearest-neighbor GP (NNGP, predicts from all data, the
# recommended default); False = exact full GP over the support (denser, slower,
# and needs the deeper calibration below). Set per model, e.g. GPM_CP6_VECCHIA =
# False to run central pressure (full-feature) as an exact full GP.
GPM_CP6_VECCHIA = True   # central pressure, full-feature model (Cp6)
GPM_CP3_VECCHIA = True   # central pressure, reduced model     (Cp3)
GPM_RM7_VECCHIA = True   # radius of max wind, full-feature model (Rm7)
GPM_RM4_VECCHIA = True   # radius of max wind, reduced model      (Rm4)
# Per-target calibration. MAX_SUPPORT and NEIGHBORS are the support-set and NNGP
# conditioning sizes; N_CAL is the hyperparameter-calibration subset size and
# N_LHS the Latin-hypercube budget for that search. Two levers drive accuracy: a
# deep calibration (N_CAL, N_LHS) and a support set scaled to each target's signal.
# Support is NOT set by raw fix count - it is the fraction of the data the target's
# structure needs. Central pressure (about 24k observed fixes) is smooth and
# high-skill, so a modest support of 6000 (about 25%) saturates accuracy: it reaches
# LOOCV 0.937 (above the MATLAB's 0.932) at N_CAL=4000 / N_LHS=250, with no gain
# beyond 6000. Radius of max wind has FEWER fixes (about 15k, the EBTRK-augmented
# set) but is noisier, shorter-range, and low-skill, so it needs a LARGER support
# FRACTION - 8000 (about 53%), not the 3000 first tried on a smaller dev subset -
# together with the deep calibration to reach 0.607 / 0.447 (above the MATLAB's
# 0.603 / 0.401). N_LHS>=250 is needed because the calibration is multi-modal. See
# the README validation section.
# Central Pressure
GPM_CP_MAX_SUPPORT   = 8000  # (optimal: 6000; better: 8000)
GPM_CP_NEIGHBORS     = 30    # (optimal: 30; best: 30)
GPM_CP_N_CAL         = 4000  # (optimal: 4000; best: 4000)
GPM_CP_N_LHS         = 250   # (optimal: 250; best: 250)
# Radius of Maximum Winds
GPM_RMAX_MAX_SUPPORT = 8000
GPM_RMAX_NEIGHBORS   = 30
GPM_RMAX_N_CAL       = 4000
GPM_RMAX_N_LHS       = 250

# ── Per-TC imputation plots (optional, off by default) ───────────────────────
# Write one PNG per tropical cyclone to visually inspect the imputed data along
# each storm's time history: the GP-metamodel-completed series as a line (GPM)
# with the originally observed values as red dots (Obs). One target per plot:
#   cp   - central-pressure deficit, 1013 - pmin (hPa)
#   rmax - radius of maximum wind (km)
# Needs IMPUTE_GPM = True (the plots show what the GP filled) and matplotlib.
# A full basin is ~1300-2000 storms, so this writes thousands of PNGs and takes a
# while; PLOT_JOBS parallelizes it.
#
# Master switch (flips all four groups at once):
#   PLOT_IMPUTATION = True  -> turn ALL four groups ON
#   PLOT_IMPUTATION = False -> turn ALL four groups OFF
#   PLOT_IMPUTATION = None  -> defer to the four per-(basin, target) flags below
PLOT_IMPUTATION    = True
PLOT_ATLANTIC_CP   = False
PLOT_ATLANTIC_RMAX = False
PLOT_PACIFIC_CP    = False
PLOT_PACIFIC_RMAX  = False
PLOT_DIR  = DATA / "outputs" / "plots"   # folder for the PNGs (per basin/target subfolders)
PLOT_JOBS = None                          # worker processes: None/0 = auto (cores-1), 1 = serial

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = DATA / "outputs"   # folder where the per-basin CSV files are written
WRITE_PARQUET = False              # True:  also write a Parquet copy beside each CSV
                                   # False: CSV only
# Output filename stem. These fields are filled in automatically from the source
# file (e.g., hurdat2-1851-2025-02272026.txt):
#   {basin}       - atlantic / pacific
#   {start_year}  - first record year     (1851 Atlantic, 1949 Pacific)
#   {end_year}    - last record year      (e.g. 2025)
#   {created}     - NHC file date, YYYYMMDD, from the filename stamp (e.g. 20260227)
# The default marks the file as augmented HURDAT2 and carries the source span and
# NHC vintage, e.g.:
#   "augmented_hurdat2_atlantic_1851-2025_20260227"
#       -> data/outputs/augmented_hurdat2_atlantic_1851-2025_20260227.csv
OUTPUT_STEM   = "augmented_hurdat2_{basin}_{start_year}-{end_year}_{created}"

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================


# ── Launcher setup (CyHAN v2.2 §A.5 path anchoring; no user options) ─────────
import sys

_BACKEND_PY = ROOT / "backend" / "python"
_COMMON_PY  = ROOT.parents[1] / "common" / "python"   # shared CyHAN common library (§5.2)
for _p in (_BACKEND_PY, _COMMON_PY):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_cpp_extension() -> None:
    """Build the _gpm C++ kernel once if absent (only when GPM imputation runs).

    A failed build is non-fatal - the GP metamodel transparently falls back to
    the pure-NumPy correlation kernel.
    """
    pkg = _BACKEND_PY / "augmented_hurricane_database" / "gp_metamodel"
    if any(pkg.glob("_gpm*.pyd")) or any(pkg.glob("_gpm*.so")) \
            or any(pkg.glob("_gpm*.dylib")):
        return
    build = ROOT / "backend" / "engines" / "cpp" / "build.py"
    if not build.is_file():
        return
    print("[run] C++ kernel _gpm not built - compiling once "
          "(falls back to pure NumPy if this fails) ...")
    import subprocess
    try:
        subprocess.run([sys.executable, str(build)], check=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[run] _gpm build failed: {exc}. Using pure-NumPy fallback.")


CONFIG = {
    "basins":            BASIN,
    "download":          DOWNLOAD,
    "overwrite":         OVERWRITE,
    "atlantic_file":     ATLANTIC_FILE,
    "pacific_file":      PACIFIC_FILE,
    "atlantic_url":      ATLANTIC_URL,
    "pacific_url":       PACIFIC_URL,
    "input_dir":         DATA / "inputs",
    "output_dir":        OUTPUT_DIR,
    "write_parquet":     WRITE_PARQUET,
    "output_stem":       OUTPUT_STEM,
    "append_ebtrk_rmax": APPEND_EBTRK_RMAX,
    "ebtrk_file":        EBTRK_FILE,
    "ebtrk_al_url":      EBTRK_AL_URL,
    "ebtrk_ep_url":      EBTRK_EP_URL,
    "ebtrk_cp_url":      EBTRK_CP_URL,
    "plot_imputation":   PLOT_IMPUTATION,
    "plot_atlantic_cp":   PLOT_ATLANTIC_CP,
    "plot_atlantic_rmax": PLOT_ATLANTIC_RMAX,
    "plot_pacific_cp":    PLOT_PACIFIC_CP,
    "plot_pacific_rmax":  PLOT_PACIFIC_RMAX,
    "plot_dir":          PLOT_DIR,
    "plot_jobs":         PLOT_JOBS,
    "impute_gpm":           IMPUTE_GPM,
    "gpm_model_dir":        MODEL_DIR,
    "gpm_retrain":          GPM_RETRAIN,
    "gpm_physical_mean":    GPM_PHYSICAL_MEAN,
    "gpm_log_rmax":         GPM_LOG_RMAX,
    "gpm_parallel":         GPM_PARALLEL,
    "gpm_cp6_vecchia":      GPM_CP6_VECCHIA,
    "gpm_cp3_vecchia":      GPM_CP3_VECCHIA,
    "gpm_rm7_vecchia":      GPM_RM7_VECCHIA,
    "gpm_rm4_vecchia":      GPM_RM4_VECCHIA,
    "gpm_cp_max_support":   GPM_CP_MAX_SUPPORT,
    "gpm_cp_neighbors":     GPM_CP_NEIGHBORS,
    "gpm_cp_n_cal":         GPM_CP_N_CAL,
    "gpm_cp_n_lhs":         GPM_CP_N_LHS,
    "gpm_rmax_max_support": GPM_RMAX_MAX_SUPPORT,
    "gpm_rmax_neighbors":   GPM_RMAX_NEIGHBORS,
    "gpm_rmax_n_cal":       GPM_RMAX_N_CAL,
    "gpm_rmax_n_lhs":       GPM_RMAX_N_LHS,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless runs (no file edits needed)."""
    import argparse
    p = argparse.ArgumentParser(
        description="Build HURDAT-like CSV(s) from NHC HURDAT2. With no "
                    "arguments it uses the USER OPTIONS above.")
    p.add_argument("--basin", choices=["atlantic", "pacific", "both"],
                   help="Override BASIN.")
    p.add_argument("--no-download", action="store_true",
                   help="Use a local file instead of fetching the latest.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if the dated file already exists.")
    p.add_argument("--ebtrk-rmax", dest="ebtrk", action="store_true", default=None,
                   help="Backfill missing Atlantic Rmax from EBTRK.")
    p.add_argument("--no-ebtrk-rmax", dest="ebtrk", action="store_false",
                   help="Disable the EBTRK Rmax backfill.")
    p.add_argument("--impute-gpm", dest="gpm", action="store_true", default=None,
                   help="Fill missing pmin/Rmax with the GP metamodels.")
    p.add_argument("--no-impute-gpm", dest="gpm", action="store_false",
                   help="Disable GP-metamodel imputation.")
    args = p.parse_args()
    config = dict(config)
    if args.basin:
        config["basins"] = args.basin
    if args.no_download:
        config["download"] = False
    if args.overwrite:
        config["overwrite"] = True
    if args.ebtrk is not None:
        config["append_ebtrk_rmax"] = args.ebtrk
    if args.gpm is not None:
        config["impute_gpm"] = args.gpm
    return config


if __name__ == "__main__":
    cfg = _apply_cli(CONFIG)
    if cfg.get("impute_gpm"):
        _ensure_cpp_extension()   # build _gpm on first GPM run if needed
    # The orchestrator entry lives in backend/python, added to sys.path above.
    # Resolve it dynamically so there is no static import for the IDE to flag.
    from importlib import import_module
    result = import_module("api_augmented_hurricane_database").run(cfg)
    print("\n[ahd] done:")
    for basin, r in result.results.items():
        extra = ""
        if r.n_rmax_filled:
            extra += f"  (+{r.n_rmax_filled:,} EBTRK Rmax)"
        if r.n_pmin_gpm or r.n_rmax_gpm:
            extra += f"  (GPM +{r.n_pmin_gpm:,} pmin, +{r.n_rmax_gpm:,} Rmax)"
        print(f"      {basin:9s} {r.n_storms:>6,} storms  {r.n_rows:>8,} rows{extra}  -> {r.csv_path}")
