"""run_augmented_hurricane_database - AHD launcher (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Augmented Hurricane Database (AHD) module. The
operator edits the USER OPTIONS block below and runs the script. No
orchestration logic lives here - the launcher hands the option block to
``main_augmented_hurricane_database.run`` per §5.3.

================================================================================
WHAT AHD PRODUCES
================================================================================
AHD turns NHC's HURDAT2 best-track archive into a tidy, HURDAT-like CSV - one
file per basin - with two motion columns derived on top of the raw record:

  * trans_kmh   - translation (forward) speed of the storm centre, km/h
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
  1. Resolve the source per basin. With DOWNLOAD = True the newest dated file is
     discovered from the NHC directory (https://www.nhc.noaa.gov/data/hurdat/)
     and fetched into data/inputs/ - ranked by record end-year then date-stamp.
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
# "atlantic", "pacific" (NE/NC Pacific), or "both".
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
OVERWRITE = False

# Pin explicit input files to skip discovery (absolute, or relative to
# data/inputs/). Leave as None for automatic resolution.
ATLANTIC_FILE = None     # e.g. "hurdat2-1851-2025-02272026.txt"
PACIFIC_FILE  = None     # e.g. "hurdat2-nepac-1949-2025-02272026.txt"

# ── EBTRK Rmax backfill ──────────────────────────────────────────────────────
# HURDAT2 records the radius of maximum wind (rmax_km) only for recent seasons.
# The Extended Best Track (EBTRK) dataset carries Rmax back to the late 1980s.
# With APPEND_EBTRK_RMAX = True the missing rmax_km values are filled from EBTRK,
# matched per storm on (nhc_id, synoptic time); HURDAT-provided values are kept.
# The Atlantic uses the AL file; the Pacific (HURDAT nepac, which contains both
# East Pacific and Central Pacific storms) uses both the EP and CP files. When
# DOWNLOAD = True the file(s) are fetched from CIRA; otherwise local copies are
# used (data/inputs/, or the legacy ebtrk/ folder).
APPEND_EBTRK_RMAX = True
# EBTRK_FILE points the backfill at specific local EBTRK file(s) instead of
# resolving them automatically. Leave it as None for normal use, where the module
# selects the correct file(s) per basin (AL for the Atlantic; EP and CP for the
# Pacific). Set it to a path, or a list of paths, only to force particular local
# copies.
EBTRK_FILE        = None

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
# Validation (per-point hold-out, against the MATLAB's own LOOCV):
#   Cp6 0.923 R2 / 5.27 hPa  (MATLAB 0.932 / 4.97); Cp3 0.923 (MATLAB 0.918)
#   Rm7 0.832 R2 / 29.1 km   (MATLAB 0.603 / 36.0); Rm4 0.768 (MATLAB 0.401)
IMPUTE_GPM      = False
# Quality/speed upgrades over the original MATLAB (all on by default). Turn the
# three quality flags OFF to closely reproduce the original MATLAB GP metamodel
# (constant-mean kriging, raw response, support-set prediction).
GPM_VECCHIA       = True    # NNGP - predict from all training data, not just the support set
GPM_PHYSICAL_MEAN = True    # wind-pressure (Cp) / lat·deficit (Rmax) kriging trend, not constant
GPM_LOG_RMAX      = True    # fit the Rmax models in log space (Rmax is ~lognormal)
GPM_PARALLEL      = True    # train the two models per target concurrently (speed only)
# Per-target method settings (empirically tuned; see README §validation). Cp is
# smooth/long-range → wants more calibration support; Rmax is short-range/noisy →
# wants a small conditioning set. The physical-mean trend leaves a short-range
# residual, so a large NNGP neighbor set does not help either target.
GPM_CP_MAX_SUPPORT   = 6000
GPM_CP_NEIGHBORS     = 30
GPM_RMAX_MAX_SUPPORT = 3000
GPM_RMAX_NEIGHBORS   = 10

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = DATA / "outputs"
WRITE_PARQUET = False    # also write a Parquet copy beside each CSV
# Output filename stem; {basin} and {end_year} are substituted.
OUTPUT_STEM   = "hurdat2_{basin}_{end_year}"

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================


# ── Launcher plumbing (CyHAN v2.0 §A.5 path anchoring; no user options) ─────
import sys

_BACKEND_PY = ROOT / "backend" / "python"
if str(_BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PY))


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
    "input_dir":         DATA / "inputs",
    "output_dir":        OUTPUT_DIR,
    "write_parquet":     WRITE_PARQUET,
    "output_stem":       OUTPUT_STEM,
    "append_ebtrk_rmax": APPEND_EBTRK_RMAX,
    "ebtrk_file":        EBTRK_FILE,
    "impute_gpm":           IMPUTE_GPM,
    "gpm_vecchia":          GPM_VECCHIA,
    "gpm_physical_mean":    GPM_PHYSICAL_MEAN,
    "gpm_log_rmax":         GPM_LOG_RMAX,
    "gpm_parallel":         GPM_PARALLEL,
    "gpm_cp_max_support":   GPM_CP_MAX_SUPPORT,
    "gpm_cp_neighbors":     GPM_CP_NEIGHBORS,
    "gpm_rmax_max_support": GPM_RMAX_MAX_SUPPORT,
    "gpm_rmax_neighbors":   GPM_RMAX_NEIGHBORS,
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
    result = import_module("main_augmented_hurricane_database").run(cfg)
    print("\n[ahd] done:")
    for basin, r in result.results.items():
        extra = ""
        if r.n_rmax_filled:
            extra += f"  (+{r.n_rmax_filled:,} EBTRK Rmax)"
        if r.n_pmin_gpm or r.n_rmax_gpm:
            extra += f"  (GPM +{r.n_pmin_gpm:,} pmin, +{r.n_rmax_gpm:,} Rmax)"
        print(f"      {basin:9s} {r.n_storms:>6,} storms  {r.n_rows:>8,} rows{extra}  -> {r.csv_path}")
