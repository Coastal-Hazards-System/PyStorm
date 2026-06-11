"""run_tc_climatological_analysis - TCA launcher (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Tropical Cyclone Climatological Analysis (TCA) module.
The operator edits the USER OPTIONS block below and runs the script. No
orchestration logic lives here - the launcher hands the option block to
``main_tc_climatological_analysis.run`` per §5.3.

================================================================================
WHAT TCA PRODUCES
================================================================================
For each CHS Coastal Reference Location (CRL), TCA selects the tropical cyclones
from the augmented HURDAT2 best-track (the augmented_hurricane_database output)
that pass within MAX_DIST, picking each storm's representative point as the fix
that maximizes the Gaussian distance weight times the central-pressure deficit.
It then computes, with the Gaussian Kernel Function (GKF):

  * SRR  - omnidirectional storm recurrence rate   (storms / km / year)
  * DSRR - directional storm recurrence rate        (storms / degree / year),
           plus the recentered heading pdf/cdf, mean, and standard deviation

both ANNUALLY and per CALENDAR MONTH (Jan-Dec; the twelve monthly rates sum to
the annual rate), for four intensity bins by deficit dp = 1013 - Cp:
  All (dp >= MIN_DP) | Low [MIN_DP, DP_LOW) | Med [DP_LOW, DP_MED) | High [DP_MED, inf)

Outputs (data/outputs/). Non-plot files carry the HURDAT vintage <v> =
<start>-<end>_<created> (NHC file date), matching the AHD source, e.g.
srr_atlantic_1851-2025_20260227.csv:
  selection_<basin>_<v>.csv  - per-CRL selected TCs (representative point + closest approach)
  srr_<basin>_<v>.csv        - annual + monthly omnidirectional SRR per bin
  dsrr_<basin>_<v>.csv       - directional mean/stdv heading per bin
  dsrr_<basin>_<v>.npz       - full DSRR arrays (rate, pdf, cdf, monthly) per bin
  srr_<R>km/srr_<R>km_<basin>_<v>.csv - optional SRR_<R>km variant (TC/yr; SRR_RADIAL)
  plots/selection_<basin>/CHS_<Basin>_CRL_<NNNN>.png  - optional per-CRL maps

================================================================================
METHOD (ports the CHS MATLAB: StormSelection + SRR_GKF)
================================================================================
  1. Load the CRL set (ID,lat,lon) and the augmented HURDAT2 best-track.
  2. Per CRL, select every TC within MAX_DIST; the representative point maximizes
     GaussianWeights(K_SIZE) * (1013 - Cp); record the closest-approach distance.
  3. SRR  = (1/Nyrs) * sum_i Wi,  Wi = distance Gaussian kernel (sigma = K_SIZE).
  4. DSRR = (1/Nyrs) * sum_i Wd_i(theta) * Wi, heading kernel (sigma = DIR_KERNEL);
     normalized + recentered into a heading pdf/cdf (mean, stdv).
  5. Repeat per intensity bin and per month of closest approach.
  6. (Optional) map the selected TCs per CRL over a Natural Earth basemap.

Run
---
  1. pip install -r requirements.txt
  2. Edit the USER OPTIONS below, then:  python run_tc_climatological_analysis.py
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Sibling augmented_hurricane_database module outputs (the default HURDAT source).
_AHD_OUTPUTS = ROOT.parent / "augmented_hurricane_database" / "data" / "outputs"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Basin selection ──────────────────────────────────────────────────────────
# "atlantic", "pacific", or "both". Both are enabled now that a Pacific CRL set
# is available.
BASIN = "both"

# ── CRL sets (Coastal Reference Locations) ───────────────────────────────────
# Source files live under data/inputs/raw/ (absolute paths also accepted). The
# Atlantic file is CSV (ID,lat,lon); the Pacific file is tab-delimited
# (Latitude,Longitude,Region,ID). None disables a basin.
ATLANTIC_CRL_FILE = "CHS_Atl_CRLs_v1.6.csv"
PACIFIC_CRL_FILE  = "CHS_PAC_CRLs_v1.2.txt"

# ── Augmented HURDAT2 source (augmented_hurricane_database output) ────────────
# None auto-links to the newest augmented_hurdat2_<basin>_*.csv under
# AHD_OUTPUTS_DIR. Set a path to pin your own file (absolute, or rel. data/inputs/).
ATLANTIC_HURDAT_FILE = None
PACIFIC_HURDAT_FILE  = None
AHD_OUTPUTS_DIR      = _AHD_OUTPUTS

# ── GKF / selection parameters (defaults match the CHS MATLAB) ───────────────
K_SIZE       = 200.0    # distance Gaussian kernel size (km)
DIR_KERNEL   = 30.0     # heading Gaussian kernel size (deg)
MAX_DIST     = 600.0    # storm-selection cutoff distance (km)
MAX_CP       = 1005.0   # drop fixes with central pressure above this (hPa)
REF_PRESSURE = 1013.0   # reference for the deficit dp = 1013 - Cp
START_YEAR   = 1938     # first season counted in the rate
END_YEAR     = None     # last season; None -> max year present in the HURDAT data
MIN_DP       = 8.0      # overall intensity floor (hPa)
DP_LOW       = 28.0     # Low/Med deficit boundary (hPa)
DP_MED       = 48.0     # Med/High deficit boundary (hPa)

# ── SRR-within-a-radius variant: SRR_<R>km (optional, OFF by default) ─────────
# A 2nd variant of the SRR results only (not DSRR): SRR_<R>km = SRR · (2·R), the
# rate (storms/km/yr) times the 2R-km diameter -> the expected storms / year
# within SRR_RADIUS_KM of each CRL (TC/yr). Written to data/outputs/srr_<R>km/ and,
# when the maps are on, plotted in separate plots/selection_<R>km_<basin>/ folders.
SRR_RADIAL    = True
SRR_RADIUS_KM = K_SIZE   # radius (km); e.g., R = 200 km
                         # the multiplier is the 2R diameter (400 km)

# ── Per-CRL selected-TC maps (optional) ──────────────────────────────────────
# Each map shows the selected TCs colored by intensity and the SRR (All, High,
# Med, Low). Needs matplotlib + pyshp; the Natural Earth basemap downloads once
# into data/inputs/raw/naturalearth/. PLOT_JOBS parallelizes.
#   PLOT_SELECTION         - one annual map per CRL (-> plots/selection_<basin>/)
#   PLOT_SELECTION_MONTHLY - one map per CRL AND month, Jan-Dec, with that month's
#                            storms + SRR, sequenced CRL1->months, CRL2->months
#                            (-> plots/selection_monthly_<basin>/). HIGH VOLUME:
#                            ~12 x the annual map count, so off by default.
PLOT_SELECTION         = True
PLOT_SELECTION_MONTHLY = True
PLOT_DIR               = DATA / "outputs" / "plots"
PLOT_JOBS              = None    # None/0 = auto (cores-1), 1 = serial
BASEMAP_RESOLUTION     = "50m"   # Natural Earth resolution: "10m", "50m", or "110m"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA / "outputs"

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================

import sys

_BACKEND_PY = ROOT / "backend" / "python"
if str(_BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PY))


CONFIG = {
    "basins":               BASIN,
    "input_dir":            DATA / "inputs",
    "output_dir":           OUTPUT_DIR,
    "atlantic_crl_file":    ATLANTIC_CRL_FILE,
    "pacific_crl_file":     PACIFIC_CRL_FILE,
    "atlantic_hurdat_file": ATLANTIC_HURDAT_FILE,
    "pacific_hurdat_file":  PACIFIC_HURDAT_FILE,
    "ahd_outputs_dir":      AHD_OUTPUTS_DIR,
    "k_size":               K_SIZE,
    "dir_kernel":           DIR_KERNEL,
    "max_dist":             MAX_DIST,
    "max_cp":               MAX_CP,
    "ref_pressure":         REF_PRESSURE,
    "start_year":           START_YEAR,
    "end_year":             END_YEAR,
    "min_dp":               MIN_DP,
    "dp_low":               DP_LOW,
    "dp_med":               DP_MED,
    "srr_radial":           SRR_RADIAL,
    "srr_radius_km":        SRR_RADIUS_KM,
    "plot_selection":       PLOT_SELECTION,
    "plot_monthly":         PLOT_SELECTION_MONTHLY,
    "plot_dir":             PLOT_DIR,
    "plot_jobs":            PLOT_JOBS,
    "basemap_resolution":   BASEMAP_RESOLUTION,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless runs (no file edits needed)."""
    import argparse
    p = argparse.ArgumentParser(
        description="CRL-based tropical-cyclone storm recurrence rates (SRR/DSRR).")
    p.add_argument("--basin", choices=["atlantic", "pacific", "both"],
                   help="Override BASIN.")
    p.add_argument("--plots", dest="plots", action="store_true", default=None,
                   help="Render the per-CRL selected-TC maps.")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="Disable the per-CRL maps.")
    args = p.parse_args()
    config = dict(config)
    if args.basin:
        config["basins"] = args.basin
    if args.plots is not None:
        config["plot_selection"] = args.plots
    return config


if __name__ == "__main__":
    cfg = _apply_cli(CONFIG)
    from importlib import import_module
    result = import_module("main_tc_climatological_analysis").run(cfg)
    print("\n[tca] done:")
    for basin, r in result.results.items():
        bits = []
        if r.n_maps:
            bits.append(f"{r.n_maps:,} maps")
        if r.n_monthly_maps:
            bits.append(f"{r.n_monthly_maps:,} monthly")
        extra = f"  ({', '.join(bits)})" if bits else ""
        print(f"      {basin:9s} {r.n_crls:>5,} CRLs  {r.n_selected:>8,} CRL-TC pairs  "
              f"Nyrs={r.nyrs}{extra}  -> {r.srr_path}")
