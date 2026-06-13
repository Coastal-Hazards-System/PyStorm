"""run_storm_climatology_analysis - SCA launcher (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Storm Climatology Analysis (SCA) module.
The operator edits the USER OPTIONS block below and runs the script. No
orchestration logic lives here - the launcher hands the option block to
``api_storm_climatology_analysis.run`` per §5.3.

================================================================================
WHAT SCA PRODUCES
================================================================================
For each CHS Coastal Reference Location (CRL), SCA selects the tropical cyclones
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

It also computes a CONTINUOUS DAILY SRR: the omnidirectional rate as a smooth
seasonal cycle over day-of-year 1..365 at each CRL, using a circular (period-365)
day-of-year Gaussian kernel (DAY_KERNEL days). Units are TC/km/yr per day-of-year, a
RATE DENSITY (the annual SRR spread across the calendar), so the 365 daily values sum
to the annual SRR. The two time terms differ: "/yr" is the rate per hurricane season
(averaged over the record); "per day-of-year" is the density across the calendar (a
position in the season, not a 2nd time axis). See the DAILY SRR notes below.

================================================================================
DAILY SRR - HOW THE SPATIAL AND TEMPORAL KERNELS COMBINE
================================================================================
The daily SRR uses TWO kernels per storm, and they are MULTIPLIED together in one
sum, NOT applied one after the other (no intermediate smoothed field):

  * Spatial kernel  Wi = 1/(sqrt(2pi)*K_SIZE) * exp(-1/2 (D_i/K_SIZE)^2)
      a single SCALAR per storm, fixed by its closest-approach distance D_i. It
      sets HOW MUCH the storm counts (proximity to the CRL).
  * Temporal kernel Wt_i(d) = 1/(sqrt(2pi)*DAY_KERNEL) * exp(-1/2 (ddoy/DAY_KERNEL)^2)
      a 365-long CURVE per storm, fixed by its closest-approach day-of-year t_i and
      the circular (period-365) day difference ddoy. It sets WHEN in the year that
      amount is deposited.

  SRR_daily(d) = (1/Nyrs) * sum_i  Wt_i(d) * Wi   (TC/km/yr per day-of-year;
                                                   sum over the 365 days = annual SRR)

In code: srr_daily = (wt * wi[:, None]).sum(axis=0) / Nyrs, with wi shape (N,) and
wt shape (N, 365). The two roles are orthogonal: the scalar Wi scales the whole
temporal curve, then storms are summed. Because Wt integrates to 1 over the year,
sum_d SRR_daily(d) = SRR exactly (the daily curve only REDISTRIBUTES the annual
rate across the calendar). This mirrors the DSRR heading kernel Wd(theta)*Wi.

Dimensionality: this is a SEPARABLE 2-D kernel in (distance, time), K(D,t) =
W(D)*W(t), NOT a 3-D (x,y,t) kernel. The map geometry was already reduced to the
single radial distance D_i during selection, so there is no independent x/y; the
estimator only resolves distance and time (and, for DSRR, heading).

Why the temporal kernel is CIRCULAR (wrapped, period 365): day-of-year is a closed
loop, not a line segment (Dec 31 is followed by Jan 1). The day difference is wrapped
to (-182.5, 182.5], ddoy = ((d - t_i + 182.5) mod 365) - 182.5, so separation is the
SHORT way around the year: Dec 31 and Jan 1 are 1 day apart, not 364. A storm near
year-end therefore deposits weight into early January and vice versa. This is the
wrapped-normal (circular Gaussian), the same construction as the DSRR heading kernel
on the 360-deg circle. Two consequences: (a) mass that would spill past one end of the
year reappears at the other end, so Wt still integrates to 1 and the daily curve sums
exactly to the annual rate; a non-circular kernel would leak mass off the ends and
break additivity near Jan 1 / Dec 31; (b) the seasonal curve is continuous across the
year boundary with no edge effect, which matters for shoulder-season activity and any
basin whose season straddles the seam.

Day-of-year and the 365 vs 366 (leap-year) divergence: the grid is a FIXED 365-day
non-leap calendar keyed by CALENDAR DATE (doy = cum_nonleap[month-1] + day, clamped
to [1,365]); this is the standard CF "noleap"/"365_day" climatological calendar. The
DECISIVE reason for 365 over 366 is uniform year-EXPOSURE: a seasonal rate is events
per calendar day per year, so each grid day must be backed by the same number of years.
Every date except Feb 29 occurs once per year (Nyrs times); Feb 29 occurs only in leap
years (~Nyrs/4). On the 365-day grid Feb 29 is folded onto the Feb28/Mar1 boundary, so
all 365 days have identical exposure (Nyrs) and dividing by Nyrs is unbiased with no
special-casing. A 366-day grid would give Feb 29 its own structurally undersampled bin:
dividing it by Nyrs stamps a spurious ~75% trough there, and fixing it needs that one
bin divided by the leap-year count instead, a permanent correction for a day no TC
reaches. Calendar-date alignment also keeps the same date (e.g., Sep 10) on the same
grid index every year (correct for a seasonal climatology), and the sub-day
misalignment of post-Feb dates is far below the DAY_KERNEL bandwidth. The TC season
(~May-Nov) is far from Feb 29, so the leap effect is immaterial. A 366-day grid is
preferable only to resolve Feb 29 as a distinct date (with per-day exposure
normalization), which a smoothed seasonal rate never needs.

Choosing DAY_KERNEL: by weighted leave-one-out likelihood cross-validation on the
closest-approach day-of-year (analysis/day_kernel_sensitivity.py), the data-driven
optimum is ~10-11 days (Atlantic, 560 CRLs) and ~13-18 days (Pacific, 247 CRLs).
The 14-day default sits between the two basins' optima, inside both per-CRL IQRs,
at a negligible cross-validated cost (< 0.006 nats/storm). It is tunable below.

Outputs (data/outputs/). Non-plot files are tagged <v> = <start>-<end>_<created>:
the effective rate start year (START_YEAR), the last season, and the NHC HURDAT
file date. E.g. srr_atlantic_1938-2025_20260227.csv:
  selection_<basin>_<v>.csv  - per-CRL selected TCs (representative point + closest approach)
  srr_<basin>_<v>.csv        - annual + monthly omnidirectional SRR per bin
  srr_daily_<basin>_<v>.csv  - continuous daily SRR per bin (long form: crl_id,lat,lon,doy,...)
  dsrr_<basin>_<v>.csv       - directional mean/stdv heading per bin
  dsrr_<basin>_<v>.npz       - full DSRR arrays (rate, pdf, cdf, monthly, daily) per bin
  srr_<R>km/srr_<R>km_<basin>_<v>.csv - optional SRR_<R>km variant (TC/yr; SRR_RADIAL)
  plots/selection_<basin>/CHS_<Basin>_CRL_<NNNN>.png  - optional per-CRL annual maps
  plots/selection_monthly_<basin>/...                 - optional per-CRL x month maps
  plots/daily_<basin>/CHS_<Basin>_CRL_<NNNN>.png      - optional per-CRL daily SRR curves
                                                        (All/High/Med/Low vs day 1..365)
  plots/daily_<R>km_<basin>/...                       - optional SRR_<R>km daily curves

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
  6. Daily SRR(doy) = (1/Nyrs) * sum_i Wt_i(doy) * Wi, circular day-of-year kernel
     (sigma = DAY_KERNEL); the continuous seasonal cycle over days 1..365.
  7. (Optional) map the selected TCs per CRL over a Natural Earth basemap.

Run
---
  1. pip install -r requirements.txt
  2. Edit the USER OPTIONS below, then:  python run_storm_climatology_analysis.py
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

# ── Storm type ────────────────────────────────────────────────────────────────
# "tc"  - tropical cyclones (the implemented analysis, from the augmented HURDAT2).
# "etc" - extratropical cyclones (PLACEHOLDER; not yet implemented: the same GKF
#         recurrence-rate machinery would run on an ETC track source).
# Overridable on the command line with --storm-type.
STORM_TYPE = "tc"

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
K_SIZE       = 200.0    # distance Gaussian kernel size (km) (default = 200 deg)
DIR_KERNEL   = 30.0     # heading Gaussian kernel size (deg) (default = 30 deg)
DAY_KERNEL   = 14.0     # day-of-year Gaussian kernel size (days) for the daily SRR
                        # (default 14 d; LOO-CV optimum ~10-11 d Atl, ~13-18 d Pac; see docstring)
MAX_DIST     = 600.0    # storm-selection cutoff distance (km)
MAX_CP       = 1005.0   # drop fixes with central pressure above this (hPa)
REF_PRESSURE = 1013.0   # reference for the deficit dp = 1013 - Cp
START_YEAR   = 1938     # first season counted in the rate; None = entire HURDAT record.
                        # Clamped up to each basin's record start (Pacific begins 1949).
                        # This start year (effective) appears in the output filenames.
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
#   PLOT_SELECTION_DAILY   - one daily-SRR curve plot per CRL: the All/High/Med/Low
#                            rates over day-of-year 1..365 on one set of axes
#                            (-> plots/daily_<basin>/). An XY curve, not a map, so it
#                            needs only matplotlib (no basemap/pyshp). With SRR_RADIAL
#                            a 2nd SRR_<R>km daily plot is written to plots/daily_<R>km_<basin>/.
# The --no-plots / --plots CLI flags toggle ALL THREE products together.
PLOT_SELECTION         = True
PLOT_SELECTION_MONTHLY = True
PLOT_SELECTION_DAILY   = True
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
_COMMON_PY  = ROOT.parents[1] / "common" / "python"   # shared CyHAN common library (§5.2)
for _p in (_BACKEND_PY, _COMMON_PY):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


CONFIG = {
    "storm_type":           STORM_TYPE,
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
    "day_kernel":           DAY_KERNEL,
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
    "plot_daily":           PLOT_SELECTION_DAILY,
    "plot_dir":             PLOT_DIR,
    "plot_jobs":            PLOT_JOBS,
    "basemap_resolution":   BASEMAP_RESOLUTION,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless runs (no file edits needed)."""
    import argparse
    p = argparse.ArgumentParser(
        description="CRL-based storm recurrence rates (SRR/DSRR).")
    p.add_argument("--storm-type", choices=["tc", "etc"],
                   help="Override STORM_TYPE (tc=tropical, implemented; etc=placeholder).")
    p.add_argument("--basin", choices=["atlantic", "pacific", "both"],
                   help="Override BASIN.")
    p.add_argument("--plots", dest="plots", action="store_true", default=None,
                   help="Render all per-CRL plots (annual + monthly maps + daily SRR).")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="Disable all per-CRL plots (annual + monthly maps + daily SRR).")
    args = p.parse_args()
    config = dict(config)
    if args.storm_type:
        config["storm_type"] = args.storm_type
    if args.basin:
        config["basins"] = args.basin
    if args.plots is not None:                 # toggles all three products together
        config["plot_selection"] = args.plots
        config["plot_monthly"] = args.plots
        config["plot_daily"] = args.plots
    return config


if __name__ == "__main__":
    cfg = _apply_cli(CONFIG)
    from importlib import import_module
    try:
        result = import_module("api_storm_climatology_analysis").run(cfg)
    except NotImplementedError as exc:
        raise SystemExit(f"[sca] {exc}")
    print("\n[sca] done:")
    for basin, r in result.results.items():
        bits = []
        if r.n_maps:
            bits.append(f"{r.n_maps:,} maps")
        if r.n_monthly_maps:
            bits.append(f"{r.n_monthly_maps:,} monthly")
        if r.n_daily_plots:
            bits.append(f"{r.n_daily_plots:,} daily")
        extra = f"  ({', '.join(bits)})" if bits else ""
        print(f"      {basin:9s} {r.n_crls:>5,} CRLs  {r.n_selected:>8,} CRL-TC pairs  "
              f"Nyrs={r.nyrs}{extra}  -> {r.srr_path}")
