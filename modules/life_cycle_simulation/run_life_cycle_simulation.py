"""run_life_cycle_simulation - LCS launcher (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Life Cycle Simulation (LCS) module. The operator edits
the USER OPTIONS block below and runs the script. No orchestration logic lives
here - the launcher hands the option block to ``api_life_cycle_simulation.run``
per §5.3.

================================================================================
WHAT LCS PRODUCES
================================================================================
For a chosen Coastal Reference Location (CRL) and a radius of influence either
side of it, LCS draws a synthetic catalog of tropical cyclones over a requested
life-cycle length (e.g. 100 years) and a requested number of independent
realizations (e.g. 1000). It is driven entirely by the storm_climatology_analysis
(SCA) storm recurrence rate (SRR): the annual SRR sets how often TCs occur, the
Low/Med/High SRR ratios set each TC's intensity stratum, and the seasonal SRR
shape sets each TC's calendar day of closest approach to the CRL.

The catalog is the raw material for downstream life-cycle / event-based hazard
work: every row is one synthetic TC with its realization, year, intensity
stratum, and month/day of occurrence.

================================================================================
METHOD & FORMULATION
================================================================================
For the chosen CRL, with SRR read from the SCA tables (units in brackets):

  1. Poisson rate    lambda = SRR_all * (2 * R)            [TC / yr]
     SRR_all is the omnidirectional annual rate density [TC / km / yr]; R is the
     radius of influence [km]; (2 * R) is the along-coast band diameter [km].
  2. Annual activity  N(y) ~ Poisson(lambda)  for each year y = 1..SIM_YEARS,
     drawn independently for each of N_REALIZATIONS life cycles.
  3. Intensity stratum  for each TC ~ Categorical(p_low, p_med, p_high), with
     p_s = SRR_s / (SRR_low + SRR_med + SRR_high)          [-]
  4. Day of occurrence  for each TC ~ the chosen stratum's seasonal SRR shape over
     day-of-year 1..365 (a smooth daily SRR curve, or the monthly SRR spread
     uniformly within each month), then mapped to a calendar month and day.
  5. Repeat 2 to 4 across all realizations (drawn in one vectorized pass).

Units note: SRR is a rate DENSITY [TC / km / yr]; multiplying by the 2R-km band
turns it into an expected count [TC / yr], the Poisson mean at the CRL.

Run
---
  1. pip install -r requirements.txt
  2. Edit the USER OPTIONS below, then:  python run_life_cycle_simulation.py
     ...or from the repository root:
         python modules/life_cycle_simulation/run_life_cycle_simulation.py

Outputs (data/outputs/), one pair per CRL (tag = crl<NNNN>_R<R>km_<Y>yr_<N>real):
  lcs_catalog_<tag>.csv  - one row per synthetic TC (realization, year, event,
                           intensity, month, day, doy)
  lcs_summary_<tag>.csv  - per-realization TC counts overall and by stratum
  plots/crl<NNNN>/        - optional per-CRL folder with all that CRL's figures
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Sibling storm_climatology_analysis module outputs (the default SRR source).
_SCA_OUTPUTS = ROOT.parent / "storm_climatology_analysis" / "data" / "outputs"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Storm type ────────────────────────────────────────────────────────────────
# "tc"  - tropical cyclones (the implemented life-cycle Monte-Carlo).
# "etc" - extratropical cyclones (PLACEHOLDER; not yet implemented).
# Overridable on the command line with --storm-type.
STORM_TYPE = "tc"

# ── SRR source (the storm_climatology_analysis output) ───────────────────────
# INPUT_CSV is the per-CRL annual + monthly SRR table in TC/km/yr,
# i.e. srr_<basin>_<v>.csv (NOT the srr_<R>km variant, already x diameter).
# DAILY_CSV is the companion daily SRR table; None auto-locates
# srr_daily_<basin>_<v>.csv next to INPUT_CSV (used only by DAY_METHOD="daily").
INPUT_CSV = _SCA_OUTPUTS / "srr_atlantic_1938-2025_20260227.csv"
DAILY_CSV = None

# ── Site and footprint ───────────────────────────────────────────────────────
# CRL_IDS  - one CRL id, or a list of ids; one synthetic catalog per CRL.
# RADIUS_KM- radius of influence either side of the CRL (default 200 km; change
#            freely). lambda = SRR * (2 * RADIUS_KM).
CRL_IDS   = [844, 128]
RADIUS_KM = 200.0

# ── Simulation size ──────────────────────────────────────────────────────────
SIM_YEARS      = 100      # length of each synthetic life cycle (years)
N_REALIZATIONS = 1000     # independent realizations of that life cycle

# ── Day-of-occurrence model ──────────────────────────────────────────────────
# "daily"   - draw the day-of-year from the smooth per-stratum daily SRR (needs
#             DAILY_CSV / the auto-located companion). Physically richer.
# "monthly" - draw the month from the per-stratum monthly SRR, then a uniform day
#             within that month. Uses only INPUT_CSV.
DAY_METHOD = "daily"

# Random seed for reproducibility (None = a fresh nondeterministic stream). Each
# CRL draws from an independent sub-stream derived from (SEED, crl_id).
SEED = 12345

# ── Serial correlation + clustering of annual counts (off by default) ─────────
# CORRELATION=False keeps the independent Poisson baseline exactly. When True, the
# annual rate gains year-to-year memory and/or overdispersion, so active and quiet
# years cluster (the annual mean rate is preserved). The three parameters below are
# CALIBRATED from each CRL's historical annual counts (the SCA selection table)
# whenever they are left as None; set a number to override the estimate.
#   AR_PHI         - AR(1) persistence of the latent climate state, [0, 1)
#   AR_BETA        - sensitivity of the log annual-rate to the state (lag-1 ACF)
#   OVERDISPERSION - variance of the i.i.d. annual rate multiplier (Fano = 1 +
#                    lambda*OVERDISPERSION)
# Note: a sparse, low-rate CRL typically calibrates to ~0 (Poisson), which is the
# statistically appropriate result; the clustering signal is basin/regional.
CORRELATION    = True
AR_PHI         = None    # None = calibrate from history; or set a value to override
AR_BETA        = None
OVERDISPERSION = None
# REGIONAL_POOL_KM: when calibrating, pool every CRL within this many km of the
# target so the basin/regional clustering signal is estimated from many records
# instead of one sparse history. None = per-CRL calibration (default); e.g. 300.
REGIONAL_POOL_KM = None

# ── Event sequencing ──────────────────────────────────────────────────────────
# Add the chronological event timeline to the catalog: event_time (years), a
# per-realization chronological order (seq), and the inter-arrival waiting time
# from the previous event (wait_yr).
SEQUENCING = True

# ── Visualization suite (optional, off by default) ───────────────────────────
# MAKE_PLOTS is the master switch. PLOTS selects which per-CRL figures to render;
# ["all"] renders every figure. To toggle individually, list only the keys you
# want (comment/uncomment the line below). Needs matplotlib.
#   annual_fan     - TCs/year median + percentile bands vs simulation year
#   annual_heatmap - year x count, color = fraction of realizations
#   annual_violin  - per-year count distribution (year-binned if long)
#   cumulative     - ensemble running-total trajectories + 5-95% envelope
#   count_dist     - TCs/year pmf vs Poisson, and per-realization totals
#   seasonality    - monthly + day-of-year occurrence by stratum vs driving SRR
#   waiting_times  - inter-event waiting time and time-to-first-TC
#   clustering     - annual-count ACF + sample trajectories (Fano); serial corr.
#   diagnostic     - the three-panel quick QC (count / stratum / seasonality)
MAKE_PLOTS = True
PLOTS      = ["all"]
#PLOTS = ["annual_fan", "annual_heatmap", "annual_violin", "cumulative",
#         "count_dist", "seasonality", "waiting_times", "clustering", "diagnostic"]
PLOT_DIR   = DATA / "outputs" / "plots"

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
    "storm_type":     STORM_TYPE,
    "input_csv":      INPUT_CSV,
    "daily_csv":      DAILY_CSV,
    "crl_ids":        CRL_IDS,
    "radius_km":      RADIUS_KM,
    "sim_years":      SIM_YEARS,
    "n_realizations": N_REALIZATIONS,
    "day_method":     DAY_METHOD,
    "seed":           SEED,
    "correlation":    CORRELATION,
    "ar_phi":         AR_PHI,
    "ar_beta":        AR_BETA,
    "overdispersion": OVERDISPERSION,
    "regional_pool_km": REGIONAL_POOL_KM,
    "sequencing":     SEQUENCING,
    "make_plots":     MAKE_PLOTS,
    "plots":          PLOTS,
    "plot_dir":       PLOT_DIR,
    "output_dir":     OUTPUT_DIR,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless runs (no file edits needed)."""
    import argparse
    p = argparse.ArgumentParser(
        description="Monte-Carlo synthetic TC life cycles for a CRL from the SCA SRR.")
    p.add_argument("--storm-type", choices=["tc", "etc"],
                   help="Override STORM_TYPE (tc=tropical, implemented; etc=placeholder).")
    p.add_argument("--input-csv", help="Override INPUT_CSV (srr_<basin>_<v>.csv).")
    p.add_argument("--daily-csv", help="Override DAILY_CSV (srr_daily_<basin>_<v>.csv).")
    p.add_argument("--crl", type=int, nargs="+", dest="crl_ids",
                   help="Override CRL_IDS (one or more CRL ids).")
    p.add_argument("--radius-km", type=float, help="Override RADIUS_KM.")
    p.add_argument("--years", type=int, dest="sim_years", help="Override SIM_YEARS.")
    p.add_argument("--realizations", type=int, dest="n_realizations",
                   help="Override N_REALIZATIONS.")
    p.add_argument("--day-method", choices=["daily", "monthly"],
                   help="Override DAY_METHOD.")
    p.add_argument("--seed", type=int, help="Override SEED.")
    p.add_argument("--correlation", dest="correlation", action="store_true", default=None,
                   help="Enable serial correlation + overdispersion of annual counts.")
    p.add_argument("--no-correlation", dest="correlation", action="store_false",
                   help="Disable correlation (independent Poisson baseline).")
    p.add_argument("--ar-phi", type=float, help="Override AR_PHI (AR(1) persistence).")
    p.add_argument("--ar-beta", type=float, help="Override AR_BETA (log-rate sensitivity).")
    p.add_argument("--overdispersion", type=float, help="Override OVERDISPERSION.")
    p.add_argument("--regional-pool-km", type=float, dest="regional_pool_km",
                   help="Pool CRLs within this many km for the calibration (regional).")
    p.add_argument("--no-sequencing", dest="sequencing", action="store_false",
                   default=None, help="Skip the chronological event timeline columns.")
    p.add_argument("--plots", dest="plots_on", action="store_true", default=None,
                   help="Render the per-CRL figure suite (master switch on).")
    p.add_argument("--no-plots", dest="plots_on", action="store_false",
                   help="Disable all per-CRL figures (master switch off).")
    p.add_argument("--plot", nargs="+", dest="plot_keys",
                   help="Render only these figures (e.g. --plot annual_fan cumulative); "
                        "implies --plots. Use 'all' for every figure.")
    args = p.parse_args()
    config = dict(config)
    for key in ("storm_type", "input_csv", "daily_csv", "crl_ids", "radius_km",
                "sim_years", "n_realizations", "day_method", "seed",
                "ar_phi", "ar_beta", "overdispersion", "regional_pool_km"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    if args.correlation is not None:
        config["correlation"] = args.correlation
    if args.sequencing is not None:
        config["sequencing"] = args.sequencing
    if args.plots_on is not None:
        config["make_plots"] = args.plots_on
    if args.plot_keys:                              # selecting figures implies plots on
        config["plots"] = args.plot_keys
        config["make_plots"] = True
    return config


if __name__ == "__main__":
    cfg = _apply_cli(CONFIG)
    from importlib import import_module
    try:
        result = import_module("api_life_cycle_simulation").run(cfg)
    except NotImplementedError as exc:
        raise SystemExit(f"[lcs] {exc}")
    print("\n[lcs] done:")
    for crl_id, r in result.results.items():
        figs = f"  ({len(r.plot_paths)} figs)" if r.plot_paths else ""
        print(f"      CRL {crl_id:<5} lambda={r.lam:.4f} TC/yr  "
              f"{r.n_events:,} TCs over {result.n_realizations:,} x "
              f"{result.sim_years} yr{figs}  -> {r.catalog_path}")
