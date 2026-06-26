"""run_joint_distribution_model - JDM launcher (CyHAN v2.2 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Joint Distribution Model (JDM) module. The operator edits
the USER OPTIONS block below and runs the script. No orchestration logic lives here;
the launcher hands the option block to ``api_joint_distribution_model.run`` per §5.3.

================================================================================
WHAT JDM PRODUCES
================================================================================
For each Coastal Reference Location (CRL) and intensity bin (All / High / Med / Low
by central-pressure deficit Dp), JDM characterizes the JOINT DISTRIBUTION of the
tropical-cyclone parameters [Heading, Dp, Rmax, forward translation Vt] used by the
Joint Probability Method:

  * MARGINAL distributions per parameter: Dp = (jitter-bootstrapped) truncated
    Weibull, Rmax = lognormal, Vt = normal (High/Med) or lognormal (Low), heading =
    the SCA DSRR directional distribution.
  * a META-GAUSSIAN COPULA linking the four parameters (Kendall tau -> Gaussian rho).

It consumes the storm_climatology_analysis (SCA) outputs: the per-CRL selected-TC
table (selection_<basin>_<v>.csv) and the DSRR arrays (dsrr_<basin>_<v>.npz). This
is the parameter-characterization layer that synthetic-storm generation builds on.

================================================================================
METHOD
================================================================================
  1. Distance-weighted adjustment: rescale each storm's parameter so the sample
     mean/std become the Gaussian-distance-weighted mean/std (z-scores preserved);
     recenter heading on the DSRR circular mean. Bin by adjusted Dp.
  2. Marginals per CRL per intensity (Dp truncated Weibull w/ bootstrap, Rmax
     lognormal, Vt normal/lognormal, heading from DSRR).
  3. Meta-Gaussian copula: rho = sin(pi/2 * Kendall_tau([Hd,Dp,Rmax,Vt])).

Run
---
  1. pip install -r requirements.txt   (needs scipy)
  2. Run SCA first (it writes selection_* and dsrr_*), then edit the USER OPTIONS
     below and:  python run_joint_distribution_model.py

Outputs (data/outputs/), tagged <v> from the SCA selection filename:
  jdm_marginals_<basin>_<v>.csv  - per CRL x intensity x parameter marginal params
  jdm_copula_<basin>_<v>.npz     - per CRL x intensity Kendall tau + Gaussian rho (4x4)
  jdm_adjusted_<basin>_<v>.csv   - adjusted, stratum-labeled per-storm [Hd,Dp,Rmax,Vt]
  plots/marginals_<basin>/CHS_<Basin>_CRL_<NNNN>.png  - optional per-CRL marginal fits
  plots/copula_<basin>/CHS_<Basin>_CRL_<NNNN>.png     - optional per-CRL copula (rho
                                                        heatmaps + All-bin pairs)
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Sibling storm_climatology_analysis module outputs (the default SCA source).
_SCA_OUTPUTS = ROOT.parent / "storm_climatology_analysis" / "data" / "outputs"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Storm type ────────────────────────────────────────────────────────────────
# "tc"  - tropical cyclones (the implemented analysis). "etc" - placeholder.
STORM_TYPE = "tc"

# ── Basin selection ──────────────────────────────────────────────────────────
# "atlantic" only for now (Pacific SCA outputs do not exist yet); "both" later.
BASIN = "atlantic"

# ── SCA source (the storm_climatology_analysis outputs) ──────────────────────
# SCA_OUTPUTS_DIR auto-links the newest selection_<basin>_*.csv and
# dsrr_<basin>_*.npz. Set explicit files to pin (absolute, or rel. data/inputs).
SCA_OUTPUTS_DIR         = _SCA_OUTPUTS
ATLANTIC_SELECTION_FILE = None
ATLANTIC_DSRR_FILE      = None

# ── Adjustment + intensity binning ───────────────────────────────────────────
REF_PRESSURE = 1013.0      # deficit reference: Dp = REF_PRESSURE - Cp (hPa)
# Which selection Cp drives the deficit Dp = 1013 - Cp:
#   "cp_gauss"   - Cp at SCA's representative fix (the Gaussian distance-weight x
#                  deficit fix; the default), so each storm's JDM intensity bin
#                  matches its SCA SRR stratum.
#   "cp_mindist" - Cp at the closest-approach fix to the CRL.
CP_SOURCE    = "cp_gauss"
START_YEAR   = 1938        # drop selected TCs before this season
MIN_DP       = 8.0         # overall deficit floor (hPa)
DP_LOW       = 28.0        # Low/Med deficit boundary (hPa)
DP_MED       = 48.0        # Med/High deficit boundary (hPa)
VT_CLIP      = (1.0, 152.0)   # translation-speed clip after adjustment (km/h)
RMAX_CLIP    = (8.0, 200.0)   # Rmax clip after adjustment (km)

# ── Dp marginal bootstrap ────────────────────────────────────────────────────
# Per-CRL jitter-bootstrap count for the Dp Weibull (default 10000). This is the
# heavy step; lower it for a quick run. N_JOBS parallelizes over CRLs.
N_BOOT = 10000
SEED   = 12345             # None for a nondeterministic bootstrap
N_JOBS = None              # None/0 = auto (cores-1), 1 = serial

# ── Per-CRL diagnostic plots (optional, off by default) ──────────────────────
MAKE_PLOTS = False
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
    "storm_type":              STORM_TYPE,
    "basins":                  BASIN,
    "input_dir":               DATA / "inputs",
    "output_dir":              OUTPUT_DIR,
    "sca_outputs_dir":         SCA_OUTPUTS_DIR,
    "atlantic_selection_file": ATLANTIC_SELECTION_FILE,
    "atlantic_dsrr_file":      ATLANTIC_DSRR_FILE,
    "ref_pressure":            REF_PRESSURE,
    "cp_source":               CP_SOURCE,
    "start_year":              START_YEAR,
    "min_dp":                  MIN_DP,
    "dp_low":                  DP_LOW,
    "dp_med":                  DP_MED,
    "vt_clip":                 VT_CLIP,
    "rmax_clip":               RMAX_CLIP,
    "n_boot":                  N_BOOT,
    "seed":                    SEED,
    "n_jobs":                  N_JOBS,
    "make_plots":              MAKE_PLOTS,
    "plot_dir":                PLOT_DIR,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless runs (no file edits needed)."""
    import argparse
    p = argparse.ArgumentParser(
        description="Per-CRL JPM joint distribution of TC parameters (marginals + copula).")
    p.add_argument("--storm-type", choices=["tc", "etc"], help="Override STORM_TYPE.")
    p.add_argument("--basin", choices=["atlantic", "pacific", "both"],
                   help="Override BASIN.")
    p.add_argument("--cp-source", choices=["cp_mindist", "cp_gauss"],
                   help="Override CP_SOURCE.")
    p.add_argument("--n-boot", type=int, help="Override N_BOOT (Dp bootstrap count).")
    p.add_argument("--n-jobs", type=int, help="Override N_JOBS (CRL parallelism).")
    p.add_argument("--seed", type=int, help="Override SEED.")
    p.add_argument("--plots", dest="plots", action="store_true", default=None,
                   help="Render the per-CRL marginal diagnostic figures.")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="Disable the per-CRL figures.")
    args = p.parse_args()
    config = dict(config)
    if args.storm_type:
        config["storm_type"] = args.storm_type
    if args.basin:
        config["basins"] = args.basin
    if args.cp_source:
        config["cp_source"] = args.cp_source
    for key in ("n_boot", "n_jobs", "seed"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    if args.plots is not None:
        config["make_plots"] = args.plots
    return config


def main():
    cfg = _apply_cli(CONFIG)
    from importlib import import_module
    try:
        result = import_module("api_joint_distribution_model").run(cfg)
    except NotImplementedError as exc:
        raise SystemExit(f"[jdm] {exc}")
    print("\n[jdm] done:")
    for basin, r in result.results.items():
        extra = f"  ({r.n_plots:,} figs)" if r.n_plots else ""
        print(f"      {basin:9s} {r.n_crls:>5,} CRLs  {r.n_records:>7,} marginal rows"
              f"{extra}  -> {r.marginals_path}")

if __name__ == "__main__":
    main()
