"""run_storm_surge_hydrograph - SSH launcher (CyHAN v2.1 5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Storm Surge Hydrograph (SSH) module. The operator edits
the USER OPTIONS block below and runs the script. No orchestration logic lives here
- the launcher hands the option block to ``main_storm_surge_hydrograph.run``.

================================================================================
WHAT SSH PRODUCES
================================================================================
For each coastal save point, SSH reads the synthetic-TC storm-surge time series
(one CSV per save point, 15-min steps, metres above NAVD88; -99999 = dry) and
builds a UNIT (scalable) HYDROGRAPH: a single dimensionless surge shape, peak = 1,
scaled by TWO parameters, a peak surge elevation and an equivalent width (a
timescale), to reconstruct a hydrograph (double normalization).

Method (per save point):
  1. Ground elevation above NAVD88 = -depth (the staID depth column is positive
     down). Surge above ground a(t) = elevation - ground; dry (-99999) -> 0.
  2. Each storm is normalized by its own peak surge (n = a/peak), shifted so time
     = 0 is the PEAK, and its time rescaled by the equivalent width W = (integral
     of a)/peak, giving dimensionless time s = tau/W (double normalization, which
     removes the large duration spread that peak-only normalization leaves).
  3. The storms are averaged into one canonical shape C(s), C(0) = 1.
  4. Separate parametric limbs are fit, rising (s <= 0) and falling (s >= 0), each
     a generalized Gaussian C(s) = exp(-0.5 (|s|/sigma)^p).
  5. A hydrograph for a target peak elevation P and equivalent width W is
     E(tau) = ground + C(tau/W)*(P - ground). W may be supplied directly, taken
     from the point's distribution, or obtained from an observed ACTUAL DURATION
     (time above max(ground, MHHW) + 0.30 m) via the canonical level-width.

Outputs (data/outputs/):
  unit_hydrograph_SP#####.csv     - s_dimensionless, u_empirical, u_parametric
  scaled/hydrograph_SP#####_peak<P>m.csv     - peak-scaling examples (m NAVD88)
  scaled/hydrograph_SP#####_widthenv_*.csv   - equivalent-width envelope (P25/P50/P75)
  ssh_parameters.csv              - per save point: geometry, peaks, equiv-width, duration, limb fit
  plots/SSH_SP#####.png           - canonical shape + fit, peak scaling, width envelope
  plots/SSH_ensemble_SP#####.png  - unnormalized peak-aligned ensemble (m NAVD88)

Run
---
  1. pip install -r requirements.txt  (numpy, pandas, pydantic, scipy, matplotlib)
  2. Edit the USER OPTIONS below, then:  python run_storm_surge_hydrograph.py
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Input files (under data/inputs/raw/) ─────────────────────────────────────
STAID_FILE      = "CTXCS_staID.csv"
# {sp} is the 5-digit save-point id (e.g. 03911). The loader finds all matches.
SURGE_FILE_GLOB = "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv"
TIME_FILE       = "CTXCS_TP_SYN_Tides_0_SLC_0_time.csv"
# Restrict to a subset of save-point ids, or None for all found.
SAVE_POINTS     = None

# ── Data semantics ───────────────────────────────────────────────────────────
DT_HOURS        = 0.25          # time step (15 min)
DRY_VALUE       = -99999.0      # dry sentinel (water below ground)
DEPTH_POS_DOWN  = True          # ground elev (m NAVD88) = -depth column

# ── Unit-hydrograph construction ─────────────────────────────────────────────
# "double_norm" (recommended): canonical shape over dimensionless time s = tau/D,
# reconstructed from peak AND duration (most accurate; see the whitepaper comparison).
# "amplitude": legacy peak-only shape (leaves duration variability; baseline).
METHOD          = "double_norm"
MIN_WET_SAMPLES = 5             # skip a storm with fewer above-ground samples
WINDOW_HOURS    = None          # peak-aligned half-window; None -> auto from data
MAX_WINDOW_HOURS = 72.0         # cap for the auto window
AGGREGATE       = "mean"        # "mean" or "median" across a point's storms

# ── Parametric limb fit ──────────────────────────────────────────────────────
PARAMETRIC      = True          # fit rising/falling generalized-Gaussian limbs

# ── Actual duration (time above a physical threshold) ────────────────────────
# Threshold z0 = max(ground, MHHW) + offset: "offset above ground" for overland points,
# "offset above MHHW" for overwater points. Converts to/from the equivalent width via
# the canonical level-width. MHHW (m NAVD88) is None here (no-tide CTXS run, all
# overland); set it (or a future per-point file) when overwater points are present.
ACTUAL_DUR_OFFSET_M = 0.30
MHHW_NAVD88         = None

# ── Scaled-hydrograph examples ───────────────────────────────────────────────
# "auto" -> each point's observed median and max peak; or a list of peak
# elevations (m NAVD88) applied to every point; or None for none.
SCALE_PEAKS     = "auto"

# ── Plots ────────────────────────────────────────────────────────────────────
PLOTS           = True

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR      = DATA / "outputs"

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
    "input_dir":              DATA / "inputs",
    "output_dir":             OUTPUT_DIR,
    "staid_file":             STAID_FILE,
    "surge_file_glob":        SURGE_FILE_GLOB,
    "time_file":              TIME_FILE,
    "save_points":            SAVE_POINTS,
    "method":                 METHOD,
    "dt_hours":               DT_HOURS,
    "dry_value":              DRY_VALUE,
    "depth_is_positive_down": DEPTH_POS_DOWN,
    "min_wet_samples":        MIN_WET_SAMPLES,
    "window_hours":           WINDOW_HOURS,
    "max_window_hours":       MAX_WINDOW_HOURS,
    "aggregate":              AGGREGATE,
    "parametric":             PARAMETRIC,
    "actual_duration_offset_m": ACTUAL_DUR_OFFSET_M,
    "mhhw_navd88":            MHHW_NAVD88,
    "scale_peaks":            SCALE_PEAKS,
    "plots":                  PLOTS,
}


def _apply_cli(config: dict) -> dict:
    import argparse
    p = argparse.ArgumentParser(
        description="Unit / scalable storm-surge hydrographs per save point (SSH).")
    p.add_argument("--no-plots", dest="plots", action="store_false", default=None,
                   help="Disable the per-save-point plots.")
    p.add_argument("--aggregate", choices=["mean", "median"], help="Override AGGREGATE.")
    p.add_argument("--method", choices=["double_norm", "amplitude"], help="Override METHOD.")
    args = p.parse_args()
    config = dict(config)
    if args.plots is not None:
        config["plots"] = args.plots
    if args.aggregate:
        config["aggregate"] = args.aggregate
    if args.method:
        config["method"] = args.method
    return config


if __name__ == "__main__":
    cfg = _apply_cli(CONFIG)
    from importlib import import_module
    result = import_module("main_storm_surge_hydrograph").run(cfg)
    print("\n[ssh] done:")
    for spid, r in sorted(result.results.items()):
        print(f"      SP{spid:05d}  {r.n_storms:3d} storms  ground={r.ground_elev:+.2f} m  "
              f"W_eq={r.median_equiv_width:.1f} h  -> {r.unit_path.name}")
