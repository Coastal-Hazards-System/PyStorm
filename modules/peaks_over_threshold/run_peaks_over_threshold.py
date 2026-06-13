"""run_peaks_over_threshold - POT launcher (CyHAN v2.1 §5.3).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Peaks-Over-Threshold (POT) module. The operator edits
the USER OPTIONS block (at the top of this file) and runs the script. No
orchestration logic lives here - the launcher hands the option block to
``main_peaks_over_threshold.run`` per §5.3, which dispatches the requested stages.

Stages (canonical order)
-------------------------
  download  fetch NOAA water level + tide prediction CSVs        -> data/inputs/raw/<station>/
  detrend   remove linear sea-level trend from the water level   -> data/inputs/processed/<station>/
  ntr       NTR = detrended water level - hourly tide            -> data/inputs/processed/<station>/
  pot       peaks over threshold of BOTH dwl and ntr             -> data/outputs/<station>/{dwl,ntr}_<station>_pot.csv

PRIMARY use is POT only (``STAGES = ["pot"]``) on a user-provided CSV. The
upstream NOAA -> NTR stages are SECONDARY and opt-in; add them to STAGES to
build the POT input from raw data, optionally in one full chain.

================================================================================
WHAT POT PRODUCES
================================================================================
POT extracts independent storm PEAKS from a continuous water-level / NTR time
series: the magnitudes (and times) that exceed an automatically chosen
threshold, DECLUSTERED into one value per storm event and trimmed to a fixed
average rate. These peaks are the input to the PST module's hazard analysis.

================================================================================
METHOD & FORMULATION
================================================================================
PRIMARY - POT extraction (the "pot" stage)
  1. Effective duration. The record length used for all rates is the EFFECTIVE
     duration = (# non-NaN hourly steps) / (365.25 × 24) years - gaps / missing
     data do not count (a 100-yr span that is 50% complete counts as ~50 yr).
  2. Iterative threshold search. The threshold is raised from START_PERCENTILE
     upward in STEP_SIZE increments (percentiles of the series). At each level
     the exceedances are DECLUSTERED into independent events by METHOD:
       "hydrograph" - consecutive exceedances within INTEREVENT_HOURS of one
           another form a single event; its peak is the event maximum.
       "peak_gap"   - a sample is dropped if it lies within INTEREVENT_HOURS of,
           and is no larger than, the previous retained peak.
     The event rate = (# events) / effective_duration. The search is ONE-SIDED:
     it keeps the HIGHEST threshold whose rate is still ≥ TARGET_EVENTS_PER_YEAR
     (the most selective cut that still meets the target), and flags it
     "converged" when that rate lands in [target, target + TOLERANCE].
  3. Rank-trim to an exact count. The retained peaks are rank-ordered by
     magnitude and trimmed to exactly round(TARGET_EVENTS_PER_YEAR ×
     effective_duration) of the largest, so the written sample has an effective
     rate of EXACTLY the target. (This is also what lets PST recover the record
     length from the peak count - keep PST's EVENTS_PER_YEAR equal to
     TARGET_EVENTS_PER_YEAR.)
  Output per series: {dwl,ntr}_<station>_pot.csv - the declustered peaks
  (datetime, value), ready for PST.

SECONDARY - build the POT input from NOAA data (optional upstream stages)
  download  Fetch hourly water level + hourly tide predictions per station/year
            from NOAA Tides & Currents into data/inputs/raw/<station>/.
  detrend   Remove the linear sea-level trend from the water level by least
            squares. "midpoint" centres time on the NTDE midpoint (the trend
            pivots there, matching NOAA datum convention); "ordinary" centres on
            the record mean. The slope is fitted from the record, or imposed via
            DETREND_SLOPE. Produces dwl = detrended water level.
  ntr       NTR = detrended water level − interpolated hourly tide - the
            non-tidal (meteorological) residual.
  With "pot" also in STAGES the chain feeds dwl and ntr straight into the POT
  extraction above (one POT sample per series).

Run (headless / CLI)
--------------------
Headless by design - figures are written to disk (no window opens), so this
runs unchanged over SSH, in a container, or under cron.

  1. Install dependencies once:
         pip install -r requirements.txt
  2. Edit the USER OPTIONS block below (stages, station IDs, POT parameters).
  3. Run from the module directory:
         python run_peaks_over_threshold.py
     ...or from the repository root:
         python modules/peaks_over_threshold/run_peaks_over_threshold.py

  CLI batch over explicit input files (no editing needed) - pass one or more
  time-series CSV paths (absolute or relative); POT-only runs on each:
         python run_peaks_over_threshold.py PATH1.csv PATH2.csv ...
         python run_peaks_over_threshold.py "C:\\data\\ntr_8518750.csv"
     Positional paths force STAGES=["pot"] and override the station chain for
     that run; the POT parameters from USER OPTIONS still apply. ``--help`` lists options.

The C++ threshold-search kernel (``_pot``) is compiled automatically on first
run; if no compiler is available it transparently falls back to pure Python.

Outputs: data files per station in data/outputs/<station>/
({dwl,ntr}_<station>_pot.csv); all plots in the shared data/outputs/plots/.
"""

from pathlib import Path

# Module root - every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# ── Stage selection ────────────────────────────────────────────────────────
# PRIMARY (default): ["pot"] POT only, on INPUT_CSV below.
# SECONDARY: build the POT input from NOAA data first by adding stages, in any
# combination, in canonical order:  "download", "detrend", "ntr", "pot".
#   Full chain example:  ["download", "detrend", "ntr", "pot"]
#   Already downloaded:  ["detrend", "ntr", "pot"]
#STAGES = ["download", "detrend", "ntr", "pot"]
STAGES = ["detrend", "ntr", "pot"]

# ── PRIMARY: POT on an existing time-series CSV ─────────────────────────────
# Used when STAGES do not build the input upstream. When the chain runs the
# ntr stage, POT instead consumes the freshly written ntr_<station>.csv.
INPUT_CSV    = DATA / "inputs" / "processed" / "ntr_8518750.csv"
DATETIME_COL = "Date Time"
VALUE_COL    = "NTR"
UNITS        = "m"
VDATUM       = ""

# POT extraction parameters (see METHOD & FORMULATION in the header for the
# algorithm these feed).
INTEREVENT_HOURS       = 48.0          # min separation between independent events (declustering window, h)
METHOD                 = "hydrograph"  # declustering rule: "hydrograph" (group + event max) or "peak_gap"
TARGET_EVENTS_PER_YEAR = 10.0          # average peaks/yr to retain - MUST match PST's EVENTS_PER_YEAR
TOLERANCE              = 0.25          # convergence band: accept a rate in [target, target + tolerance]
START_PERCENTILE       = 75.0          # series percentile where the upward threshold scan begins
STEP_SIZE              = 0.01          # percentile increment per scan step (smaller = finer + slower)

# ── SECONDARY: NOAA → NTR pipeline (only used by download/detrend/ntr) ──────
# One or more NOAA stations. The chain runs once per station, with per-station
# data folders derived automatically under data/ (raw, processed, outputs).
STATION_IDS    = ["8518750", "8651370", "8724580", "8761724", "8771450"]    # e.g. ["8518750", "8531680", "8534720"]
START_YEAR     = 1900
END_YEAR       = 2026
DATUM          = "MSL"          # NOAA vertical datum (MSL, MLLW, MHHW, …)
TIME_ZONE      = "GMT"          # "GMT" or "LST/LDT"
DOWNLOAD_UNITS = "metric"       # NOAA API units: "metric" or "english"
TIDE_INTERVAL  = "h"            # hourly tide predictions (match the WL grid)
DETREND_METHOD = "midpoint"     # "midpoint" (NTDE-centered) or "ordinary"

# National Tidal Datum Epoch (NTDE) - two ways to specify it:
#   1) ONE epoch for every station (even in batch): give a single year.
#          NTDE_START = 1983
#          NTDE_END   = 2001
#   2) PER-STATION epochs: give a list parallel to STATION_IDS (one per
#      station, same order). The list length MUST equal len(STATION_IDS) or the
#      run errors out. Use this when stations are tied to different epochs.
#          NTDE_START = [1983, 1983, 1983, 2012.42, 1983]   # one per station
#          NTDE_END   = [2001, 2001, 2001, 2016,    2001]
#   (Either field may be a single value or a list independently - e.g. a shared
#    end year with per-station start years.)
#NTDE_START     = 1983
#NTDE_END       = 2001
NTDE_START = [1983, 1983, 1983, 2012.42, 1983]
NTDE_END   = [2001, 2001, 2001, 2016,    2001]

# Sea-level slope override (value-units/yr, e.g. +0.0048 m/yr). By default the
# detrend stage FITS the slope from each record; set this to impose a known
# slope instead. Same scalar-or-per-station rule as NTDE:
#   None                       → fit the slope from data for every station.
#   0.0048                     → impose this slope for every station.
#   [None, None, 0.0048, ...]  → per-station list parallel to STATION_IDS;
#                                use None to fit that station, a number to
#                                override it. Length MUST equal len(STATION_IDS).
DETREND_SLOPE = None

# ── Data layout ─────────────────────────────────────────────────────────────
# Station chain: data files go per station to data/outputs/<station>/ and all
# plots to the shared data/outputs/plots/ (derived in main). The two below are
# used only by the PRIMARY POT-only (path) mode.
OUTPUT_DIR    = DATA / "outputs"
PLOTS_DIR     = DATA / "outputs" / "plots"

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================


# ── Launcher plumbing (CyHAN v2.1 §A.5 path anchoring; no user options) ─────
import os
import sys

# Guarantee headless rendering (no display needed) unless the operator overrides.
os.environ.setdefault("MPLBACKEND", "Agg")

_BACKEND_PY = ROOT / "backend" / "python"
_COMMON_PY  = ROOT.parents[1] / "common" / "python"   # shared CyHAN common library (§5.2)
for _p in (_BACKEND_PY, _COMMON_PY):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_cpp_extension() -> None:
    """Build the _pot C++ kernel once if it isn't already compiled.

    Must run before the package is imported (its solver probes for _pot at
    import time). A failed build is non-fatal - the pure-Python fallback runs.
    """
    pkg = _BACKEND_PY / "peaks_over_threshold"
    if any(p.suffix in (".pyd", ".so", ".dylib") for p in pkg.glob("_pot*")):
        return
    build = ROOT / "backend" / "engines" / "cpp" / "build.py"
    if not build.is_file():
        return
    print("[run] C++ kernel _pot not built - compiling once "
          "(falls back to pure Python if this fails) ...")
    import subprocess
    try:
        subprocess.run([sys.executable, str(build)], check=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[run] _pot build failed: {exc}. Using pure-Python fallback.")


CONFIG = {
    # stage selection
    "stages": STAGES,

    # POT stage
    "input_csv":              INPUT_CSV,
    "output_dir":             OUTPUT_DIR,
    "plots_dir":              PLOTS_DIR,
    "datetime_col":           DATETIME_COL,
    "value_col":              VALUE_COL,
    "units":                  UNITS,
    "vdatum":                 VDATUM,
    "interevent_hours":       INTEREVENT_HOURS,
    "method":                 METHOD,
    "target_events_per_year": TARGET_EVENTS_PER_YEAR,
    "tolerance":              TOLERANCE,
    "start_percentile":       START_PERCENTILE,
    "step_size":              STEP_SIZE,

    # preprocessing stages (download / detrend / ntr) - per-station dirs are
    # derived in main from data_dir; the chain runs once per station id.
    "station_ids":    STATION_IDS,
    "data_dir":       DATA,
    "start_year":     START_YEAR,
    "end_year":       END_YEAR,
    "datum":          DATUM,
    "time_zone":      TIME_ZONE,
    "download_units": DOWNLOAD_UNITS,
    "tide_interval":  TIDE_INTERVAL,
    "detrend_method": DETREND_METHOD,
    "ntde_start":     NTDE_START,
    "ntde_end":       NTDE_END,
    "detrend_slope":  DETREND_SLOPE,
}


def _apply_cli(config: dict) -> dict:
    """Apply CLI overrides for headless/batch runs (no file edits needed).

    Positional arguments are one or more input time-series CSV paths; when
    given, POT runs POT-only on each (batch), overriding STAGES/STATION_IDS.
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Run POT headless. With no arguments it uses the USER "
                    "OPTIONS above; pass input CSV paths to batch POT-only.")
    p.add_argument("inputs", nargs="*", type=Path,
                   help="One or more time-series CSV paths (absolute or "
                        "relative). Each is processed POT-only (batch).")
    args = p.parse_args()
    if args.inputs:
        config = dict(config)
        config["stages"]      = ["pot"]
        config["input_csvs"]  = [Path(x).expanduser().resolve() for x in args.inputs]
    return config


if __name__ == "__main__":
    _ensure_cpp_extension()   # build _pot on first run if needed
    # The orchestrator entry (main_peaks_over_threshold) lives in backend/python,
    # added to sys.path above at runtime. Resolve it dynamically so there is no
    # static import for the IDE to flag as unresolved.
    from importlib import import_module
    import_module("main_peaks_over_threshold").run(_apply_cli(CONFIG))
