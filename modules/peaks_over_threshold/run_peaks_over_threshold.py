"""run_peaks_over_threshold — POT launcher (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Peaks-Over-Threshold (POT) module. The operator edits
the USER OPTIONS block (at the top of this file) and runs the script. No
orchestration logic lives here — the launcher hands the option block to
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

Usage
-----
    python run_peaks_over_threshold.py
"""

from pathlib import Path

# Module root — every path in the options below is relative to this file.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================

# ── Stage selection ────────────────────────────────────────────────────────
# PRIMARY (default): ["pot"] POT only, on INPUT_CSV below.
# SECONDARY: build the POT input from NOAA data first by adding stages, in any
# combination, in canonical order:  "download", "detrend", "ntr", "pot".
#   Full chain example:  ["download", "detrend", "ntr", "pot"]
#   Already downloaded:  ["detrend", "ntr", "pot"]
STAGES = ["download", "detrend", "ntr", "pot"]

# ── PRIMARY: POT on an existing time-series CSV ─────────────────────────────
# Used when STAGES does not build the input upstream. When the chain runs the
# ntr stage, POT instead consumes the freshly written ntr_<station>.csv.
INPUT_CSV    = DATA / "inputs" / "processed" / "ntr_8518750.csv"
DATETIME_COL = "Date Time"
VALUE_COL    = "NTR"
UNITS        = "m"
VDATUM       = ""

# POT extraction parameters
INTEREVENT_HOURS       = 48.0
METHOD                 = "hydrograph"   # "hydrograph" or "peak_gap"
TARGET_EVENTS_PER_YEAR = 10.0
TOLERANCE              = 0.25
START_PERCENTILE       = 75.0
STEP_SIZE              = 0.01

# ── SECONDARY: NOAA → NTR pipeline (only used by download/detrend/ntr) ──────
STATION_ID     = "8518750"
START_YEAR     = 1900
END_YEAR       = 2025
DATUM          = "MSL"          # NOAA vertical datum (MSL, MLLW, MHHW, …)
TIME_ZONE      = "GMT"          # "GMT" or "LST/LDT"
DOWNLOAD_UNITS = "metric"       # NOAA API units: "metric" or "english"
TIDE_INTERVAL  = "h"            # hourly tide predictions (match the WL grid)
DETREND_METHOD = "midpoint"     # "midpoint" (NTDE-centered) or "ordinary"
NTDE_START     = 1983           # National Tidal Datum Epoch start year (e.g., 1983)
NTDE_END       = 2001           # National Tidal Datum Epoch end year (e.g., 2001)

# ── Data layout ─────────────────────────────────────────────────────────────
# Raw NOAA data is kept per-gauge; the detrend/ntr chain writes processed data;
# POT writes peaks + plots. Edit if your study area uses a different layout.
RAW_DIR       = DATA / "inputs" / "raw"       / STATION_ID
PROCESSED_DIR = DATA / "inputs" / "processed" / STATION_ID
OUTPUT_DIR    = DATA / "outputs" / STATION_ID
PLOTS_DIR     = OUTPUT_DIR / "plots"

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================


# ── Launcher plumbing (CyHAN v2.0 §A.5 path anchoring; no user options) ─────
import sys

_BACKEND_PY = ROOT / "backend" / "python"
if str(_BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PY))

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

    # preprocessing stages (download / detrend / ntr)
    "station_id":     STATION_ID,
    "raw_dir":        RAW_DIR,
    "processed_dir":  PROCESSED_DIR,
    "start_year":     START_YEAR,
    "end_year":       END_YEAR,
    "datum":          DATUM,
    "time_zone":      TIME_ZONE,
    "download_units": DOWNLOAD_UNITS,
    "tide_interval":  TIDE_INTERVAL,
    "detrend_method": DETREND_METHOD,
    "ntde_start":     NTDE_START,
    "ntde_end":       NTDE_END,
}


if __name__ == "__main__":
    # The orchestrator entry (main_peaks_over_threshold) lives in backend/python,
    # added to sys.path above at runtime. Resolve it dynamically so there is no
    # static import for the IDE to flag as unresolved.
    from importlib import import_module
    import_module("main_peaks_over_threshold").run(CONFIG)
