"""run_peaks_over_threshold — POT launcher (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Peaks-Over-Threshold (POT) extraction. The operator
edits values in the USER OPTIONS block below and runs the script; no
orchestration logic lives here — the launcher delegates immediately to
``main_peaks_over_threshold.run`` per §5.3.

Usage
-----
    python run_peaks_over_threshold.py

CLI overrides are provided for ad-hoc runs but the file is intended as a
readable substitute for a long command-line invocation.

Input  : data/inputs/processed/<name>.csv         (columns DATETIME_COL, VALUE_COL)
Outputs: data/outputs/<name>_POT.csv               (selected peaks)
         data/outputs/plots/<name>_POT.png         (diagnostic plot)
"""

import argparse
import sys
from pathlib import Path


# ── Module-root path anchoring (CyHAN v2.0 §A.5) ──────────────────────────
ROOT = Path(__file__).resolve().parent       # run_<name>.py lives at module root
_BACKEND_PY = ROOT / "backend" / "python"
if str(_BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PY))


# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================

# ── Paths ────────────────────────────────────────────────────────────────
INPUT_CSV  = ROOT / "data" / "inputs" / "processed" / "storm_surge_8518750_1920_2025.csv"
OUTPUT_DIR = ROOT / "data" / "outputs"
PLOTS_DIR  = ROOT / "data" / "outputs" / "plots"

# ── Input columns ────────────────────────────────────────────────────────
DATETIME_COL = "Date Time"
VALUE_COL    = "Storm Surge"
UNITS        = "m"
VDATUM       = ""

# ── Event independence ──────────────────────────────────────────────────
INTEREVENT_HOURS = 48.0
METHOD           = "hydrograph"     # "hydrograph" or "peak_gap"

# ── Threshold-search target ─────────────────────────────────────────────
TARGET_EVENTS_PER_YEAR = 10.0
TOLERANCE              = 0.25
START_PERCENTILE       = 75.0
STEP_SIZE              = 0.01

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Peaks Over Threshold (POT) from a time-series CSV.",
    )
    p.add_argument("--input",         type=Path, help="Override input time-series CSV path")
    p.add_argument("--output-dir",    type=Path, help="Override output directory")
    p.add_argument("--plots-dir",     type=Path, help="Override plots directory")
    p.add_argument("--datetime-col",  type=str,  help="Datetime column name")
    p.add_argument("--value-col",     type=str,  help="Value column name")
    p.add_argument("--method",        type=str,  choices=["hydrograph", "peak_gap", "peaks"],
                   help="Event segmentation method")
    p.add_argument("--interevent-hours", type=float)
    p.add_argument("--target-events", type=float, dest="target_events_per_year")
    p.add_argument("--start-percentile", type=float)
    p.add_argument("--step-size",     type=float)
    p.add_argument("--tolerance",     type=float)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    config = {
        "input_csv":              args.input            or INPUT_CSV,
        "output_dir":             args.output_dir       or OUTPUT_DIR,
        "plots_dir":              args.plots_dir        or PLOTS_DIR,
        "datetime_col":           args.datetime_col     or DATETIME_COL,
        "value_col":              args.value_col        or VALUE_COL,
        "units":                  UNITS,
        "vdatum":                 VDATUM,
        "interevent_hours":       args.interevent_hours if args.interevent_hours is not None else INTEREVENT_HOURS,
        "method":                 args.method           or METHOD,
        "target_events_per_year": args.target_events_per_year if args.target_events_per_year is not None else TARGET_EVENTS_PER_YEAR,
        "tolerance":              args.tolerance        if args.tolerance        is not None else TOLERANCE,
        "start_percentile":       args.start_percentile if args.start_percentile is not None else START_PERCENTILE,
        "step_size":              args.step_size        if args.step_size        is not None else STEP_SIZE,
    }

    # Delegate to the orchestrator (§5.3: orchestration logic lives in main_<name>.py).
    from main_peaks_over_threshold import run
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
