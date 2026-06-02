"""run_probabilistic_simulation_technique — PST launcher (CyHAN v2.0 §5.3).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

User-facing entry for the Probabilistic Simulation Technique (PST). The
operator edits values in the USER OPTIONS block below and runs the script;
no orchestration logic lives here — the launcher delegates immediately to
``main_probabilistic_simulation_technique.run`` per §5.3.

Usage
-----
    python run_probabilistic_simulation_technique.py

CLI overrides are provided for ad-hoc runs but the file is intended as a
readable substitute for a long command-line invocation: edit the USER OPTIONS
block once for your study, then run.

Input  : data/inputs/processed/<base>_POT.csv  (column = STORM_COLUMN)
Outputs: data/outputs/<base>_PST*.csv
         data/outputs/plots/<base>_PST_HC.png
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
INPUT_CSV  = ROOT / "data" / "inputs" / "processed" / "storm_surge_8518750_1920_2025_POT.csv"
OUTPUT_DIR = ROOT / "data" / "outputs"
PLOTS_DIR  = ROOT / "data" / "outputs" / "plots"

# ── POT extraction ───────────────────────────────────────────────────────
STORM_COLUMN        = "value"
RECORD_LENGTH_YEARS = 106.0       # e.g. 2025 − 1920 + 1

# ── Monte Carlo / bootstrap ──────────────────────────────────────────────
NUM_SIMULATIONS = 1000
RANDOM_SEED     = 628

BOOTSTRAP_DISTRIBUTION = "gaussian"    # "gaussian" or "uniform"
BOOTSTRAP_TRUNCATION   = (-1.0, 1.0)

# ── Plotting ─────────────────────────────────────────────────────────────
Y_AXIS_LABEL = "Storm Surge Level (SSL, m)"

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Probabilistic Simulation Technique (PST).",
    )
    p.add_argument("--input",           type=Path, help="Override input POT CSV path")
    p.add_argument("--output-dir",      type=Path, help="Override output directory")
    p.add_argument("--plots-dir",       type=Path, help="Override plots directory")
    p.add_argument("--storm-column",    type=str,  help="POT column name")
    p.add_argument("--record-length",   type=float, dest="record_length_years",
                   help="Record length in years")
    p.add_argument("--num-simulations", type=int,
                   help="Number of bootstrap realizations")
    p.add_argument("--seed",            type=int, dest="random_seed",
                   help="RNG seed")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Compose operator-edited defaults with CLI overrides.
    config = {
        "input_csv":           args.input               or INPUT_CSV,
        "output_dir":          args.output_dir          or OUTPUT_DIR,
        "plots_dir":           args.plots_dir           or PLOTS_DIR,
        "storm_column":        args.storm_column        or STORM_COLUMN,
        "record_length_years": args.record_length_years if args.record_length_years is not None else RECORD_LENGTH_YEARS,
        "num_simulations":     args.num_simulations     if args.num_simulations     is not None else NUM_SIMULATIONS,
        "random_seed":         args.random_seed         if args.random_seed         is not None else RANDOM_SEED,
        "y_axis_label":        Y_AXIS_LABEL,
        "bootstrap": {
            "distribution": BOOTSTRAP_DISTRIBUTION,
            "truncation":   BOOTSTRAP_TRUNCATION,
        },
    }

    # Delegate to the orchestrator (§5.3: orchestration logic lives in main_<name>.py).
    from main_probabilistic_simulation_technique import run
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
