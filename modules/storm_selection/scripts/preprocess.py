"""storm_selection — data ingestion launcher.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Ingest raw source files (.mat / .csv / .h5 / .npy / .npz / .txt) and write
the canonical tc_data.h5 store consumed by run_storm_selection.py.

Usage
-----
  1. Edit the USER OPTIONS block below.
  2. Run:  python scripts/preprocess.py

Optional flags (no CONFIG editing required):
  python scripts/preprocess.py --generate-config
      Print an annotated YAML config template and exit.

  python scripts/preprocess.py --validate data/processed/tc_data.h5
      Validate an existing store and exit.

  python scripts/preprocess.py --export-csv data/processed/tc_data.h5 [--export-dir DIR]
      Export /X /Y /HC to CSV files and exit.
"""

import argparse
import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))


# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================

# ---------------------------------------------------------------------------
# CONFIG  — paths to raw source files and HDF5 output
# ---------------------------------------------------------------------------
CONFIG = {
    "output_path": str(_MODULE_ROOT / "data/processed/tc_data.h5"),

    # X — TC atmospheric parameters
    "X_source":       str(_MODULE_ROOT / "data/raw/lpv/CHS-LA_ITCS_Param_MasterTable.mat"),
    "X_variable":     "Param_MT",
    "X_param_names":  None,
    "X_storm_id_col": None,
    "X_columns":      None,
    "X_transpose":    False,

    # Y — ADCIRC peak surge fields
    "Y_source":    str(_MODULE_ROOT / "data/raw/lpv/CHS-LA_SSL_TC_slc0_tide0_wave1_m_NAVD88.mat"),
    "Y_variable":  "Resp",
    "Y_node_ids":  None,
    "Y_units":     "m NAVD88",
    "Y_transpose": True,

    # Node filter — subset Y columns to kept nodes
    "Y_node_filter_source":   str(_MODULE_ROOT / "data/raw/lpv/CHS-LA_nodeID_probQ.mat"),
    "Y_node_filter_variable": "nodeID",
    "Y_node_filter_col":      1,
    "Y_node_id_col":          0,

    # HC — Benchmark hazard curves
    "HC_source":     str(_MODULE_ROOT / "data/raw/lpv/CHS-LA-24_HC_tbl_TC_SSL_slc0_BE_22.mat"),
    "HC_variable":   "BE_22",
    "HC_units":      "m NAVD88",
    "HC_transpose":  False,
    "HC_aer_levels": None,

    "validate": True,
}

# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================


def main():
    from storm_selection.io.store import export_to_csv
    from storm_selection.workflows.ingest import Preprocessor
    from storm_selection.config.loader import PREPROCESS_CONFIG_YAML

    parser = argparse.ArgumentParser(prog="preprocess.py")
    parser.add_argument("--generate-config", action="store_true")
    parser.add_argument("--validate",   metavar="H5_FILE")
    parser.add_argument("--export-csv", metavar="H5_FILE")
    parser.add_argument("--export-dir", metavar="DIR", default=None)
    args = parser.parse_args()

    if args.generate_config:
        print(PREPROCESS_CONFIG_YAML); return

    if args.validate:
        Preprocessor.validate(Path(args.validate)); return

    if args.export_csv:
        export_to_csv(Path(args.export_csv),
                      Path(args.export_dir) if args.export_dir else None)
        return

    Preprocessor(CONFIG).run()


if __name__ == "__main__":
    main()
