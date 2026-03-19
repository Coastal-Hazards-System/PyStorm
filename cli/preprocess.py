"""
cli/preprocess.py
==================
Thin launcher — ingest raw source files and write the standard tc_data.h5 store.

Usage
-----
  1. Set the paths in CONFIG below.
  2. Run:  python cli/preprocess.py

All raw source files live under  data/raw/.
The output tc_data.h5 is written to  data/processed/.

Optional flags (no CONFIG editing required):
  python cli/preprocess.py --generate-config
      Print an annotated YAML config template and exit.

  python cli/preprocess.py --validate data/processed/tc_data.h5
      Validate an existing store and exit.

  python cli/preprocess.py --export-csv data/processed/tc_data.h5 [--export-dir DIR]
      Export /X /Y /HC to CSV files and exit.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------------
CONFIG = {
    "output_path": str(_ROOT / "data/processed/tc_data.h5"),

    # X — TC atmospheric parameters
    "X_source":       str(_ROOT / "data/raw/lpv/CHS-LA_ITCS_Param_MasterTable.mat"),
    "X_variable":     "Param_MT",      # shape (645, 10) = [n_storms x p_params]
    "X_param_names":  None,
    "X_storm_id_col": None,
    "X_columns":      None,
    "X_transpose":    False,

    # Y — ADCIRC peak surge fields
    "Y_source":    str(_ROOT / "data/raw/lpv/CHS-LA_SSL_TC_slc0_tide0_wave1_m_NAVD88.mat"),
    "Y_variable":  "Resp",             # MATLAB shape [m_nodes x n_storms]
    "Y_node_ids":  None,
    "Y_units":     "m NAVD88",
    "Y_transpose": True,               # restore MATLAB order → [n_storms x m_nodes]

    # Node filter — subset Y columns to kept nodes
    "Y_node_filter_source":   str(_ROOT / "data/raw/lpv/CHS-LA_nodeID_probQ.mat"),
    "Y_node_filter_variable": "nodeID",
    "Y_node_filter_col":      1,       # column index in the (n_kept, k) array

    # HC — Benchmark hazard curves
    "HC_source":     str(_ROOT / "data/raw/lpv/CHS-LA-24_HC_tbl_TC_SSL_slc0_BE_22.mat"),
    "HC_variable":   "BE_22",
    "HC_units":      "m NAVD88",
    "HC_transpose":  False,
    "HC_aer_levels": None,

    "validate": True,
}
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

from backend.io.store import export_to_csv
from backend.orch.python.workflows.ingest import Preprocessor
from config.loader import PREPROCESS_CONFIG_YAML


def main():
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
