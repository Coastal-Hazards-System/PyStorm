"""YAML / JSON configuration loader for the reduced_tc_suite module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_config(config_path: str | Path) -> dict:
    """Load a YAML (.yaml / .yml) or JSON (.json) configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        content = fh.read()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError(
                "pyyaml required for YAML configs.  pip install pyyaml")
        return yaml.safe_load(content)
    if suffix == ".json":
        return json.loads(content)
    raise ValueError(f"Unsupported config format '{suffix}'. Use .yaml or .json.")


PREPROCESS_CONFIG_YAML = """\
# reduced_tc_suite preprocessor - Example Configuration
# =====================================================
# Edit paths and variable names for your data, then run:
#   python scripts/preprocess.py --config this_file.yaml

output_path: "tc_data.h5"

# X - TC atmospheric parameters
X_source:       "X_parameters.mat"   # .mat / .csv / .npy / .npz / .txt / .h5
X_variable:     "X"                  # variable/dataset name (required for .mat, .npz, .h5)
X_param_names:                       # null = use CSV headers or auto X0..Xp
  - lat_ref
  - lon_ref
  - heading
  - delta_p
  - rmw
  - fts
X_storm_id_col: null                 # CSV column name for storm IDs, or null
X_columns:      null                 # list of column names/indices; null = all numeric
X_transpose:    false                # true if raw array is [p_params x n_storms]

# Per-Y-row storm IDs. When X carries no IDs (X_storm_id_col null) and
# storm_id_track_dir is set, IDs are derived from the TROP filenames: Y row i
# is the i-th storm in ascending ID order. Required for SACS subsets (gom/sa)
# whose master IDs are non-contiguous; yields 1..N for contiguous suites.
# Skipped if the folder is absent/empty; raises if file count != n_storms.
storm_id_track_dir:     null         # folder of per-storm TROP files, or null
storm_id_track_pattern: "LACPR2_JPM{:04d}_TROP.txt"   # {:04d} ← storm ID

# Y - ADCIRC peak surge fields
Y_source:       "Y_surges.mat"
Y_variable:     "eta_max"
Y_node_ids:     null
Y_units:        "m NAVD88"
Y_transpose:    false                # true if raw array is [m_nodes x n_storms]

# HC - Benchmark hazard curves  (set HC_source to null to omit /HC entirely)
HC_source:      "HC_benchmark.mat"
HC_variable:    "HC_bench"
HC_units:       "m NAVD88"
HC_transpose:   false                # true if raw array is [N_AER x m_nodes]
HC_aer_levels:  null                 # null = default 22-level table

validate: true
"""
