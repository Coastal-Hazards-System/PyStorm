"""
config/loader.py
=================
Load YAML or JSON configuration files.

Cross-cutting: imported by both the API layer and orchestration workflows.

Public API
----------
  load_config(path)  -> dict
  PREPROCESS_CONFIG_YAML  (str)  — annotated template, printed by --generate-config
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
    """
    Load a YAML (.yaml / .yml) or JSON (.json) configuration file.

    Raises FileNotFoundError, ImportError, or ValueError as appropriate.
    """
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


# Annotated example YAML — printed by  scripts/preprocess.py --generate-config
PREPROCESS_CONFIG_YAML = """\
# tc_preprocess — Example Configuration
# ======================================
# Edit paths and variable names for your data, then run:
#   python scripts/preprocess.py --config this_file.yaml

output_path: "tc_data.h5"

# X — TC atmospheric parameters
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

# Y — ADCIRC peak surge fields
Y_source:       "Y_surges.mat"
Y_variable:     "eta_max"
Y_node_ids:     null
Y_units:        "m NAVD88"
Y_transpose:    false                # true if raw array is [m_nodes x n_storms]

# HC — Benchmark hazard curves  (set HC_source to null to omit /HC entirely)
HC_source:      "HC_benchmark.mat"
HC_variable:    "HC_bench"
HC_units:       "m NAVD88"
HC_transpose:   false                # true if raw array is [N_AER x m_nodes]
HC_aer_levels:  null                 # null = default 22-level table
# HC_aer_levels:
#   - 10.0
#   - 5.0
#   - 2.0
#   - 1.0
#   - 0.5
#   - 0.2
#   - 0.1

validate: true
"""
