"""
backend/orch/workflows/ingest.py
==================================
Ingestion workflow: load raw source files → validate shapes → write tc_data.h5.

Owns the sequence; delegates format I/O to backend.io.readers and
store writes to backend.io.store.  The Preprocessor class is the verbatim
logic of tc_preprocess.py, re-homed here.

Public API
----------
  Preprocessor(config).run()         -> Path
  Preprocessor.validate(path)        -> None   (static)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from backend.io.readers import load_array
from backend.io.store import write_store, validate_store


_DEFAULT_AER: list = (1.0 / np.array([
    0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000,
    2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
    1_000_000,
])).tolist()


class Preprocessor:
    """
    Orchestrate ingestion of X, Y, HC source files and write tc_data.h5.

    config keys: see config/templates/preprocess.yaml for documentation.
    """

    def __init__(self, config: dict):
        self.cfg = config

    # -------------------------------------------------------------------------
    # Private loaders  (verbatim from tc_preprocess.py Preprocessor)
    # -------------------------------------------------------------------------

    def _load_X(self) -> tuple[np.ndarray, list, list, str]:
        cfg  = self.cfg
        path = Path(cfg["X_source"])
        print(f"\n  [X]  Loading: {path}")

        arr, col_names, ids = load_array(
            path,
            varname=cfg.get("X_variable"),
            columns=cfg.get("X_columns"),
            id_col=cfg.get("X_storm_id_col"),
        )
        if cfg.get("X_transpose", False):
            print("       Transposing X  (X_transpose=true)")
            arr = arr.T

        n, p = arr.shape
        print(f"       Shape: {n} storms x {p} params  (float64)")

        pnames = cfg.get("X_param_names") or col_names or [f"X{i}" for i in range(p)]
        if len(pnames) != p:
            warnings.warn(
                f"X_param_names has {len(pnames)} entries but p={p}. "
                "Names will be padded/trimmed with generic labels.", UserWarning)
            pnames = (list(pnames) + [f"X{i}" for i in range(p)])[:p]

        return arr.astype(np.float64), list(pnames), ids, str(path)

    def _load_Y(self, n_expected: int) -> tuple[np.ndarray, list, str, str]:
        cfg  = self.cfg
        path = Path(cfg["Y_source"])
        print(f"\n  [Y]  Loading: {path}")

        arr, _, _ = load_array(path, varname=cfg.get("Y_variable"))
        if cfg.get("Y_transpose", False):
            print("       Transposing Y  (Y_transpose=true)")
            arr = arr.T

        n, m = arr.shape
        if n != n_expected:
            raise ValueError(
                f"Y has {n} storms (rows) but X has {n_expected}. "
                "Row counts must match exactly.")
        print(f"       Shape: {n} storms x {m} nodes  (stored as float32)")

        node_ids = cfg.get("Y_node_ids") or []
        units    = cfg.get("Y_units", "m NAVD88")
        return arr.astype(np.float32), node_ids, units, str(path)

    def _load_node_filter(self) -> Optional[np.ndarray]:
        """
        Load 0-based column indices for subsetting Y nodes.

        Config keys:
          Y_node_filter_source   : path to .mat/.npy/etc. file
          Y_node_filter_variable : variable name in the file (required for .mat)
          Y_node_filter_col      : which column (0-based) in the 2-D array holds
                                   the 1-based ADCIRC node IDs (default 1)

        Returns a sorted int64 array of 0-based indices, or None if not configured.
        """
        cfg = self.cfg
        src = cfg.get("Y_node_filter_source")
        if not src:
            return None

        path = Path(src)
        print(f"\n  [NF] Loading node filter: {path}")
        arr, _, _ = load_array(path, varname=cfg.get("Y_node_filter_variable"))
        col = int(cfg.get("Y_node_filter_col", 1))
        ids_1based = arr[:, col].astype(np.int64)   # 1-indexed ADCIRC node IDs
        indices    = ids_1based - 1                  # convert to 0-based
        print(f"       Keeping {len(indices):,} nodes  "
              f"(node IDs {ids_1based.min():,} – {ids_1based.max():,})")
        return np.sort(indices)

    def _load_HC(
        self, m_expected: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
        cfg = self.cfg
        if not cfg.get("HC_source"):
            print("\n  [HC] HC_source not set — /HC group will be omitted.")
            return None, None, "m NAVD88", ""

        path = Path(cfg["HC_source"])
        print(f"\n  [HC] Loading: {path}")

        arr, _, _ = load_array(path, varname=cfg.get("HC_variable"))
        if cfg.get("HC_transpose", False):
            print("       Transposing HC  (HC_transpose=true)")
            arr = arr.T

        m, n_aer = arr.shape
        if m != m_expected:
            raise ValueError(
                f"HC has {m} rows (nodes) but Y has {m_expected} columns. "
                "HC row count must equal the number of mesh nodes in Y.")

        aer = cfg.get("HC_aer_levels")
        if aer is None:
            aer = _DEFAULT_AER
            print(f"       HC_aer_levels not set — using default {len(aer)}-level table.")
        aer = np.asarray(aer, dtype=np.float64)

        if len(aer) != n_aer:
            raise ValueError(
                f"HC has {n_aer} columns but HC_aer_levels has {len(aer)} entries.")
        print(f"       Shape: {m} nodes x {n_aer} AER levels  (float64)")

        units = cfg.get("HC_units", "m NAVD88")
        return arr.astype(np.float64), aer, units, str(path)

    # -------------------------------------------------------------------------
    # Validation (static)
    # -------------------------------------------------------------------------

    @staticmethod
    def validate(out_path: Path) -> None:
        out_path = Path(out_path)
        print(f"\n  Validating {out_path.name} ...")
        validate_store(out_path)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self) -> Path:
        out_path = Path(self.cfg.get("output_path", "tc_data.h5"))

        print(f"\n{'='*60}")
        print(f"  TC Data Pre-Processor")
        print(f"  Output: {out_path}")
        print(f"{'='*60}")

        X_arr, pnames, storm_ids, X_src = self._load_X()
        n_storms, p_params = X_arr.shape

        Y_arr, node_ids, Y_units, Y_src = self._load_Y(n_storms)

        node_filter = self._load_node_filter()
        if node_filter is not None:
            Y_arr    = Y_arr[:, node_filter]
            node_ids = [str(i) for i in (node_filter + 1).tolist()]  # store 1-based IDs as strings
            print(f"       Y after filter: {Y_arr.shape}")

        _, m_nodes = Y_arr.shape

        HC_arr, aer, HC_units, HC_src = self._load_HC(m_nodes)

        print(f"\n  Writing {out_path} ...")
        write_store(
            path=out_path,
            X=X_arr, Y=Y_arr,
            param_names=pnames,
            storm_ids=storm_ids if storm_ids else None,
            node_ids=node_ids   if node_ids  else None,
            Y_units=Y_units, Y_source=Y_src,
            X_source=X_src,
            HC=HC_arr, aer_levels=aer,
            HC_units=HC_units, HC_source=HC_src,
        )

        size_mb = out_path.stat().st_size / 1024 ** 2
        print(f"\n  File size : {size_mb:.2f} MB")
        print(f"  /X        : {X_arr.shape}   params={pnames}")
        print(f"  /Y        : {Y_arr.shape}")
        if HC_arr is not None:
            print(f"  /HC       : {HC_arr.shape}   {len(aer)} AER levels")

        if self.cfg.get("validate", True):
            self.validate(out_path)

        print(f"\n  Done.  Standard input file ready: {out_path}\n")
        return out_path
