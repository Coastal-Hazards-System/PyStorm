"""Ingestion workflow: load raw source files → validate shapes → write tc_data.h5.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Owns the sequence; delegates format I/O to reduced_tc_suite.io.readers and
store writes to reduced_tc_suite.io.store.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from reduced_tc_suite.io.readers import load_array
from reduced_tc_suite.io.store import write_store, validate_store


_DEFAULT_AER: list = (1.0 / np.array([
    0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000,
    2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
    1_000_000,
])).tolist()


class Preprocessor:
    """Orchestrate ingestion of X, Y, HC source files and write tc_data.h5.

    config keys: see config/templates/preprocess.yaml for documentation.
    """

    def __init__(self, config: dict):
        self.cfg = config

    # -------------------------------------------------------------------------
    # Private loaders
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

    def _resolve_storm_ids(self, storm_ids: list, n_storms: int) -> list:
        """Resolve per-Y-row storm IDs, deriving from the track folder when the
        X source carries none.

        Priority:
          1. IDs already read from the X source (e.g. a CSV id column) — kept.
          2. Else, if ``storm_id_track_dir`` is configured, derive integer IDs
             from the TROP filenames. By dataset convention Y row i is the i-th
             storm in ascending ID order, so the sorted filename IDs ARE the
             per-row IDs. This is what gives the SACS subsets (gom/sa) their
             true master IDs; for a contiguous suite it yields 1..N.
          3. Else — left as-is (empty/None); row order is purely positional.

        Deriving requires exactly one TROP file per storm: a non-zero file
        count that differs from ``n_storms`` raises (it would silently
        misalign), while a missing folder or zero matches is tolerated (so a
        regional-only dataset can be preprocessed before tracks are staged).
        """
        if storm_ids:
            return storm_ids

        track_dir = self.cfg.get("storm_id_track_dir")
        if not track_dir:
            return storm_ids

        track_dir = Path(track_dir)
        if not track_dir.is_dir():
            print(f"       storm_id_track_dir absent ({track_dir}); "
                  "storm_ids left empty (positional row order).")
            return storm_ids

        # Lazy import: keep the geo stack out of the ingest import path.
        from reduced_tc_suite.geo.bbox_filter import storm_ids_from_track_dir
        pattern = self.cfg.get("storm_id_track_pattern",
                               "LACPR2_JPM{:04d}_TROP.txt")
        ids = storm_ids_from_track_dir(track_dir, pattern)

        if not ids:
            print(f"       No TROP files matching '{pattern}' in {track_dir}; "
                  "storm_ids left empty (positional row order).")
            return storm_ids

        if len(ids) != n_storms:
            raise ValueError(
                f"storm_id derivation: {len(ids)} TROP files match '{pattern}' "
                f"in {track_dir}, but X/Y have {n_storms} storms. The track "
                "folder must hold exactly one file per storm (ascending ID "
                "order == Y-row order). Fix the track folder or unset "
                "storm_id_track_dir.")

        contiguous = ids == list(range(1, n_storms + 1))
        kind = "contiguous 1..N" if contiguous else "non-contiguous subset"
        print(f"       Derived {len(ids)} storm IDs from track filenames "
              f"({kind}, IDs {ids[0]}..{ids[-1]}).")
        return [str(i) for i in ids]

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

    def _load_node_filter(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load 0-based column indices for subsetting Y nodes, plus main node IDs."""
        cfg = self.cfg
        src = cfg.get("Y_node_filter_source")
        if not src:
            return None, None

        path = Path(src)
        print(f"\n  [NF] Loading node filter: {path}")
        arr, _, _ = load_array(path, varname=cfg.get("Y_node_filter_variable"))

        filter_col = int(cfg.get("Y_node_filter_col", 1))
        adcirc_ids = arr[:, filter_col].astype(np.int64)
        indices    = adcirc_ids - 1

        id_col   = int(cfg.get("Y_node_id_col", 0))
        main_ids = arr[:, id_col].astype(np.int64)

        sort_order = np.argsort(indices)
        indices  = indices[sort_order]
        main_ids = main_ids[sort_order]

        print(f"       Keeping {len(indices):,} nodes  "
              f"(ADCIRC IDs {adcirc_ids.min():,} – {adcirc_ids.max():,}, "
              f"main IDs {main_ids.min():,} – {main_ids.max():,})")
        return indices, main_ids

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
        storm_ids = self._resolve_storm_ids(storm_ids, n_storms)

        Y_arr, node_ids, Y_units, Y_src = self._load_Y(n_storms)

        filter_indices, main_node_ids = self._load_node_filter()
        if filter_indices is not None:
            Y_arr    = Y_arr[:, filter_indices]
            node_ids = [str(i) for i in main_node_ids.tolist()]
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
