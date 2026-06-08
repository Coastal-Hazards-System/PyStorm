"""Read, write, validate, and CSV-export the standard reduced_tc_suite HDF5 store.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Standard layout (tc_data.h5)
-----------------------------
  /X   float64  [n_storms x p_params]
       attrs: param_names, storm_ids, n_storms, p_params, source_file

  /Y   float32  [n_storms x m_nodes]
       attrs: node_ids, storm_ids, n_storms, m_nodes, units, source_file

  /HC  float64  [m_nodes x N_AER]          (optional)
       attrs: aer_levels, m_nodes, N_AER, units, source_file
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class StoreData(NamedTuple):
    X:           np.ndarray            # float64  [n x p]
    Y:           np.ndarray            # float64  [n x m]  (cast up from float32)
    HC:          Optional[np.ndarray]  # float64  [m x N_AER] or None
    aer_levels:  Optional[np.ndarray]  # float64  [N_AER] or None
    param_names: list                  # list[str]
    storm_ids:   Optional[list]        # list[str] or None
    node_ids:    list                  # list[str]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_store(
    path:        Path,
    X:           np.ndarray,
    Y:           np.ndarray,
    param_names: list,
    storm_ids:   Optional[list]       = None,
    node_ids:    Optional[list]       = None,
    Y_units:     str                  = "m NAVD88",
    Y_source:    str                  = "",
    X_source:    str                  = "",
    HC:          Optional[np.ndarray] = None,
    aer_levels:  Optional[np.ndarray] = None,
    HC_units:    str                  = "m NAVD88",
    HC_source:   str                  = "",
) -> None:
    """Write X, Y, and optionally HC to the standard HDF5 store."""
    if not _HAS_H5PY:
        raise ImportError("h5py required to write the store.  pip install h5py")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_storms, p_params = X.shape
    _, m_nodes         = Y.shape

    str_dt = h5py.string_dtype(encoding="utf-8")

    def _str_dataset(fh, name, strings):
        encoded = [s.encode("utf-8") if isinstance(s, str) else str(s).encode("utf-8")
                   for s in (strings or [])]
        fh.create_dataset(name, data=encoded or np.empty(0, dtype=str_dt),
                          dtype=str_dt, shape=(len(encoded),))

    with h5py.File(path, "w") as fh:
        fh.attrs["description"]    = "reduced_tc_suite standard input store"
        fh.attrs["format_version"] = "1.0"

        _str_dataset(fh, "storm_ids", storm_ids)
        _str_dataset(fh, "node_ids",  node_ids)

        ds = fh.create_dataset("X", data=X.astype(np.float64),
                               compression="gzip", compression_opts=4, chunks=True)
        ds.attrs["description"] = "TC parameter matrix  [n_storms x p_params]"
        ds.attrs["n_storms"]    = n_storms
        ds.attrs["p_params"]    = p_params
        ds.attrs["param_names"] = [s.encode("utf-8") for s in param_names]
        ds.attrs["source_file"] = X_source

        ds = fh.create_dataset("Y", data=Y.astype(np.float32),
                               compression="gzip", compression_opts=4, chunks=True)
        ds.attrs["description"] = "ADCIRC peak surge fields  [n_storms x m_nodes]"
        ds.attrs["n_storms"]    = n_storms
        ds.attrs["m_nodes"]     = m_nodes
        ds.attrs["units"]       = Y_units
        ds.attrs["source_file"] = Y_source

        if HC is not None:
            if aer_levels is None:
                raise ValueError("aer_levels required when writing HC.")
            ds = fh.create_dataset("HC", data=HC.astype(np.float64),
                                   compression="gzip", compression_opts=4, chunks=True)
            ds.attrs["description"] = "Benchmark hazard curves  [m_nodes x N_AER]"
            ds.attrs["m_nodes"]     = m_nodes
            ds.attrs["N_AER"]       = len(aer_levels)
            ds.attrs["aer_levels"]  = np.asarray(aer_levels, dtype=np.float64)
            ds.attrs["units"]       = HC_units
            ds.attrs["source_file"] = HC_source


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_store(path: Path) -> StoreData:
    """Load all datasets from the standard HDF5 store."""
    if not _HAS_H5PY:
        raise ImportError("h5py required to read the store.  pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HDF5 store not found: {path}\n"
            "Run scripts/preprocess.py to generate it, "
            "or set h5_path = None in CONFIG to use the CSV fallback."
        )

    with h5py.File(path, "r") as fh:
        if "X" not in fh or "Y" not in fh:
            raise KeyError(f"Store {path.name} is missing /X or /Y.")

        X = fh["X"][()].astype(np.float64)
        Y = fh["Y"][()].astype(np.float64)

        # Y carries two distinct missing markers with different meanings:
        #   -99999.0  node WAS simulated but stayed dry  (valid datum)
        #   NaN       node was NOT simulated for this storm (no calculation)
        # A storm whose ENTIRE row is NaN was never simulated anywhere — a
        # failed HPC run. Capture that mask NOW, on the raw values, before the
        # -99999 -> NaN normalisation below; otherwise an all-dry (-99999)
        # storm would be normalised to all-NaN and wrongly flagged as failed.
        failed_storm = np.isnan(Y).all(axis=1)

        # ADCIRC stores dry/inactive nodes as -99999.0; normalise to NaN so
        # downstream code uses a uniform missing-data marker instead of
        # relying on implicit "< 0" masks. Threshold of -9000 matches the
        # convention documented in data/inputs/processed/README.md.
        sentinel_mask = Y < -9000
        n_sentinel = int(sentinel_mask.sum())
        if n_sentinel:
            Y[sentinel_mask] = np.nan
            pct = 100.0 * n_sentinel / Y.size
            print(f"    Y: cleaned {n_sentinel:,} -99999 sentinels → NaN  ({pct:.2f}%)")

        try:
            raw = fh["X"].attrs["param_names"]
            param_names = [s.decode() if isinstance(s, bytes) else s for s in raw]
        except Exception:
            param_names = [f"X{i}" for i in range(X.shape[1])]

        try:
            raw = fh["storm_ids"][()]
            storm_ids = [s.decode() if isinstance(s, bytes) else s for s in raw] or None
        except Exception:
            storm_ids = None

        try:
            raw = fh["node_ids"][()]
            node_ids = [s.decode() if isinstance(s, bytes) else s for s in raw]
        except Exception:
            node_ids = []

        HC = None; aer_levels = None
        if "HC" in fh:
            HC = fh["HC"][()].astype(np.float64)
            try:
                aer_levels = fh["HC"].attrs["aer_levels"].astype(np.float64)
            except Exception:
                aer_levels = None

    # Drop the failed-HPC storms detected above (all-NaN rows = never
    # simulated; all-dry -99999 storms are valid and are NOT dropped). Done
    # after loading storm_ids so X / Y / storm_ids drop in lockstep and every
    # consumer (both read_store call sites) sees the same survivors. HC is
    # per-node, not per-storm, so it is unaffected.
    n_failed = int(failed_storm.sum())
    if n_failed:
        keep = ~failed_storm
        failed_pos = np.where(failed_storm)[0]
        if storm_ids is not None and len(storm_ids) == len(failed_storm):
            dropped = [str(storm_ids[i]) for i in failed_pos]
            storm_ids = [storm_ids[i] for i in np.where(keep)[0]]
            label = "IDs"
        else:
            dropped = [str(int(i)) for i in failed_pos]
            label = "rows"
        shown = ", ".join(dropped[:40]) + (" ..." if len(dropped) > 40 else "")
        X = X[keep]
        Y = Y[keep]
        print(f"    Y: dropped {n_failed:,} not-simulated storm(s) -- failed HPC "
              f"runs, all-NaN rows ({label} {shown}); {int(keep.sum()):,} kept")

    return StoreData(X=X, Y=Y, HC=HC, aer_levels=aer_levels,
                     param_names=param_names, storm_ids=storm_ids, node_ids=node_ids)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_store(path: Path) -> None:
    """Assert shape and attribute consistency. Raises AssertionError on failure."""
    if not _HAS_H5PY:
        raise ImportError("h5py required for store validation.")

    path = Path(path)
    with h5py.File(path, "r") as fh:
        assert "X" in fh and "Y" in fh, f"Missing /X or /Y in {path.name}"

        nX = int(fh["X"].attrs["n_storms"]); pX = int(fh["X"].attrs["p_params"])
        nY = int(fh["Y"].attrs["n_storms"]); mY = int(fh["Y"].attrs["m_nodes"])

        assert nX == nY, f"/X n_storms={nX} != /Y n_storms={nY}"
        assert fh["X"].shape == (nX, pX), f"/X shape mismatch"
        assert fh["Y"].shape == (nY, mY), f"/Y shape mismatch"
        print(f"       /X  : {fh['X'].shape}  float64  OK")
        print(f"       /Y  : {fh['Y'].shape}  float32  OK")

        if "storm_ids" in fh and fh["storm_ids"].shape[0] > 0:
            n_sid = int(fh["storm_ids"].shape[0])
            assert n_sid == nX, (
                f"/storm_ids has {n_sid} entries != n_storms={nX}; the per-row "
                "storm-ID map is misaligned with X/Y.")
            print(f"       /storm_ids : {n_sid}  OK")

        if "HC" in fh:
            mHC  = int(fh["HC"].attrs["m_nodes"])
            NAER = int(fh["HC"].attrs["N_AER"])
            assert mHC == mY, f"/HC m_nodes={mHC} != /Y m_nodes={mY}"
            assert fh["HC"].shape == (mHC, NAER), f"/HC shape mismatch"
            print(f"       /HC : {fh['HC'].shape}  float64  OK")

        print("       All consistency checks passed.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_to_csv(h5_path: Path, out_dir: Optional[Path] = None) -> None:
    """Export /X, /Y, /HC from a standard store to CSV files."""
    if not _HAS_PANDAS:
        raise ImportError("pandas required for CSV export.  pip install pandas")

    h5_path = Path(h5_path)
    out_dir = Path(out_dir) if out_dir else h5_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as fh:
        X = fh["X"][()]
        try:
            pnames = [b.decode() for b in fh["X"].attrs["param_names"]]
        except Exception:
            pnames = [f"X{i}" for i in range(X.shape[1])]
        pd.DataFrame(X, columns=pnames).to_csv(
            out_dir / "X_parameters.csv", index=False)
        print(f"  X_parameters.csv  {X.shape}")

        Y = fh["Y"][()]
        try:
            nids = [b.decode() for b in fh["Y"].attrs["node_ids"]]
            if len(nids) != Y.shape[1]:
                nids = [f"node{j}" for j in range(Y.shape[1])]
        except Exception:
            nids = [f"node{j}" for j in range(Y.shape[1])]
        pd.DataFrame(Y.astype(np.float64), columns=nids).to_csv(
            out_dir / "Y_surges.csv", index=False)
        print(f"  Y_surges.csv      {Y.shape}")

        if "HC" in fh:
            HC = fh["HC"][()]
            try:
                aer  = fh["HC"].attrs["aer_levels"]
                hdrs = [f"AER_{a:.3e}" for a in aer]
            except Exception:
                hdrs = [f"AER_{j}" for j in range(HC.shape[1])]
            pd.DataFrame(HC, columns=hdrs).to_csv(
                out_dir / "HC_benchmark.csv", index=False)
            print(f"  HC_benchmark.csv  {HC.shape}")

    print(f"  Exported to: {out_dir}")
