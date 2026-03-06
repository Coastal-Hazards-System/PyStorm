"""
backend/engines/io/readers.py
==============================
Low-level array loaders for every supported source format.

These functions are intentionally dumb: receive a path, return a numpy array.
No domain knowledge, no pipeline config, no HDF5 schema awareness.

Supported formats
-----------------
  .mat           MATLAB v5  (scipy.io)  or  v7.3/HDF5  (h5py, auto-detected)
  .csv           pandas.read_csv
  .txt / .dat    numpy.loadtxt
  .npy           numpy.load
  .npz           numpy.load archive
  .h5 / .hdf5    h5py dataset by path

Public API
----------
  load_array(path, varname, columns, id_col)
      -> (arr: ndarray float64, col_names: list[str], ids: list[str])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# MATLAB readers
# ---------------------------------------------------------------------------

def _is_mat73(path: Path) -> bool:
    """Return True if the .mat file is MATLAB v7.3 (HDF5-based).
    The HDF5 magic bytes start at offset 128 (after the MATLAB header),
    so at least 136 bytes must be read to detect them.
    """
    with open(path, "rb") as fh:
        return b"\x89HDF\r\n\x1a\n" in fh.read(136)


def _read_mat_v5(path: Path, varname: str) -> np.ndarray:
    if not _HAS_SCIPY:
        raise ImportError("scipy required for MATLAB v5 .mat files.  pip install scipy")
    data = sio.loadmat(str(path), variable_names=[varname])
    if varname not in data:
        avail = [k for k in data if not k.startswith("_")]
        raise KeyError(f"Variable '{varname}' not found in {path.name}.\n"
                       f"Available variables: {avail}")
    arr = data[varname]
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=float)


def _read_mat_v73(path: Path, varname: str) -> np.ndarray:
    """
    Load from MATLAB v7.3 (HDF5) .mat via h5py.
    MATLAB stores arrays column-major, so h5py reads them transposed;
    .T restores the expected [rows x cols] orientation.
    """
    if not _HAS_H5PY:
        raise ImportError("h5py required for MATLAB v7.3 .mat files.  pip install h5py")
    with h5py.File(path, "r") as fh:
        if varname not in fh:
            raise KeyError(f"Dataset '{varname}' not found in {path.name}.\n"
                           f"Available keys: {list(fh.keys())}")
        arr = fh[varname][()]
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr.T   # undo MATLAB HDF5 column-major transposition
    return arr


def _read_mat(path: Path, varname: str) -> np.ndarray:
    if _is_mat73(path):
        print("      Detected MATLAB v7.3 (HDF5) format.")
        return _read_mat_v73(path, varname)
    print("      Detected MATLAB v5 format.")
    try:
        return _read_mat_v5(path, varname)
    except NotImplementedError:
        # scipy confirmed it's actually v7.3 despite the header check missing it
        print("      Re-detected as MATLAB v7.3 (HDF5) format.")
        return _read_mat_v73(path, varname)


# ---------------------------------------------------------------------------
# Text / binary readers
# ---------------------------------------------------------------------------

def _read_csv(
    path: Path,
    varname: Optional[str] = None,
    columns: Optional[list] = None,
    id_col:  Optional[str]  = None,
) -> tuple[np.ndarray, list, list]:
    """
    Returns (arr, col_names, ids).
    varname is ignored (kept for call-site symmetry).
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas required for CSV files.  pip install pandas")
    df = pd.read_csv(path)
    ids: list = []
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
        df  = df.drop(columns=[id_col])
    if columns is not None:
        df = df[columns] if isinstance(columns[0], str) else df.iloc[:, columns]
    df = df.select_dtypes(include=[np.number])
    return df.values.astype(float), list(df.columns), ids


def _read_txt(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=float)


def _read_npy(path: Path) -> np.ndarray:
    return np.load(str(path)).astype(float)


def _read_npz(path: Path, varname: Optional[str] = None) -> np.ndarray:
    archive = np.load(str(path))
    keys = list(archive.files)
    if varname and varname in keys:
        return archive[varname].astype(float)
    if not varname and len(keys) == 1:
        return archive[keys[0]].astype(float)
    raise KeyError(f"npz '{path.name}' contains: {keys}.\n"
                   "Specify the array via X/Y/HC_variable in config.")


def _read_h5_dataset(path: Path, varname: str) -> np.ndarray:
    if not _HAS_H5PY:
        raise ImportError("h5py required for HDF5 files.  pip install h5py")
    with h5py.File(path, "r") as fh:
        if varname not in fh:
            dsets: list = []
            fh.visititems(
                lambda n, o: dsets.append(f"  /{n}  {o.shape}")
                if isinstance(o, h5py.Dataset) else None
            )
            raise KeyError(f"Dataset '{varname}' not found in {path.name}.\n"
                           "Available datasets:\n" + "\n".join(dsets))
        return fh[varname][()].astype(float)


# ---------------------------------------------------------------------------
# Unified public entry point
# ---------------------------------------------------------------------------

def load_array(
    path:    Path,
    varname: Optional[str]  = None,
    columns: Optional[list] = None,
    id_col:  Optional[str]  = None,
) -> tuple[np.ndarray, list, list]:
    """
    Load a 2-D float array from any supported file format.

    Returns
    -------
    arr       : float64 ndarray, always 2-D
    col_names : list[str]  — column headers (CSV only; [] otherwise)
    ids       : list[str]  — ID strings (CSV id_col only; [] otherwise)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix    = path.suffix.lower()
    col_names: list = []
    ids:       list = []

    if suffix == ".mat":
        if not varname:
            raise ValueError(f"varname required for .mat files ({path.name})")
        arr = _read_mat(path, varname)
    elif suffix == ".csv":
        arr, col_names, ids = _read_csv(path, varname, columns, id_col)
    elif suffix in (".txt", ".dat"):
        arr = _read_txt(path)
    elif suffix == ".npy":
        arr = _read_npy(path)
    elif suffix == ".npz":
        arr = _read_npz(path, varname)
    elif suffix in (".h5", ".hdf5", ".he5"):
        if not varname:
            raise ValueError(f"varname required for HDF5 files ({path.name})")
        arr = _read_h5_dataset(path, varname)
    else:
        raise ValueError(f"Unsupported extension '{suffix}' ({path.name}).\n"
                         "Supported: .mat  .csv  .txt  .dat  .npy  .npz  .h5  .hdf5")

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Array from {path.name} is {arr.ndim}-D; expected 2-D.")

    return arr, col_names, ids
