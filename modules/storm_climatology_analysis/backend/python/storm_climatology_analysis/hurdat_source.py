"""hurdat_source - locate and load the augmented HURDAT2 source (the AHD module output).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The storm selection runs on the augmented best-track CSV produced by the sibling
``augmented_hurricane_database`` (AHD) module - the HURDAT-like table with
GP-metamodel-completed central pressure and Rmax. By default this module links to
the newest ``augmented_hurdat2_<basin>_*.csv`` under the AHD module's
``data/outputs``; the operator can instead pin an explicit path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# augmented_hurdat2_<basin>_<startyr>-<endyr>_<created>.csv  (created = YYYYMMDD)
_AHD_NAME = re.compile(
    r"augmented_hurdat2_(?P<basin>[a-z]+)_(\d{4})-(\d{4})_(?P<created>\d{8})\.csv$")

# Columns consumed downstream (loaded from the AHD CSV by name).
_NEEDED = ["tc_no", "year", "nhc_id", "name", "ymd",
           "lat", "lon", "vmax_kmh", "pmin_hpa", "trans_kmh", "heading_deg", "rmax_km"]


def sibling_ahd_outputs() -> Optional[Path]:
    """The sibling AHD module's data/outputs dir, found by walking up to modules/."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == "modules":
            return p / "augmented_hurricane_database" / "data" / "outputs"
    return None


def _rank_key(name: str) -> tuple:
    """Rank AHD filenames by (created date, end year); newest wins."""
    m = _AHD_NAME.search(name)
    if not m:
        return (0, 0)
    return (int(m.group("created")), int(m.group(3)))


def locate_augmented_hurdat(
    basin: str,
    *,
    explicit_file: Optional[Path] = None,
    input_dir: Path,
    ahd_outputs_dir: Optional[Path] = None,
) -> Path:
    """Resolve the augmented-HURDAT CSV for ``basin``.

    ``explicit_file`` (absolute, or relative to ``input_dir``) wins; otherwise the
    newest ``augmented_hurdat2_<basin>_*.csv`` under ``ahd_outputs_dir`` (or the
    auto-discovered sibling AHD outputs) is used.
    """
    if explicit_file:
        p = Path(explicit_file)
        if not p.is_absolute():
            p = Path(input_dir) / p
        if not p.is_file():
            raise FileNotFoundError(f"{basin}: augmented-HURDAT file not found: {p}")
        return p

    out_dir = Path(ahd_outputs_dir) if ahd_outputs_dir else sibling_ahd_outputs()
    if out_dir is None or not Path(out_dir).is_dir():
        raise FileNotFoundError(
            f"{basin}: cannot locate the augmented_hurricane_database outputs "
            f"(looked in {out_dir}). Set the augmented-HURDAT file path explicitly "
            f"or point ahd_outputs_dir at the AHD module's data/outputs.")
    out_dir = Path(out_dir)
    candidates = [p for p in out_dir.glob(f"augmented_hurdat2_{basin}_*.csv")
                  if _AHD_NAME.search(p.name)]
    if not candidates:
        raise FileNotFoundError(
            f"{basin}: no augmented_hurdat2_{basin}_*.csv found in {out_dir}. "
            f"Run the augmented_hurricane_database module first, or pin a path.")
    return max(candidates, key=lambda p: _rank_key(p.name))


def created_date(path) -> str:
    """The NHC HURDAT file creation date (YYYYMMDD) for output filenames.

    Parsed from the AHD filename (e.g. ``augmented_hurdat2_atlantic_1851-2025_
    20260227.csv`` -> ``20260227``). If the source is not AHD-named, it falls back
    to the file's modification date; returns "" if unavailable. The start/end
    years of the output tag come from the rate period (the orchestrator), not from
    the source filename.
    """
    m = _AHD_NAME.search(Path(path).name)
    if m:
        return m.group("created")
    try:
        import datetime as _dt
        ts = Path(path).stat().st_mtime
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y%m%d")
    except OSError:
        return ""


def load_augmented_hurdat(path) -> pd.DataFrame:
    """Load the augmented-HURDAT CSV (only the columns the analysis needs)."""
    path = Path(path)
    df = pd.read_csv(path, usecols=lambda c: c in _NEEDED)
    missing = [c for c in _NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return df
