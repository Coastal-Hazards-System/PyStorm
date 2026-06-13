"""io - input loaders for the storm_surge_hydrograph (SSH) module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Reads the CTXS (Coastal Texas Study) save-point metadata and the per-save-point synthetic-TC surge
matrices. Surge values are water-surface elevation in metres above NAVD88, with
``-99999`` meaning the point is DRY (water below ground) and trailing ``NaN`` being
padding (storms of unequal length). Ground elevation above NAVD88 is ``-depth``
(the staID depth column is positive-down).
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# staID columns (no header in the source file).
_STAID_COLUMNS = ["sp_id", "lat", "lon", "depth"]


@dataclass
class SavePoint:
    sp_id: int
    lat: float
    lon: float
    depth: float          # m, positive-down (as provided)
    ground_elev: float    # m above NAVD88 (= -depth when depth is positive-down)
    surge_path: Path


def load_staid(path, *, depth_is_positive_down: bool = True) -> pd.DataFrame:
    """Load the save-point metadata table (id, lat, lon, depth, ground_elev)."""
    df = pd.read_csv(path, header=None, names=_STAID_COLUMNS)
    df["sp_id"] = df["sp_id"].astype(int)
    sign = -1.0 if depth_is_positive_down else 1.0
    df["ground_elev"] = sign * df["depth"].astype(float)
    return df


def _sp_from_name(name: str) -> Optional[int]:
    m = re.search(r"SP0*([0-9]+)", name)
    return int(m.group(1)) if m else None


def discover_save_points(
    raw_dir: Path,
    staid: pd.DataFrame,
    surge_file_glob: str,
    *,
    only: Optional[List[int]] = None,
) -> List[SavePoint]:
    """Find every surge matrix on disk and pair it with its staID metadata."""
    raw_dir = Path(raw_dir)
    pattern = surge_file_glob.replace("{sp}", "*")
    meta = {int(r.sp_id): r for r in staid.itertuples()}
    pts: List[SavePoint] = []
    for fp in sorted(raw_dir.glob(pattern)):
        spid = _sp_from_name(fp.name)
        if spid is None or spid not in meta:
            continue
        if only is not None and spid not in set(only):
            continue
        r = meta[spid]
        pts.append(SavePoint(sp_id=spid, lat=float(r.lat), lon=float(r.lon),
                             depth=float(r.depth), ground_elev=float(r.ground_elev),
                             surge_path=fp))
    return pts


def load_surge_matrix(path) -> np.ndarray:
    """Surge matrix as (n_timesteps, n_storms) float array (NaN padding preserved)."""
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def confirm_time_step(path, dt_hours: float) -> float:
    """Return the median time step (h) inferred from the first storm's timestamps.

    Best-effort: parses the first column of the timestamp matrix. Returns the
    configured ``dt_hours`` if parsing fails (the step is otherwise assumed).
    """
    try:
        col = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0].dropna()
        ts = pd.to_datetime(col, format="%d-%b-%Y %H:%M:%S", errors="coerce").dropna()
        if len(ts) < 2:
            return dt_hours
        step = np.median(np.diff(ts.to_numpy()).astype("timedelta64[s]").astype(float))
        return step / 3600.0
    except Exception:                                              # noqa: BLE001
        return dt_hours
