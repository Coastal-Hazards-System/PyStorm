"""EBTRK Rmax backfill — fill missing HURDAT2 radius-of-maximum-wind.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Port of ``ebtrk/CHS_TC_HURDAT_Atlantic_with_ebtrk_Rm_append_v3.m``, extended to
the Pacific. HURDAT2 records the radius of maximum wind (Rmax) only for recent
seasons; the Extended Best Track (EBTRK) dataset (Demuth/Pennington/CIRA) carries
Rmax back to the late 1980s. This module downloads/parses the EBTRK file(s) for a
basin and fills the ``rmax_km`` of HURDAT rows where it is missing (NaN), leaving
any HURDAT-provided Rmax untouched.

Basins. The Atlantic uses the AL file. The HURDAT nepac ("pacific") record
contains both East Pacific (EP) and Central Pacific (CP) storms, which have
separate EBTRK files, so the Pacific basin uses both (EP and CP). The cyclone-id
prefixes (AL / EP / CP) match HURDAT directly.

Join key. The original MATLAB matched EBTRK to HURDAT on synoptic datetime alone,
which is ambiguous when two storms share a synoptic time. The EBTRK record's
first eight characters are exactly the HURDAT cyclone id (e.g. ``AL071988``), so
here the join is on (nhc_id, synoptic datetime), exact per storm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

# CIRA EBTRK download location.
EBTRK_URL_BASE = "https://rammb2.cira.colostate.edu/wp-content/uploads/2020/11/"

# Per-basin EBTRK sources: (default filename, local-search glob). The HURDAT
# nepac ("pacific") record contains both East Pacific (EP) and Central Pacific
# (CP) storms, each with its own EBTRK file, so the Pacific basin needs both.
# The cyclone-id prefixes (AL / EP / CP) match HURDAT directly.
EBTRK_SOURCES = {
    "atlantic": [
        ("EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt", "EBTRK_AL_*.txt"),
    ],
    "pacific": [
        ("EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt", "EBTRK_EP_*.txt"),
        ("EBTRK_CP_final_1950-2021_new_format_02-Sep-2022-1.txt", "EBTRK_CP_*.txt"),
    ],
}
EBTRK_BASINS = tuple(EBTRK_SOURCES)

_NM2KM = 1.852
_RMAX_BAD = -99
_TIMEOUT = 120

# Fixed-width column slices (from the MATLAB textscan formatSpec
# %4s%4s%13s%2s%2s%2s%5s%6s%7s%4s%5s%4s...). 0-based [start, stop).
_SLICES = {
    "nhc_id": (0, 8),     # AL## + YYYY == HURDAT cyclone id
    "month":  (21, 23),
    "day":    (23, 25),
    "hour":   (25, 27),
    "year":   (27, 32),
    "rmax":   (54, 58),   # radius of maximum wind, nautical miles (-99 = missing)
}


def download_ebtrk(dest_dir: Path, *, filename: str,
                   overwrite: bool = False) -> Path:
    """Download an EBTRK file from CIRA into ``dest_dir`` (skip if present)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and not overwrite:
        return dest
    resp = requests.get(EBTRK_URL_BASE + filename, timeout=_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_ebtrk(path: Path) -> pd.DataFrame:
    """Parse the EBTRK fixed-width file into rows with a usable Rmax.

    Returns a DataFrame with columns ``nhc_id``, ``dt_key`` (YYYYMMDDHH string),
    and ``rmax_km`` (nm -> km, rounded). Rows whose Rmax is the -99 sentinel are
    dropped.
    """
    records = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            def f(col: str) -> str:
                a, b = _SLICES[col]
                return line[a:b].strip()
            try:
                rmax_nm = int(f("rmax"))
            except ValueError:
                continue
            if rmax_nm == _RMAX_BAD:
                continue
            dt_key = f("year") + f("month").zfill(2) + f("day").zfill(2) + f("hour").zfill(2)
            records.append((f("nhc_id"), dt_key, float(round(rmax_nm * _NM2KM))))

    return pd.DataFrame(records, columns=["nhc_id", "dt_key", "rmax_km"])


def _hurdat_dt_key(df: pd.DataFrame) -> pd.Series:
    """YYYYMMDDHH synoptic key for HURDAT rows (from ymd + hhmm)."""
    ymd = df["ymd"].astype("int64").astype(str).str.zfill(8)
    hour = (df["hhmm"].astype("int64") // 100).astype(str).str.zfill(2)
    return ymd + hour


def fill_missing_rmax(df: pd.DataFrame, ebtrk: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Fill NaN ``rmax_km`` in ``df`` from ``ebtrk``, matched on (nhc_id, datetime).

    Returns the updated DataFrame and the number of rows filled. HURDAT rows
    that already have an Rmax are left unchanged.
    """
    if ebtrk.empty:
        return df, 0

    # EBTRK may carry duplicate (id, time) keys across format quirks; keep first.
    lookup = (ebtrk.drop_duplicates(subset=["nhc_id", "dt_key"])
                   .set_index(["nhc_id", "dt_key"])["rmax_km"])

    df = df.copy()
    missing = df["rmax_km"].isna()
    if not missing.any():
        return df, 0

    keys = pd.MultiIndex.from_arrays(
        [df.loc[missing, "nhc_id"], _hurdat_dt_key(df.loc[missing])])
    filled_vals = lookup.reindex(keys).to_numpy()

    target_idx = df.index[missing]
    got = ~np.isnan(filled_vals)
    df.loc[target_idx[got], "rmax_km"] = filled_vals[got]
    return df, int(got.sum())


def parse_ebtrk_files(paths) -> pd.DataFrame:
    """Parse and concatenate one or more EBTRK files into a single table."""
    frames = [parse_ebtrk(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["nhc_id", "dt_key", "rmax_km"])
    return pd.concat(frames, ignore_index=True)


def _resolve_one(filename: str, glob_pat: str, *, download: bool,
                 input_dir: Path, extra_search_dirs, overwrite: bool) -> Path:
    """Resolve one EBTRK file: download it, or find it locally by name or glob."""
    if download:
        return download_ebtrk(input_dir, filename=filename, overwrite=overwrite)
    for d in (input_dir, *extra_search_dirs):
        d = Path(d)
        cand = d / filename
        if cand.is_file():
            return cand
        if d.is_dir():
            hits = sorted(d.glob(glob_pat))
            if hits:
                return hits[-1]
    raise FileNotFoundError(
        f"No local EBTRK file ({filename} or {glob_pat}) found and downloading "
        f"is disabled.")


def resolve_ebtrk_sources(
    basin: str,
    *,
    download: bool,
    input_dir: Path,
    explicit_files=None,
    extra_search_dirs: tuple[Path, ...] = (),
    overwrite: bool = False,
) -> list[Path]:
    """Resolve the EBTRK file path(s) for ``basin`` (one for Atlantic, two for Pacific).

    ``explicit_files`` (a path or list of paths) overrides automatic resolution.
    Otherwise each of the basin's files is downloaded (``download``) or found in
    ``input_dir`` / ``extra_search_dirs`` by exact name or glob.
    """
    input_dir = Path(input_dir)
    if explicit_files:
        if isinstance(explicit_files, (str, Path)):
            explicit_files = [explicit_files]
        out = []
        for f in explicit_files:
            p = Path(f)
            if not p.is_absolute():
                p = input_dir / p
            if not p.is_file():
                raise FileNotFoundError(f"EBTRK explicit file not found: {p}")
            out.append(p)
        return out

    if basin not in EBTRK_SOURCES:
        raise ValueError(f"No EBTRK source defined for basin '{basin}'.")
    return [
        _resolve_one(fn, pat, download=download, input_dir=input_dir,
                     extra_search_dirs=extra_search_dirs, overwrite=overwrite)
        for fn, pat in EBTRK_SOURCES[basin]
    ]
