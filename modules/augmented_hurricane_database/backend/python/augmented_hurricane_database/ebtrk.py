"""EBTRK Rmax backfill - fill missing HURDAT2 radius-of-maximum-wind.

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

Latest-file discovery. Like the HURDAT2 source resolver, this module finds the
newest published EBTRK file rather than pinning a fixed name. CIRA lists the
files on a landing page (``EBTRK_LIST_URL``) with names that carry the record
END YEAR and a DD-Mon-YYYY publish stamp, e.g.

    EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt

``discover_latest_ebtrk`` fetches that listing and returns the newest "new
format" file URL for a code (AL/EP/CP), ranking by end year then stamp. If the
listing cannot be reached or parsed it falls back to the known default file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

# CIRA EBTRK landing page (lists the files) and the uploads directory the files
# live under (used to build the fallback default URL).
EBTRK_LIST_URL = ("https://rammb2.cira.colostate.edu/research/tropical-cyclones/"
                  "tc_extended_best_track_dataset/")
EBTRK_URL_BASE = "https://rammb2.cira.colostate.edu/wp-content/uploads/2020/11/"

# Per-code EBTRK file metadata, keyed by the cyclone-id prefix (which matches
# HURDAT directly): the known default filename (a last-resort fallback when
# discovery is unavailable), a local-search glob, and a discovery regex that
# matches only the canonical "new format" file and captures (end_year, stamp).
# "stamp" is the DD-Mon-YYYY publish date; an optional "-N" duplicate suffix is
# allowed. The "old format" files NHC also hosts are intentionally excluded.
def _ebtrk_re(code: str) -> "re.Pattern[str]":
    return re.compile(
        rf"EBTRK_{code}_final_\d{{4}}-(\d{{4}})_new_format_"
        r"(\d{2}-[A-Za-z]{3}-\d{4})(?:-\d+)?\.txt")


_EBTRK_CODE = {
    "AL": {"default": "EBTRK_AL_final_1851-2021_new_format_02-Sep-2022-1.txt",
           "glob": "EBTRK_AL_*.txt", "re": _ebtrk_re("AL")},
    "EP": {"default": "EBTRK_EP_final_1949-2021_new_format_02-Sep-2022.txt",
           "glob": "EBTRK_EP_*.txt", "re": _ebtrk_re("EP")},
    "CP": {"default": "EBTRK_CP_final_1950-2021_new_format_02-Sep-2022-1.txt",
           "glob": "EBTRK_CP_*.txt", "re": _ebtrk_re("CP")},
}

# Which EBTRK file code(s) each HURDAT basin needs. The HURDAT nepac ("pacific")
# record contains both East Pacific (EP) and Central Pacific (CP) storms, each
# with its own EBTRK file, so the Pacific basin needs both.
EBTRK_BASIN_CODES = {
    "atlantic": ["AL"],
    "pacific":  ["EP", "CP"],
}
EBTRK_BASINS = tuple(EBTRK_BASIN_CODES)

# href="..." extractor for the listing page (handles relative or absolute URLs).
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

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


def _stamp_key(stamp: str) -> tuple:
    """Normalize an EBTRK publish stamp (DD-Mon-YYYY) to (yyyy, mm, dd)."""
    try:
        dd, mon, yyyy = stamp.split("-")
        return (int(yyyy), _MONTHS.get(mon[:3].title(), 0), int(dd))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def list_remote_ebtrk(code: str, *, html: Optional[str] = None) -> list[str]:
    """All "new format" EBTRK file URLs for ``code`` (AL/EP/CP) in the listing.

    ``html`` may be supplied to parse a pre-fetched listing (used in tests);
    otherwise the live CIRA landing page is fetched. Relative hrefs are resolved
    against ``EBTRK_LIST_URL``.
    """
    if code not in _EBTRK_CODE:
        raise ValueError(f"Unknown EBTRK code '{code}'. Expected one of "
                         f"{tuple(_EBTRK_CODE)}.")
    if html is None:
        resp = requests.get(EBTRK_LIST_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    pat = _EBTRK_CODE[code]["re"]
    urls = {}
    for href in _HREF_RE.findall(html):
        if pat.search(href):
            urls[urljoin(EBTRK_LIST_URL, href)] = None
    return sorted(urls)


def discover_latest_ebtrk(code: str, *, html: Optional[str] = None) -> str:
    """URL of the newest published "new format" EBTRK file for ``code``."""
    pat = _EBTRK_CODE[code]["re"]

    def key(url: str) -> tuple:
        m = pat.search(url)
        end_year, stamp = m.groups()
        return (int(end_year), _stamp_key(stamp))

    urls = list_remote_ebtrk(code, html=html)
    if not urls:
        raise RuntimeError(
            f"No {code} new-format EBTRK file found at {EBTRK_LIST_URL} "
            f"(pattern {pat.pattern}).")
    return max(urls, key=key)


def _latest_url(code: str) -> str:
    """Discovered newest URL for ``code``, or the known default on any failure."""
    try:
        return discover_latest_ebtrk(code)
    except Exception as exc:                                   # noqa: BLE001
        default = EBTRK_URL_BASE + _EBTRK_CODE[code]["default"]
        print(f"[ahd] EBTRK {code}: discovery failed ({exc}); "
              f"falling back to default {default}")
        return default


def download_ebtrk(url: str, dest_dir: Path, *, overwrite: bool = False) -> Path:
    """Download an EBTRK file ``url`` into ``dest_dir`` (skip if present).

    The local filename is the URL basename.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    dest = dest_dir / filename
    if dest.exists() and not overwrite:
        return dest
    resp = requests.get(url, timeout=_TIMEOUT)
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


def find_local_ebtrk(code: str, *search_dirs: Path) -> Optional[Path]:
    """Newest "new format" EBTRK file for ``code`` already on disk (or None).

    Mirrors the HURDAT2 resolver: rank matching files by (end year, publish
    stamp). Falls back to the broadest glob (newest by name) if nothing matches
    the canonical new-format pattern, so legacy local copies still resolve.
    """
    pat = _EBTRK_CODE[code]["re"]
    glob_pat = _EBTRK_CODE[code]["glob"]
    matched: list[tuple[tuple, Path]] = []
    glob_hits: list[Path] = []
    for d in search_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in d.glob(glob_pat):
            glob_hits.append(p)
            m = pat.fullmatch(p.name)
            if m:
                end_year, stamp = m.groups()
                matched.append(((int(end_year), _stamp_key(stamp)), p))
    if matched:
        return max(matched, key=lambda kv: kv[0])[1]
    if glob_hits:
        return sorted(glob_hits)[-1]
    return None


def _resolve_one(code: str, *, download: bool, input_dir: Path,
                 extra_search_dirs, overwrite: bool,
                 url: Optional[str] = None) -> Path:
    """Resolve one EBTRK file (by code): URL/discovery download, or find locally.

    On the download path an operator-supplied ``url`` overrides discovery;
    otherwise the newest file is discovered from the CIRA listing. When
    ``download`` is False the newest matching local file is used.
    """
    if download:
        target = url or _latest_url(code)
        return download_ebtrk(target, input_dir, overwrite=overwrite)
    local = find_local_ebtrk(code, input_dir, *extra_search_dirs)
    if local is not None:
        return local
    raise FileNotFoundError(
        f"No local EBTRK {code} file (glob {_EBTRK_CODE[code]['glob']}) found "
        f"and downloading is disabled.")


def resolve_ebtrk_sources(
    basin: str,
    *,
    download: bool,
    input_dir: Path,
    explicit_files=None,
    code_urls: Optional[dict] = None,
    extra_search_dirs: tuple[Path, ...] = (),
    overwrite: bool = False,
) -> list[Path]:
    """Resolve the EBTRK file path(s) for ``basin`` (one for Atlantic, two for Pacific).

    ``explicit_files`` (a path or list of paths) overrides automatic resolution
    with operator-pinned local files. Otherwise each of the basin's files is, on
    the download path, fetched from its per-code ``code_urls[code]`` override (if
    given) or the newest file discovered from the CIRA listing; with downloading
    off, the newest matching local file is used. ``code_urls`` is keyed by EBTRK
    code (AL/EP/CP) and is honored only when ``download`` is True.
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

    if basin not in EBTRK_BASIN_CODES:
        raise ValueError(f"No EBTRK source defined for basin '{basin}'.")
    code_urls = code_urls or {}
    return [
        _resolve_one(code, download=download, input_dir=input_dir,
                     extra_search_dirs=extra_search_dirs, overwrite=overwrite,
                     url=code_urls.get(code))
        for code in EBTRK_BASIN_CODES[basin]
    ]
