"""HURDAT2 source resolution - basin registry, NHC discovery, download.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

NHC publishes best-track files in a listable directory
(``https://www.nhc.noaa.gov/data/hurdat/``) with date-stamped names, e.g.

    hurdat2-1851-2025-02272026.txt              (Atlantic)
    hurdat2-nepac-1949-2025-02272026.txt        (NE/NC Pacific)

``discover_latest`` fetches that listing and returns the newest file for a basin,
ranking by END YEAR first (the second 4-digit field) then by the trailing
MM-DD-YY[YY] stamp. ``resolve_source`` ties the policy together: an explicit
local file wins; otherwise download the latest (when allowed) or fall back to the
newest matching file already on disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import requests

NHC_DIR_URL = "https://www.nhc.noaa.gov/data/hurdat/"

# Per-basin canonical filename pattern. The "atl"/"nepac"-without-year variants
# NHC also hosts are intentionally excluded - only the full-record files match.
_BASIN_PATTERN = {
    "atlantic": re.compile(r"hurdat2-1851-(\d{4})-(\d{6,8})\.txt"),
    "pacific":  re.compile(r"hurdat2-nepac-1949-(\d{4})-(\d{6,8})\.txt"),
}

BASINS = tuple(_BASIN_PATTERN)   # ("atlantic", "pacific")

_TIMEOUT = 60   # seconds, for the directory listing and the file download


def _stamp_key(stamp: str) -> tuple:
    """Normalize an NHC date stamp (MMDDYY or MMDDYYYY) to (yyyy, mm, dd)."""
    if len(stamp) == 6:
        mm, dd, yyyy = int(stamp[:2]), int(stamp[2:4]), 2000 + int(stamp[4:6])
    elif len(stamp) == 8:
        mm, dd, yyyy = int(stamp[:2]), int(stamp[2:4]), int(stamp[4:8])
    else:
        return (0, 0, 0)
    return (yyyy, mm, dd)


def _validate_basin(basin: str) -> str:
    b = basin.strip().lower()
    if b not in _BASIN_PATTERN:
        raise ValueError(f"Unknown basin '{basin}'. Expected one of {BASINS}.")
    return b


def list_remote_files(basin: str, *, html: Optional[str] = None) -> list[str]:
    """All canonical filenames for ``basin`` found in the NHC directory listing.

    ``html`` may be supplied to parse a pre-fetched listing (used in tests);
    otherwise the live NHC directory is fetched.
    """
    basin = _validate_basin(basin)
    if html is None:
        resp = requests.get(NHC_DIR_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    # De-duplicate while preserving the match (anchors appear twice in the HTML).
    return sorted({m.group(0) for m in _BASIN_PATTERN[basin].finditer(html)})


def discover_latest(basin: str, *, html: Optional[str] = None) -> str:
    """Filename of the newest published best-track file for ``basin``."""
    basin = _validate_basin(basin)
    files = list_remote_files(basin, html=html)
    if not files:
        raise RuntimeError(
            f"No {basin} HURDAT2 files found at {NHC_DIR_URL} "
            f"(pattern {_BASIN_PATTERN[basin].pattern}).")

    def key(name: str) -> tuple:
        end_year, stamp = _BASIN_PATTERN[basin].search(name).groups()
        return (int(end_year), _stamp_key(stamp))

    return max(files, key=key)


def download(filename: str, dest_dir: Path, *, overwrite: bool = False) -> Path:
    """Download ``filename`` from the NHC directory into ``dest_dir``.

    Returns the local path. Skips the network when the file already exists and
    ``overwrite`` is False.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and not overwrite:
        return dest
    resp = requests.get(NHC_DIR_URL + filename, timeout=_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def find_local_latest(basin: str, *search_dirs: Path) -> Optional[Path]:
    """Newest matching file already on disk across ``search_dirs`` (or None)."""
    basin = _validate_basin(basin)
    pat = _BASIN_PATTERN[basin]
    candidates: list[Path] = []
    for d in search_dirs:
        d = Path(d)
        if d.is_dir():
            candidates += [p for p in d.iterdir() if pat.fullmatch(p.name)]
    if not candidates:
        return None

    def key(p: Path) -> tuple:
        end_year, stamp = pat.search(p.name).groups()
        return (int(end_year), _stamp_key(stamp))

    return max(candidates, key=key)


def resolve_source(
    basin: str,
    *,
    download_latest: bool,
    input_dir: Path,
    explicit_file: Optional[Path] = None,
    extra_search_dirs: tuple[Path, ...] = (),
    overwrite: bool = False,
) -> Path:
    """Resolve the local HURDAT2 path to parse for ``basin``.

    Policy (first match wins):
      1. ``explicit_file`` - an operator-pinned path (absolute, or relative to
         ``input_dir``).
      2. ``download_latest`` - discover and fetch the newest NHC file into
         ``input_dir``.
      3. otherwise - newest matching file already under ``input_dir`` (or any
         ``extra_search_dirs``).
    """
    basin = _validate_basin(basin)
    input_dir = Path(input_dir)

    if explicit_file:
        p = Path(explicit_file)
        if not p.is_absolute():
            p = input_dir / p
        if not p.is_file():
            raise FileNotFoundError(f"{basin}: explicit file not found: {p}")
        return p

    if download_latest:
        return download(discover_latest(basin), input_dir, overwrite=overwrite)

    local = find_local_latest(basin, input_dir, *extra_search_dirs)
    if local is None:
        raise FileNotFoundError(
            f"{basin}: no local HURDAT2 file under {input_dir} and downloading "
            f"is disabled. Set DOWNLOAD = True or drop a hurdat2-*.txt there.")
    return local
