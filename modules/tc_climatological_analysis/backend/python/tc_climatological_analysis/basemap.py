"""Natural Earth basemap (coastline + political boundaries) via pyshp.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Replaces the low-resolution legacy ``NOAA_WCL_Coastline_LowRes.mat`` /
``NOAA_WDB_Political.mat`` with Natural Earth vector data (coastline, country
borders, and state/province lines). The maps here are equirectangular (linear
lon/lat), so the shapefile line segments are drawn directly on a matplotlib axes
- no projection machinery (cartopy/GEOS/PROJ) is required.

Layers are downloaded once from the Natural Earth S3 bucket and cached under the
module's ``data/inputs/naturalearth`` folder; subsequent runs read the cache. If
the download (or pyshp) is unavailable the caller can still render maps without
the basemap.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Natural Earth official S3 mirror. {res} in {10m, 50m, 110m}.
_NE_BASE = "https://naturalearth.s3.amazonaws.com"
_LAYERS = {
    # name: (category, shapefile stem, draw style)
    "coastline": ("physical", "ne_{res}_coastline",
                  {"color": "black", "linewidth": 0.8, "zorder": 2.0}),
    "admin0": ("cultural", "ne_{res}_admin_0_boundary_lines_land",
               {"color": "0.45", "linewidth": 0.7, "zorder": 1.9}),
    "admin1": ("cultural", "ne_{res}_admin_1_states_provinces_lines",
               {"color": "0.6", "linewidth": 0.4, "zorder": 1.8}),
}

# Loaded line segments are cached per (resolution, cache_dir) within a process.
_CACHE: Dict[Tuple[str, str], Dict[str, List]] = {}


def _ensure_shapefile(res: str, category: str, stem: str, cache_dir: Path) -> Path:
    """Return the local .shp for a layer, downloading+extracting the zip if absent."""
    name = stem.format(res=res)
    dest = Path(cache_dir) / f"{res}_{category}" / name
    shp = dest.with_suffix(".shp")
    if shp.is_file():
        return shp
    import requests
    url = f"{_NE_BASE}/{res}_{category}/{name}.zip"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest.parent)
    if not shp.is_file():
        raise FileNotFoundError(f"Natural Earth shapefile missing after extract: {shp}")
    return shp


def _read_lines(shp: Path) -> List[Tuple]:
    """Read a (poly)line/polygon shapefile into a list of (lon, lat) part arrays."""
    import numpy as np
    import shapefile  # pyshp

    segments: List[Tuple] = []
    r = shapefile.Reader(str(shp))
    for shape in r.shapes():
        pts = shape.points
        if not pts:
            continue
        parts = list(shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            seg = pts[a:b]
            if len(seg) < 2:
                continue
            arr = np.asarray(seg, dtype=float)
            segments.append((arr[:, 0], arr[:, 1]))
    return segments


def basemap_lines(resolution: str, cache_dir) -> Dict[str, List]:
    """All basemap layers' line segments (cached per resolution+cache_dir)."""
    key = (resolution, str(cache_dir))
    if key in _CACHE:
        return _CACHE[key]
    layers: Dict[str, List] = {}
    for name, (category, stem, _style) in _LAYERS.items():
        shp = _ensure_shapefile(resolution, category, stem, Path(cache_dir))
        layers[name] = _read_lines(shp)
    _CACHE[key] = layers
    return layers


def _wrap360_pieces(lon, lat):
    """Wrap lon to [0,360) and split a segment where it crosses the dateline seam."""
    lon = np.asarray(lon) % 360.0
    brk = np.nonzero(np.abs(np.diff(lon)) > 180.0)[0]
    if brk.size == 0:
        return [(lon, lat)]
    pieces, start = [], 0
    for b in brk:
        pieces.append((lon[start:b + 1], lat[start:b + 1]))
        start = b + 1
    pieces.append((lon[start:], lat[start:]))
    return pieces


def draw_basemap(ax, layers: Dict[str, List], *, lon360: bool = False) -> None:
    """Draw cached basemap line segments onto ``ax`` (lon = x, lat = y).

    ``lon360`` renders in a 0-360 longitude frame (Pacific-centric), splitting any
    coastline segment that crosses the antimeridian so it draws no seam artifact.
    """
    for name, (_cat, _stem, style) in _LAYERS.items():
        for lon, lat in layers.get(name, []):
            if lon360:
                for plon, plat in _wrap360_pieces(lon, lat):
                    ax.plot(plon, plat, **style)
            else:
                ax.plot(lon, lat, **style)
