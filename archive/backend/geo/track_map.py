"""
backend/geo/track_map.py
=========================
Lightweight map for the bounding-box filter diagnostic.

Renders a single PNG with:
  (a) satellite/terrain basemap  (Stamen Terrain tiles via urllib)
  (b) bounding box rectangle
  (c) all store nodes  (light grey)
  (d) nodes within the bounding box  (red)
  (e) all TC tracks  (thin grey)
  (f) TC tracks within the radial filter  (coloured)
  (g) radial distance circle around the bbox centroid
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Tile fetcher  (OpenStreetMap — no API key required)
# ---------------------------------------------------------------------------

def _deg2num(lat_deg: float, lon_deg: float, zoom: int):
    """Convert lat/lon to tile x, y at a given zoom level."""
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _num2deg(x: int, y: int, zoom: int):
    """Convert tile x, y to NW corner lat/lon."""
    n = 2 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def _merc_y(lat_deg):
    """Convert latitude to Web Mercator Y (in degrees, for imshow extent)."""
    lat_rad = math.radians(lat_deg)
    return math.degrees(math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)))


def _fetch_basemap(ax, lon_min, lon_max, lat_min, lat_max, zoom=7):
    """
    Fetch OSM tiles and composite onto the axes as a background image.

    OSM tiles are in Web Mercator projection.  We warp each tile column
    to lat/lon space so the basemap aligns with the data coordinates.
    Falls back gracefully if network is unavailable.
    """
    try:
        x_min, y_min = _deg2num(lat_max, lon_min, zoom)  # NW corner
        x_max, y_max = _deg2num(lat_min, lon_max, zoom)  # SE corner

        # Clamp tile range to avoid excessive downloads
        n_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
        if n_tiles > 100:
            print(f"    [basemap] Too many tiles ({n_tiles}) — skipping satellite layer")
            return

        # Assemble tile mosaic
        rows = []
        for ty in range(y_min, y_max + 1):
            row_imgs = []
            for tx in range(x_min, x_max + 1):
                url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                req = urllib.request.Request(url, headers={"User-Agent": "PyStorm/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    img_data = resp.read()
                img = plt.imread(io.BytesIO(img_data), format="png")
                row_imgs.append(img)
            rows.append(np.concatenate(row_imgs, axis=1))
        mosaic = np.concatenate(rows, axis=0)

        # Tile edges in geographic coordinates
        nw_lat, nw_lon = _num2deg(x_min, y_min, zoom)
        se_lat, se_lon = _num2deg(x_max + 1, y_max + 1, zoom)

        # The mosaic pixels are uniformly spaced in Mercator Y, but the
        # axes are in geographic lat.  Re-sample each pixel column from
        # Mercator Y to lat so the image aligns with the data.
        h, w = mosaic.shape[:2]
        merc_top = _merc_y(nw_lat)
        merc_bot = _merc_y(se_lat)
        # Target lat rows (uniform in lat)
        target_lats = np.linspace(nw_lat, se_lat, h)
        # Source row for each target lat (uniform in Mercator Y)
        target_merc = np.array([_merc_y(la) for la in target_lats])
        src_rows = (merc_top - target_merc) / (merc_top - merc_bot) * (h - 1)
        src_rows = np.clip(src_rows, 0, h - 1).astype(int)
        warped = mosaic[src_rows, :, :]

        ax.imshow(warped, extent=[nw_lon, se_lon, se_lat, nw_lat],
                  aspect="auto", zorder=0, alpha=0.6)
    except Exception as e:
        print(f"    [basemap] Could not fetch tiles: {e}")


# ---------------------------------------------------------------------------
# Radial circle in lat/lon
# ---------------------------------------------------------------------------

def _geodesic_circle(lat_c, lon_c, radius_km, n_pts=120):
    """Approximate circle on a sphere, returned as (lons, lats) arrays."""
    R = 6_371.0
    lats, lons = [], []
    for az in np.linspace(0, 2 * np.pi, n_pts, endpoint=True):
        d_lat = (radius_km / R) * np.cos(az)
        d_lon = (radius_km / R) * np.sin(az) / np.cos(np.radians(lat_c))
        lats.append(lat_c + np.degrees(d_lat))
        lons.append(lon_c + np.degrees(d_lon))
    return np.array(lons), np.array(lats)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_bbox_map(
    bbox: dict,
    all_node_lats: np.ndarray,
    all_node_lons: np.ndarray,
    bbox_node_lats: np.ndarray,
    bbox_node_lons: np.ndarray,
    tracks: list[np.ndarray],
    storm_indices_near: np.ndarray,
    medoid_lat: float,
    medoid_lon: float,
    max_dist_km: float,
    output_dir: str | Path,
    filename: str = "bbox_filter_map.png",
    node_stride_map: int = 10,
):
    """
    Render the diagnostic map and save as PNG.

    Parameters
    ----------
    bbox              : {"lat_min", "lat_max", "lon_min", "lon_max"}
    all_node_lats/lons: coordinates of all store nodes
    bbox_node_lats/lons: coordinates of nodes inside the bbox
    tracks            : list of (N,2) arrays [lat, lon] per storm
    storm_indices_near: 0-based indices of storms passing the radial filter
    medoid_lat/lon    : centroid of bbox nodes
    max_dist_km       : radial filter distance
    output_dir        : directory for the output PNG
    filename          : output filename
    node_stride_map   : subsample all-nodes for plotting (avoid overplotting)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    near_set = set(storm_indices_near.tolist())

    fig, ax = plt.subplots(figsize=(14, 10))

    # ── Determine map extent ────────────────────────────────────────────
    pad = 2.0  # degrees padding
    # Use bbox + track extent
    all_track_lats = np.concatenate([t[:, 0] for t in tracks if t.shape[0] > 0])
    all_track_lons = np.concatenate([t[:, 1] for t in tracks if t.shape[0] > 0])

    map_lon_min = min(bbox["lon_min"], float(all_track_lons.min())) - pad
    map_lon_max = max(bbox["lon_max"], float(all_track_lons.max())) + pad
    map_lat_min = min(bbox["lat_min"], float(all_track_lats.min())) - pad
    map_lat_max = max(bbox["lat_max"], float(all_track_lats.max())) + pad

    # Clamp to reasonable range
    map_lat_min = max(map_lat_min, 10.0)
    map_lat_max = min(map_lat_max, 50.0)
    map_lon_min = max(map_lon_min, -100.0)
    map_lon_max = min(map_lon_max, -60.0)

    ax.set_xlim(map_lon_min, map_lon_max)
    ax.set_ylim(map_lat_min, map_lat_max)

    # (a) Satellite / terrain basemap
    print("    Fetching basemap tiles ...")
    _fetch_basemap(ax, map_lon_min, map_lon_max, map_lat_min, map_lat_max, zoom=6)

    # (e) All TC tracks (thin grey, behind everything)
    for i, trk in enumerate(tracks):
        if trk.shape[0] < 2:
            continue
        if i not in near_set:
            ax.plot(trk[:, 1], trk[:, 0], color="#999999", linewidth=0.3,
                    alpha=0.4, zorder=1)

    # (f) Tracks within radial distance (coloured)
    for i in storm_indices_near:
        trk = tracks[i]
        if trk.shape[0] < 2:
            continue
        ax.plot(trk[:, 1], trk[:, 0], color="#2196F3", linewidth=0.6,
                alpha=0.7, zorder=2)

    # (c) All store nodes (light grey, subsampled)
    valid = ~np.isnan(all_node_lats)
    stride_idx = np.arange(0, valid.sum(), max(1, node_stride_map))
    ax.scatter(all_node_lons[valid][stride_idx], all_node_lats[valid][stride_idx],
               s=0.3, c="#CCCCCC", alpha=0.3, zorder=3, label="All nodes")

    # (d) Nodes within bbox (red)
    bbox_stride = max(1, len(bbox_node_lats) // 5000)
    ax.scatter(bbox_node_lons[::bbox_stride], bbox_node_lats[::bbox_stride],
               s=1.0, c="#E53935", alpha=0.6, zorder=4, label="Bbox nodes")

    # (b) Bounding box rectangle
    bx = [bbox["lon_min"], bbox["lon_max"], bbox["lon_max"], bbox["lon_min"], bbox["lon_min"]]
    by = [bbox["lat_min"], bbox["lat_min"], bbox["lat_max"], bbox["lat_max"], bbox["lat_min"]]
    ax.plot(bx, by, color="#FF6F00", linewidth=2.5, zorder=5, label="Bounding box")

    # (g) Radial distance circle
    circ_lons, circ_lats = _geodesic_circle(medoid_lat, medoid_lon, max_dist_km)
    ax.plot(circ_lons, circ_lats, color="#7B1FA2", linewidth=1.8, linestyle="--",
            zorder=5, label=f"Radius {max_dist_km:.0f} km")

    # Centroid marker
    ax.plot(medoid_lon, medoid_lat, marker="*", color="#FF6F00", markersize=14,
            zorder=6, markeredgecolor="black", markeredgewidth=0.5,
            label="Nodes medoid")

    # ── Annotation & legend ─────────────────────────────────────────────
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"Bounding-Box Filter — "
        f"{len(bbox_node_lats)} nodes, "
        f"{len(storm_indices_near)} storms (within {max_dist_km:.0f} km)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Map saved -> {out_path}")
