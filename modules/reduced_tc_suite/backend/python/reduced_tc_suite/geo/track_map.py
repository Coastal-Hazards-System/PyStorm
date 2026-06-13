"""Lightweight map for the bounding-box filter diagnostic.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Renders a single PNG with:
  (a) OSM tile basemap
  (b) bounding box rectangle
  (c) all store nodes  (light grey)
  (d) nodes within the bounding box  (red)
  (e) all TC tracks  (thin grey)
  (f) TC tracks within the radial filter  (colored)
  (g) radial distance circle around the bbox centroid
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

from pystorm_common import save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tile fetcher  (OpenStreetMap - no API key required)
# ---------------------------------------------------------------------------

def _deg2num(lat_deg: float, lon_deg: float, zoom: int):
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _num2deg(x: int, y: int, zoom: int):
    n = 2 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def _merc_y(lat_deg):
    lat_rad = math.radians(lat_deg)
    return math.degrees(math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)))


def _fetch_basemap(ax, lon_min, lon_max, lat_min, lat_max, zoom=7):
    try:
        x_min, y_min = _deg2num(lat_max, lon_min, zoom)
        x_max, y_max = _deg2num(lat_min, lon_max, zoom)

        n_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
        if n_tiles > 100:
            print(f"    [basemap] Too many tiles ({n_tiles}) - skipping satellite layer")
            return

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

        nw_lat, nw_lon = _num2deg(x_min, y_min, zoom)
        se_lat, se_lon = _num2deg(x_max + 1, y_max + 1, zoom)

        h, w = mosaic.shape[:2]
        merc_top = _merc_y(nw_lat)
        merc_bot = _merc_y(se_lat)
        target_lats = np.linspace(nw_lat, se_lat, h)
        target_merc = np.array([_merc_y(la) for la in target_lats])
        src_rows = (merc_top - target_merc) / (merc_top - merc_bot) * (h - 1)
        src_rows = np.clip(src_rows, 0, h - 1).astype(int)
        warped = mosaic[src_rows, :, :]

        ax.imshow(warped, extent=[nw_lon, se_lon, se_lat, nw_lat],
                  aspect="auto", zorder=0, alpha=0.6)
    except Exception as e:
        print(f"    [basemap] Could not fetch tiles: {e}")


def _geodesic_circle(lat_c, lon_c, radius_km, n_pts=120):
    R = 6_371.0
    lats, lons = [], []
    for az in np.linspace(0, 2 * np.pi, n_pts, endpoint=True):
        d_lat = (radius_km / R) * np.cos(az)
        d_lon = (radius_km / R) * np.sin(az) / np.cos(np.radians(lat_c))
        lats.append(lat_c + np.degrees(d_lat))
        lons.append(lon_c + np.degrees(d_lon))
    return np.array(lons), np.array(lats)


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
    """Render the diagnostic map and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    near_set = set(storm_indices_near.tolist())

    fig, ax = plt.subplots(figsize=(14, 10))

    pad = 2.0
    all_track_lats = np.concatenate([t[:, 0] for t in tracks if t.shape[0] > 0])
    all_track_lons = np.concatenate([t[:, 1] for t in tracks if t.shape[0] > 0])

    map_lon_min = min(bbox["lon_min"], float(all_track_lons.min())) - pad
    map_lon_max = max(bbox["lon_max"], float(all_track_lons.max())) + pad
    map_lat_min = min(bbox["lat_min"], float(all_track_lats.min())) - pad
    map_lat_max = max(bbox["lat_max"], float(all_track_lats.max())) + pad

    map_lat_min = max(map_lat_min, 10.0)
    map_lat_max = min(map_lat_max, 50.0)
    map_lon_min = max(map_lon_min, -100.0)
    map_lon_max = min(map_lon_max, -60.0)

    ax.set_xlim(map_lon_min, map_lon_max)
    ax.set_ylim(map_lat_min, map_lat_max)

    print("    Fetching basemap tiles ...")
    _fetch_basemap(ax, map_lon_min, map_lon_max, map_lat_min, map_lat_max, zoom=6)

    valid = ~np.isnan(all_node_lats)
    stride_idx = np.arange(0, valid.sum(), max(1, node_stride_map))
    ax.scatter(all_node_lons[valid][stride_idx], all_node_lats[valid][stride_idx],
               s=0.3, c="#94A3B8", alpha=0.35, zorder=1, label="All nodes",
               linewidths=0)

    bbox_stride = max(1, len(bbox_node_lats) // 5000)
    ax.scatter(bbox_node_lons[::bbox_stride], bbox_node_lats[::bbox_stride],
               s=1.2, c="#B91C1C", alpha=0.55, zorder=2, label="Bbox nodes",
               linewidths=0)

    for i, trk in enumerate(tracks):
        if trk.shape[0] < 2:
            continue
        if i not in near_set:
            ax.plot(trk[:, 1], trk[:, 0], color="#475569", linewidth=0.35,
                    alpha=0.30, zorder=3, solid_capstyle="round")

    for i in storm_indices_near:
        trk = tracks[i]
        if trk.shape[0] < 2:
            continue
        ax.plot(trk[:, 1], trk[:, 0], color="#0F766E", linewidth=0.75,
                alpha=0.80, zorder=4, solid_capstyle="round")

    bx = [bbox["lon_min"], bbox["lon_max"], bbox["lon_max"], bbox["lon_min"], bbox["lon_min"]]
    by = [bbox["lat_min"], bbox["lat_min"], bbox["lat_max"], bbox["lat_max"], bbox["lat_min"]]
    ax.plot(bx, by, color="#B45309", linewidth=2.2, zorder=5, label="Bounding box",
            solid_joinstyle="round")

    circ_lons, circ_lats = _geodesic_circle(medoid_lat, medoid_lon, max_dist_km)
    ax.plot(circ_lons, circ_lats, color="#4C1D95", linewidth=1.6, linestyle=(0, (6, 3)),
            zorder=5, label=f"Radius {max_dist_km:.0f} km")

    ax.plot(medoid_lon, medoid_lat, marker="*", color="#F59E0B", markersize=16,
            zorder=6, markeredgecolor="#1F2937", markeredgewidth=0.8,
            label="Nodes medoid")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"PyStorm-RTCS Bounding-Box Filter — "
        f"{len(bbox_node_lats)} nodes, "
        f"{len(storm_indices_near)} storms (within {max_dist_km:.0f} km)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    out_path = output_dir / filename
    save_figure(fig, out_path, close=True)
    print(f"    Map saved -> {out_path}")
