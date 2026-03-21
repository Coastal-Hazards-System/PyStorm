"""
backend/geo/track_map.py
=========================
Publication-quality geographic map for the bounding-box filter diagnostic.

Renders a single high-resolution PNG with:
  (a) OSM terrain basemap (fetched via urllib, Mercator-to-geographic warp)
  (b) Bounding-box rectangle
  (c) Full computational mesh nodes (light grey point cloud)
  (d) Nodes within the bounding box (coloured by density)
  (e) All TC tracks (thin grey, low opacity)
  (f) TC tracks within the radial filter (coloured by storm index)
  (g) Radial distance circle around the geographic medoid
  (h) Scale bar and north arrow
  (i) Formatted graticule with degree-minute labels

Uses LineCollection for efficient batch rendering of hundreds of tracks,
serif typography for journal-ready appearance, and 300 DPI output.

Developed by: Norberto C. Nadal-Caraballo, PhD
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
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

_JOURNAL_RC = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
}

# Muted, colour-blind-safe palette
_CLR = {
    "mesh_all":     "#D5D0CB",   # warm grey — full mesh (recedes)
    "mesh_bbox":    "#2A9D8F",   # teal — bbox mesh nodes
    "track_out":    "#A0A0A0",   # cool grey — excluded tracks
    "track_in":     "#4A7FB5",   # steel blue — selected tracks
    "bbox_edge":    "#264653",   # dark navy — bounding-box rectangle
    "radius":       "#264653",   # dark navy — radial circle (unified with bbox)
    "medoid":       "#E76F51",   # coral/vermillion — focal accent
    "medoid_edge":  "#264653",   # dark navy — marker edge
    "scale_bar":    "#264653",
}


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

    OSM tiles are in Web Mercator projection.  Each pixel row is re-sampled
    from Mercator Y to geographic latitude so the basemap aligns with the
    data coordinate system.  Falls back gracefully if network is unavailable.
    """
    try:
        x_min, y_min = _deg2num(lat_max, lon_min, zoom)  # NW corner
        x_max, y_max = _deg2num(lat_min, lon_max, zoom)  # SE corner

        n_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
        if n_tiles > 100:
            print(f"    [basemap] Too many tiles ({n_tiles}) — skipping basemap")
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
                  aspect="auto", zorder=0, alpha=0.45)
    except Exception as e:
        print(f"    [basemap] Could not fetch tiles: {e}")


# ---------------------------------------------------------------------------
# Geodesic circle
# ---------------------------------------------------------------------------

def _geodesic_circle(lat_c, lon_c, radius_km, n_pts=180):
    """Approximate circle on a sphere, returned as (lons, lats) arrays."""
    R = 6_371.0
    az = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    d_lat = (radius_km / R) * np.cos(az)
    d_lon = (radius_km / R) * np.sin(az) / np.cos(np.radians(lat_c))
    return lon_c + np.degrees(d_lon), lat_c + np.degrees(d_lat)


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

def _add_scale_bar(ax, lat_ref, lon_ref, bar_km=100):
    """
    Draw a two-tone scale bar in the lower-right corner.

    The bar is divided into two equal segments (black/white) with a km label.
    """
    R = 6_371.0
    half_km = bar_km / 2.0
    d_lon = np.degrees(half_km / (R * np.cos(np.radians(lat_ref))))

    x0, x1, x2 = lon_ref, lon_ref + d_lon, lon_ref + 2 * d_lon
    y0 = lat_ref
    bar_h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.008

    ax.fill_between([x0, x1], y0, y0 + bar_h,
                     color=_CLR["scale_bar"], zorder=20)
    ax.fill_between([x1, x2], y0, y0 + bar_h,
                     color="white", edgecolor=_CLR["scale_bar"],
                     linewidth=0.5, zorder=20)
    ax.plot([x0, x2, x2, x0, x0],
            [y0, y0, y0 + bar_h, y0 + bar_h, y0],
            color=_CLR["scale_bar"], linewidth=0.5, zorder=20)

    ax.text(x0, y0 - bar_h * 0.5, "0",
            ha="center", va="top", fontsize=7, zorder=20)
    ax.text(x2, y0 - bar_h * 0.5, f"{bar_km} km",
            ha="center", va="top", fontsize=7, zorder=20)


# ---------------------------------------------------------------------------
# North arrow
# ---------------------------------------------------------------------------

def _add_north_arrow(ax, x_frac=0.95, y_frac=0.92, size=0.04):
    """Draw a simple north arrow in axes-fraction coordinates."""
    ax.annotate("N", xy=(x_frac, y_frac + size),
                xycoords="axes fraction",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                zorder=20)
    ax.annotate("", xy=(x_frac, y_frac + size),
                xytext=(x_frac, y_frac),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5, color=_CLR["scale_bar"]),
                zorder=20)


# ---------------------------------------------------------------------------
# Degree-minute tick formatter
# ---------------------------------------------------------------------------

def _format_lon(x, _pos=None):
    """Format longitude as e.g. 90°W or 88.5°W."""
    deg = abs(x)
    if deg == int(deg):
        label = f"{int(deg)}"
    else:
        label = f"{deg:.1f}"
    hemi = "W" if x < 0 else ("E" if x > 0 else "")
    return f"{label}\u00b0{hemi}"


def _format_lat(y, _pos=None):
    """Format latitude as e.g. 30°N or 28.5°N."""
    deg = abs(y)
    if deg == int(deg):
        label = f"{int(deg)}"
    else:
        label = f"{deg:.1f}"
    hemi = "N" if y > 0 else ("S" if y < 0 else "")
    return f"{label}\u00b0{hemi}"


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
    Render a publication-quality geographic map of the bounding-box filter
    and save as a high-resolution PNG (300 DPI).

    Parameters
    ----------
    bbox              : {"lat_min", "lat_max", "lon_min", "lon_max"}
    all_node_lats/lons: coordinates of all store nodes
    bbox_node_lats/lons: coordinates of nodes inside the bbox
    tracks            : list of (N,2) arrays [lat, lon] per storm
    storm_indices_near: 0-based indices of storms passing the radial filter
    medoid_lat/lon    : geographic medoid of bbox mesh nodes (the actual
                        node minimising total distance to all other bbox nodes)
    max_dist_km       : radial filter distance
    output_dir        : directory for the output PNG
    filename          : output filename
    node_stride_map   : subsample all-nodes for plotting (avoid overplotting)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    near_set = set(storm_indices_near.tolist())
    n_total_storms = len(tracks)
    n_near = len(storm_indices_near)

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(10, 8))

        # ── Map extent ──────────────────────────────────────────────────
        pad = 2.5
        all_track_lats = np.concatenate(
            [t[:, 0] for t in tracks if t.shape[0] > 0])
        all_track_lons = np.concatenate(
            [t[:, 1] for t in tracks if t.shape[0] > 0])

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

        # ── (a) Basemap ────────────────────────────────────────────────
        print("    Fetching basemap tiles ...")
        _fetch_basemap(ax, map_lon_min, map_lon_max,
                       map_lat_min, map_lat_max, zoom=6)

        # ── (e) Excluded TC tracks — LineCollection for efficiency ────
        segs_out = []
        for i, trk in enumerate(tracks):
            if trk.shape[0] < 2 or i in near_set:
                continue
            segs_out.append(np.column_stack([trk[:, 1], trk[:, 0]]))
        if segs_out:
            lc_out = LineCollection(
                segs_out, colors=_CLR["track_out"],
                linewidths=0.25, alpha=0.35, zorder=1)
            ax.add_collection(lc_out)

        # ── (f) Selected TC tracks — LineCollection, coloured by index ─
        segs_in = []
        for i in storm_indices_near:
            trk = tracks[i]
            if trk.shape[0] < 2:
                continue
            segs_in.append(np.column_stack([trk[:, 1], trk[:, 0]]))
        if segs_in:
            # Uniform steel blue for all selected tracks
            colors_in = [_CLR["track_in"]] * len(segs_in)
            lc_in = LineCollection(
                segs_in, colors=colors_in,
                linewidths=0.5, alpha=0.65, zorder=2)
            ax.add_collection(lc_in)

        # ── (c) All mesh nodes (light grey, subsampled) ───────────────
        valid = ~np.isnan(all_node_lats)
        stride_idx = np.arange(0, valid.sum(), max(1, node_stride_map))
        ax.scatter(
            all_node_lons[valid][stride_idx],
            all_node_lats[valid][stride_idx],
            s=0.15, c=_CLR["mesh_all"], alpha=0.25, zorder=3,
            rasterized=True, edgecolors="none")

        # ── (d) Bbox mesh nodes ──────────────────────────────────────
        bbox_stride = max(1, len(bbox_node_lats) // 8000)
        ax.scatter(
            bbox_node_lons[::bbox_stride],
            bbox_node_lats[::bbox_stride],
            s=0.4, c=_CLR["mesh_bbox"], alpha=0.5, zorder=4,
            rasterized=True, edgecolors="none")

        # ── (b) Bounding-box rectangle ───────────────────────────────
        rect = mpatches.FancyBboxPatch(
            (bbox["lon_min"], bbox["lat_min"]),
            bbox["lon_max"] - bbox["lon_min"],
            bbox["lat_max"] - bbox["lat_min"],
            boxstyle="square,pad=0",
            linewidth=1.8, edgecolor=_CLR["bbox_edge"],
            facecolor="none", zorder=5)
        ax.add_patch(rect)

        # ── (g) Radial distance circle ───────────────────────────────
        circ_lons, circ_lats = _geodesic_circle(
            medoid_lat, medoid_lon, max_dist_km)
        ax.plot(circ_lons, circ_lats,
                color=_CLR["radius"], linewidth=1.4, linestyle=(0, (5, 3)),
                zorder=5)

        # ── Medoid marker ────────────────────────────────────────────
        ax.plot(medoid_lon, medoid_lat, marker="*",
                color=_CLR["medoid"], markersize=12,
                markeredgecolor=_CLR["medoid_edge"], markeredgewidth=0.6,
                zorder=6)

        # ── Graticule ────────────────────────────────────────────────
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_lon))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_lat))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.grid(True, linewidth=0.3, alpha=0.4, color="#666666",
                linestyle=":")
        ax.set_aspect("equal")

        # ── Scale bar & north arrow ──────────────────────────────────
        sb_lat = map_lat_min + (map_lat_max - map_lat_min) * 0.04
        sb_lon = map_lon_max - (map_lon_max - map_lon_min) * 0.22
        _add_scale_bar(ax, sb_lat, sb_lon, bar_km=200)
        _add_north_arrow(ax, x_frac=0.96, y_frac=0.88)

        # ── Labels ───────────────────────────────────────────────────
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # ── Legend (outside plot, below) ─────────────────────────────
        legend_handles = [
            plt.scatter([], [], s=8, color=_CLR["mesh_all"],
                        edgecolors="none",
                        label=f"Computational mesh ({len(all_node_lats):,} nodes)"),
            plt.scatter([], [], s=12, color=_CLR["mesh_bbox"],
                        edgecolors="none",
                        label=f"Bounding-box nodes ({len(bbox_node_lats):,})"),
            plt.Line2D([], [], color=_CLR["track_out"], linewidth=0.6,
                       alpha=0.5,
                       label=f"TC tracks — excluded "
                             f"({n_total_storms - n_near})"),
            plt.Line2D([], [], color=_CLR["track_in"], linewidth=1.0,
                       label=f"TC tracks — within "
                             f"{max_dist_km:.0f} km ({n_near})"),
            plt.Line2D([], [], color=_CLR["bbox_edge"], linewidth=1.8,
                       label="Bounding box"),
            plt.Line2D([], [], color=_CLR["radius"], linewidth=1.4,
                       linestyle=(0, (5, 3)),
                       label=f"Radial filter ({max_dist_km:.0f} km)"),
            plt.Line2D([], [], color=_CLR["medoid"], marker="*",
                       markersize=10, markeredgecolor=_CLR["medoid_edge"],
                       markeredgewidth=0.6, linestyle="none",
                       label="Bounding-box node medoid"),
        ]
        ax.legend(
            handles=legend_handles, loc="upper left",
            fontsize=8, framealpha=0.85, edgecolor="#999999",
            borderpad=0.6, labelspacing=0.45,
            handletextpad=0.5, handlelength=2.0)

        # ── Title ────────────────────────────────────────────────────
        ax.set_title(
            f"Geographic Bounding-Box Filter\n"
            f"{len(bbox_node_lats):,} nodes, "
            f"{n_near}/{n_total_storms} TC tracks selected "
            f"(radius = {max_dist_km:.0f} km)",
            fontsize=11, fontweight="bold", pad=10)

        # ── Save ─────────────────────────────────────────────────────
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"    Map saved -> {out_path}")
