"""
backend/geo/bbox_filter.py
===========================
Bounding-box node/storm filtering for regional RTCS workflows.

Provides:
  - load_node_coordinates   : read ADCIRC node lat/lon from the probQ .mat file
  - filter_nodes_in_bbox    : return column indices of Y whose nodes fall inside a bbox
  - load_tc_tracks          : read ITCS TROP .txt track files → list of (lat, lon) arrays
  - filter_storms_near_point: return 0-based storm indices whose track passes within a
                              radial distance of a given point
  - apply_bbox_filter       : high-level orchestrator called from the CLI run scripts

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from backend.io.readers import load_array


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between a single point and array(s) of points."""
    R = 6_371.0  # Earth mean radius (km)
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Node coordinates
# ---------------------------------------------------------------------------

def load_node_coordinates(
    mat_path: str | Path,
    variable: str = "nodeID",
    node_id_col: int = 0,
    lat_col: int = 2,
    lon_col: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load node coordinates from the nodeID .mat file.

    Parameters
    ----------
    node_id_col : int
        Column containing the node IDs to use for lookups.
        Default 0 = main node IDs (must match IDs stored in HDF5).

    Returns
    -------
    node_ids : int64 array [N]
    lats     : float64 array [N]
    lons     : float64 array [N]
    """
    arr, _, _ = load_array(Path(mat_path), varname=variable)
    node_ids = arr[:, node_id_col].astype(np.int64)
    lats = arr[:, lat_col]
    lons = arr[:, lon_col]
    return node_ids, lats, lons


# ---------------------------------------------------------------------------
# Bounding-box node filter
# ---------------------------------------------------------------------------

def filter_nodes_in_bbox(
    store_node_ids: list[str],
    all_node_ids: np.ndarray,
    all_lats: np.ndarray,
    all_lons: np.ndarray,
    bbox: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Y-column indices of nodes that fall within the bounding box.

    Parameters
    ----------
    store_node_ids : node IDs from the HDF5 store (strings, main node IDs)
    all_node_ids   : full coordinate table node IDs (int64, same ID scheme as store)
    all_lats, all_lons : coordinate arrays matching all_node_ids
    bbox : dict with keys "lat_min", "lat_max", "lon_min", "lon_max"

    Returns
    -------
    col_indices  : 0-based column indices into Y / HC  (sorted)
    bbox_lats    : latitudes of kept nodes
    bbox_lons    : longitudes of kept nodes
    """
    # Build lookup:  1-based node ID → (lat, lon)
    id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    store_ids_int = np.array([int(s) for s in store_node_ids], dtype=np.int64)

    col_indices = []
    kept_lats = []
    kept_lons = []

    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    for col_j, nid in enumerate(store_ids_int):
        pos = id_to_idx.get(nid)
        if pos is None:
            continue
        lat, lon = all_lats[pos], all_lons[pos]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            col_indices.append(col_j)
            kept_lats.append(lat)
            kept_lons.append(lon)

    col_indices = np.array(col_indices, dtype=int)
    order = np.argsort(col_indices)
    return col_indices[order], np.array(kept_lats)[order], np.array(kept_lons)[order]


def compute_geographic_medoid(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Return the (lat, lon) of the point that minimises total haversine distance."""
    best_dist = np.inf
    best_i = 0
    for i in range(len(lats)):
        d = haversine_km(lats[i], lons[i], lats, lons).sum()
        if d < best_dist:
            best_dist = d
            best_i = i
    return float(lats[best_i]), float(lons[best_i])


# ---------------------------------------------------------------------------
# TC track loader
# ---------------------------------------------------------------------------

def load_tc_tracks(
    track_dir: str | Path,
    n_storms: int,
    file_pattern: str = "LACPR2_JPM{:04d}_TROP.txt",
) -> list[np.ndarray]:
    """
    Read ITCS TROP .txt files.  One file per storm, 1-indexed.

    Returns list of (N_pts, 2) arrays — each row is [lat, lon].
    Missing files yield an empty (0, 2) array at that index.
    """
    track_dir = Path(track_dir)
    tracks: list[np.ndarray] = []

    for storm_1based in range(1, n_storms + 1):
        fname = track_dir / file_pattern.format(storm_1based)
        if not fname.exists():
            tracks.append(np.empty((0, 2), dtype=float))
            continue
        try:
            pts = []
            with open(fname, "r") as fh:
                header = fh.readline()  # skip header
                for line in fh:
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    lat = float(parts[2].strip())
                    lon = float(parts[3].strip())
                    pts.append((lat, lon))
            tracks.append(np.array(pts, dtype=float) if pts else np.empty((0, 2)))
        except Exception:
            tracks.append(np.empty((0, 2), dtype=float))

    return tracks


# ---------------------------------------------------------------------------
# Storm-distance filter
# ---------------------------------------------------------------------------

def filter_storms_near_point(
    tracks: list[np.ndarray],
    center_lat: float,
    center_lon: float,
    max_dist_km: float = 200.0,
) -> np.ndarray:
    """
    Return 0-based storm indices whose track has at least one point
    within *max_dist_km* of (center_lat, center_lon).
    """
    keep = []
    for i, trk in enumerate(tracks):
        if trk.shape[0] == 0:
            continue
        dists = haversine_km(center_lat, center_lon, trk[:, 0], trk[:, 1])
        if dists.min() <= max_dist_km:
            keep.append(i)
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# All-node coordinate lookup  (for map)
# ---------------------------------------------------------------------------

def get_store_node_coords(
    store_node_ids: list[str],
    all_node_ids: np.ndarray,
    all_lats: np.ndarray,
    all_lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) arrays aligned with store_node_ids order."""
    id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    lats = np.empty(len(store_node_ids))
    lons = np.empty(len(store_node_ids))
    for j, sid in enumerate(store_node_ids):
        pos = id_to_idx.get(int(sid))
        if pos is not None:
            lats[j] = all_lats[pos]
            lons[j] = all_lons[pos]
        else:
            lats[j] = lons[j] = np.nan
    return lats, lons


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def apply_bbox_filter(
    bbox_cfg: dict,
    h5_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Run the full bounding-box filter pipeline.

    Parameters (bbox_cfg keys)
    --------------------------
    bbox          : dict  {"lat_min", "lat_max", "lon_min", "lon_max"}
    node_coord_source   : path to .mat with node coordinates (probQ)
    node_coord_variable : variable name in .mat  (default "nodeID")
    track_dir           : path to ITCS TROP track directory
    track_file_pattern  : filename pattern  (default "LACPR2_JPM{:04d}_TROP.txt")
    max_track_dist_km   : radial distance for storm filtering  (default 200)

    Returns
    -------
    dict with keys:
      "node_col_indices" : int array  — Y/HC column indices to keep
      "storm_indices"    : int array  — 0-based storm rows to keep
      "medoid_lat"       : float
      "medoid_lon"       : float
      "bbox_node_lats"   : float array
      "bbox_node_lons"   : float array
      "all_node_lats"    : float array  (store nodes)
      "all_node_lons"    : float array  (store nodes)
      "tracks"           : list of (N,2) arrays
      "all_tracks_in_bbox" : int array  (storm indices near medoid)
    """
    from backend.io.store import read_store

    print("\n[0] Bounding-box filter ...")
    bbox = bbox_cfg["bbox"]

    # Load node coordinates from full probQ file
    print(f"    Loading node coordinates from {bbox_cfg['node_coord_source']} ...")
    all_node_ids, all_lats, all_lons = load_node_coordinates(
        bbox_cfg["node_coord_source"],
        bbox_cfg.get("node_coord_variable", "nodeID"),
        bbox_cfg.get("node_id_col", 0),
        bbox_cfg.get("lat_col", 2),
        bbox_cfg.get("lon_col", 3),
    )
    print(f"    Full coordinate table : {len(all_node_ids)} nodes")

    # Get store node IDs
    data = read_store(Path(h5_path))
    store_node_ids = data.node_ids
    n_storms = data.X.shape[0]

    # Coordinates for ALL store nodes (for mapping)
    all_store_lats, all_store_lons = get_store_node_coords(
        store_node_ids, all_node_ids, all_lats, all_lons)

    # Filter nodes in bbox
    print(f"    Bounding box : lat [{bbox['lat_min']:.4f}, {bbox['lat_max']:.4f}]"
          f"  lon [{bbox['lon_min']:.4f}, {bbox['lon_max']:.4f}]")
    node_col_indices, bbox_lats, bbox_lons = filter_nodes_in_bbox(
        store_node_ids, all_node_ids, all_lats, all_lons, bbox)
    print(f"    Nodes in bbox : {len(node_col_indices)} / {len(store_node_ids)}")

    if len(node_col_indices) == 0:
        raise ValueError("No nodes found within the bounding box. "
                         "Check bbox coordinates.")

    # Compute geographic medoid of bbox nodes (an actual node).
    # Subsample to keep it fast (O(n_sample * n) instead of O(n^2)).
    n_bbox = len(bbox_lats)
    max_sample = 2000
    if n_bbox <= max_sample:
        sample_idx = np.arange(n_bbox)
    else:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_bbox, size=max_sample, replace=False)
    best_dist = np.inf
    best_i = 0
    for i in sample_idx:
        d = haversine_km(bbox_lats[i], bbox_lons[i], bbox_lats, bbox_lons).sum()
        if d < best_dist:
            best_dist = d
            best_i = i
    medoid_lat = float(bbox_lats[best_i])
    medoid_lon = float(bbox_lons[best_i])
    print(f"    Nodes medoid   : ({medoid_lat:.4f}, {medoid_lon:.4f})")

    # Load TC tracks
    track_dir = bbox_cfg["track_dir"]
    pattern = bbox_cfg.get("track_file_pattern", "LACPR2_JPM{:04d}_TROP.txt")
    print(f"    Loading TC tracks from {track_dir} ...")
    tracks = load_tc_tracks(track_dir, n_storms, pattern)
    loaded = sum(1 for t in tracks if t.shape[0] > 0)
    print(f"    Tracks loaded : {loaded} / {n_storms}")

    # Filter storms near medoid
    max_dist = bbox_cfg.get("max_track_dist_km", 200.0)
    storm_indices = filter_storms_near_point(tracks, medoid_lat, medoid_lon, max_dist)
    print(f"    Storms within {max_dist:.0f} km of medoid : "
          f"{len(storm_indices)} / {n_storms}")

    if len(storm_indices) == 0:
        raise ValueError(
            f"No storms found within {max_dist} km of bbox medoid. "
            "Increase max_track_dist_km or check bbox coordinates.")

    return {
        "node_col_indices": node_col_indices,
        "storm_indices":    storm_indices,
        "medoid_lat":       medoid_lat,
        "medoid_lon":       medoid_lon,
        "bbox_node_lats":   bbox_lats,
        "bbox_node_lons":   bbox_lons,
        "all_node_lats":    all_store_lats,
        "all_node_lons":    all_store_lons,
        "tracks":           tracks,
        "n_storms_total":   n_storms,
    }
