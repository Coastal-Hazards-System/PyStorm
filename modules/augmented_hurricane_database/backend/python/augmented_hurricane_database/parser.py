"""HURDAT2 parser - best-track records into tidy track points.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Object-oriented re-implementation of the CHS hurricane track pre-processor
(``CHS_TC_HURDAT_Atlantic.m``). Reads an NHC HURDAT2 best-track text file into
``Storm`` objects, each holding a list of ``TrackPoint`` snapshots, then derives
the two motion columns the house format adds on top of raw HURDAT2:

  * trans_kmh   - translation (forward) speed, km/h
  * heading_deg - heading (forward azimuth), degrees in (-180, 180]

Unit conventions (matching the legacy MATLAB house format):
  * maximum sustained wind : knots -> km/h        (x 1.852)
  * wind-radii / Rmax      : nautical miles -> km  (x 1.852)
  * minimum central pressure : hPa (unchanged)

HURDAT2 sentinels (-99 for wind, -999 for pressure/radii) become NaN. As in the
MATLAB reference, when the rounded translation speed is 0 km/h both speed and
heading are set to NaN (a stationary snapshot has no defined heading).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from pyproj import Geod

# WGS-84 ellipsoid for geodesic distance/azimuth (matches MATLAB wgs84Ellipsoid).
_WGS84 = Geod(ellps="WGS84")

_KT2KMH = 1.852   # knots -> km/h
_NM2KM  = 1.852   # nautical miles -> km

# NHC two-letter basin prefixes embedded in the cyclone id (e.g. AL092021).
_BASIN_OF_PREFIX = {"AL": "atlantic", "EP": "pacific", "CP": "pacific"}

# Record-identifier and status code maps (HURDAT2 column 3 and 4).
_LANDFLAG = {"C": 1, "G": 2, "I": 3, "L": 4, "P": 5,
             "R": 6, "S": 7, "T": 8, "W": 9}
_STATUS   = {"TD": 1, "TS": 2, "HU": 3, "EX": 4, "SD": 5,
             "SS": 6, "LO": 7, "WV": 8, "DB": 9}


def latlon_str_to_float(token: str) -> float:
    """Convert a HURDAT2 lat/lon token to a signed float.

    '28.0N' -> 28.0, '95.4W' -> -95.4 (with the same <-180 -> +360 wrap the
    MATLAB reference applies, keeping west-Pacific tracks contiguous).
    """
    t = token.strip().upper()
    if t.endswith("N"):
        return float(t[:-1])
    if t.endswith("S"):
        return -float(t[:-1])
    if t.endswith("E"):
        return float(t[:-1])
    if t.endswith("W"):
        val = -float(t[:-1])
        return val + 360 if val < -180 else val
    raise ValueError(f"Cannot parse lat/lon token '{token}'")


def parse_datetime(ymd: int, hhmm: int) -> dt.datetime:
    """Build a UTC datetime from HURDAT2's YYYYMMDD and HHMM integer codes."""
    year, month, day = ymd // 10000, (ymd // 100) % 100, ymd % 100
    hour, minute = hhmm // 100, hhmm % 100
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)


@dataclass
class TrackPoint:
    ymd: int                       # YYYYMMDD
    hhmm: int                      # HHMM (UTC)
    dt_utc: dt.datetime            # exact timestamp
    lat: float
    lon: float
    landflag: int
    status: int
    vmax_kmh: Optional[float]
    pmin_hpa: Optional[float]
    trans_kmh: Optional[float] = None
    heading_deg: Optional[float] = None
    radii34_ne_km: Optional[float] = None
    radii34_se_km: Optional[float] = None
    radii34_sw_km: Optional[float] = None
    radii34_nw_km: Optional[float] = None
    radii50_ne_km: Optional[float] = None
    radii50_se_km: Optional[float] = None
    radii50_sw_km: Optional[float] = None
    radii50_nw_km: Optional[float] = None
    radii64_ne_km: Optional[float] = None
    radii64_se_km: Optional[float] = None
    radii64_sw_km: Optional[float] = None
    radii64_nw_km: Optional[float] = None
    rmax_km: Optional[float] = None


@dataclass
class Storm:
    nhc_id: str
    year: int
    number: int
    name: str
    basin: str
    track: List[TrackPoint] = field(default_factory=list)

    def compute_motion(self) -> None:
        """Fill trans_kmh and heading_deg for every track point.

        Each point i (i >= 1) gets the geodesic speed/azimuth of the segment
        ending at it (i-1 -> i); the first point is forward-filled from the
        second, matching the MATLAB reference. A rounded speed of 0 km/h yields
        NaN speed and heading (no defined heading when stationary).
        """
        n = len(self.track)
        if n < 2:
            if n == 1:
                self.track[0].trans_kmh = np.nan
                self.track[0].heading_deg = np.nan
            return

        lats = np.array([pt.lat for pt in self.track])
        lons = np.array([pt.lon for pt in self.track])

        # Geodesic inverse over consecutive points: (fwd_az, back_az, dist_m).
        fwd_az, _, dist_m = _WGS84.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        dists_km = dist_m * 1e-3

        times = np.array([pt.dt_utc.timestamp() for pt in self.track])
        dt_h = np.diff(times) / 3600.0
        # Guard zero/negative dt (duplicate timestamps) -> NaN rather than inf.
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_speed = np.where(dt_h > 0, dists_km / dt_h, np.nan)

        speeds = np.insert(seg_speed, 0, seg_speed[0])      # forward-fill point 0
        fwd_az = np.where(fwd_az > 180, fwd_az - 360, fwd_az)  # -> (-180, 180]
        headings = np.insert(fwd_az, 0, fwd_az[0])

        for pt, spd, hdg in zip(self.track, speeds, headings):
            spd_r = round(spd) if np.isfinite(spd) else np.nan
            if spd_r == 0:                                   # stationary -> undefined
                pt.trans_kmh = np.nan
                pt.heading_deg = np.nan
            else:
                pt.trans_kmh = spd_r
                pt.heading_deg = round(hdg) if np.isfinite(hdg) else np.nan


# Canonical output column order (HURDAT-like, house format).
COLUMNS = [
    "tc_no", "snap_no", "year", "nhc_id", "basin", "name",
    "ymd", "hhmm", "time_utc", "lat", "lon", "landflag", "status",
    "vmax_kmh", "pmin_hpa", "trans_kmh", "heading_deg",
    "radii34_ne_km", "radii34_se_km", "radii34_sw_km", "radii34_nw_km",
    "radii50_ne_km", "radii50_se_km", "radii50_sw_km", "radii50_nw_km",
    "radii64_ne_km", "radii64_se_km", "radii64_sw_km", "radii64_nw_km",
    "rmax_km",
]


class HURDAT2:
    """Reader for a single NHC HURDAT2 best-track text file."""

    def __init__(self, filepath: Union[str, Path], basin: Optional[str] = None) -> None:
        self.path = Path(filepath)
        if not self.path.exists():
            raise FileNotFoundError(f"HURDAT2 file not found: {self.path}")
        # If the caller does not tell us the basin, infer it per storm from the
        # cyclone-id prefix (AL/EP/CP).
        self.basin = basin

    @staticmethod
    def _num(cols: List[str], idx: int, bad: int, scale: float = 1.0) -> float:
        """Parse data column ``idx`` to float; HURDAT2 sentinel -> NaN.

        Scaled values (the knots->km/h and nm->km conversions, scale != 1) are
        rounded to whole units to match the MATLAB house format. Tolerates short
        rows (older format revisions without the Rmax column) by returning NaN
        when the column is absent.
        """
        if idx >= len(cols) or cols[idx] == "":
            return np.nan
        val = int(cols[idx])
        if val == bad:
            return np.nan
        return float(round(val * scale)) if scale != 1.0 else float(val)

    def storms(self) -> Iterator[Storm]:
        with self.path.open("r", encoding="utf-8") as fh:
            lines = (ln.rstrip() for ln in fh)
            for header in lines:
                if not header.strip():
                    continue
                parts = [p.strip() for p in header.split(",")]
                nhc_id, rawname, nrows = parts[0], parts[1], int(parts[2])
                name = rawname.replace(" ", "") or "UNNAMED"
                basin = self.basin or _BASIN_OF_PREFIX.get(nhc_id[:2].upper(), "unknown")
                storm = Storm(
                    nhc_id=nhc_id,
                    year=int(nhc_id[4:8]),
                    number=int(nhc_id[2:4]),
                    name=name,
                    basin=basin,
                )

                for _ in range(nrows):
                    cols = [c.strip() for c in next(lines).split(",")]
                    ymd, hhmm = int(cols[0]), int(cols[1])
                    storm.track.append(TrackPoint(
                        ymd=ymd,
                        hhmm=hhmm,
                        dt_utc=parse_datetime(ymd, hhmm),
                        lat=latlon_str_to_float(cols[4]),
                        lon=latlon_str_to_float(cols[5]),
                        landflag=_LANDFLAG.get(cols[2], 0),
                        status=_STATUS.get(cols[3], 0),
                        vmax_kmh=self._num(cols, 6, -99, _KT2KMH),
                        pmin_hpa=self._num(cols, 7, -999),
                        radii34_ne_km=self._num(cols, 8,  -999, _NM2KM),
                        radii34_se_km=self._num(cols, 9,  -999, _NM2KM),
                        radii34_sw_km=self._num(cols, 10, -999, _NM2KM),
                        radii34_nw_km=self._num(cols, 11, -999, _NM2KM),
                        radii50_ne_km=self._num(cols, 12, -999, _NM2KM),
                        radii50_se_km=self._num(cols, 13, -999, _NM2KM),
                        radii50_sw_km=self._num(cols, 14, -999, _NM2KM),
                        radii50_nw_km=self._num(cols, 15, -999, _NM2KM),
                        radii64_ne_km=self._num(cols, 16, -999, _NM2KM),
                        radii64_se_km=self._num(cols, 17, -999, _NM2KM),
                        radii64_sw_km=self._num(cols, 18, -999, _NM2KM),
                        radii64_nw_km=self._num(cols, 19, -999, _NM2KM),
                        rmax_km=self._num(cols, 20, -999, _NM2KM),
                    ))

                storm.compute_motion()
                yield storm

    def to_dataframe(self) -> pd.DataFrame:
        """Parse the whole file into a tidy, HURDAT-like DataFrame."""
        rows = []
        for tc_no, storm in enumerate(self.storms(), start=1):
            for snap, pt in enumerate(storm.track, start=1):
                rec = asdict(pt)
                rec.update({
                    "tc_no": tc_no,
                    "snap_no": snap,
                    "year": storm.year,
                    "nhc_id": storm.nhc_id,
                    "basin": storm.basin,
                    "name": storm.name,
                    "time_utc": pt.dt_utc,
                })
                rows.append(rec)
        return pd.DataFrame(rows, columns=COLUMNS)
