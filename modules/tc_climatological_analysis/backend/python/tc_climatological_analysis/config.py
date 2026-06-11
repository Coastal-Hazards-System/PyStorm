"""Configuration model for the tc_climatological_analysis module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A single validated ``TCAConfig`` carries the operator options from the launcher
to the orchestrator. Paths are coerced to ``Path``; ``basins`` is normalized to
the canonical lower-case basin list. The Pacific is scaffolded but off by default
(no Pacific CRLs are available yet).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator

# Canonical basins. Pacific is supported by the code but disabled by default
# because no Pacific CRL set exists yet.
BASINS = ("atlantic", "pacific")


class TCAConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Which basins to analyze. Both by default now that Pacific CRLs exist.
    basins: List[str] = ["atlantic", "pacific"]

    # I/O layout. Source files live under input_dir/raw; input_dir/processed holds
    # any derived/preprocessed inputs (CyHAN raw-vs-processed convention).
    input_dir: Path = Path("data/inputs")
    output_dir: Path = Path("data/outputs")

    # ── CRL sets (Coastal Reference Locations), one file per basin ─────────────
    # Atlantic CSV (ID,lat,lon) or Pacific tab-delimited (Latitude,Longitude,
    # Region,ID). Paths are absolute or relative to the raw inputs dir
    # (input_dir/raw). None for a basin disables it.
    atlantic_crl_file: Optional[Union[str, Path]] = "CHS_Atl_CRLs_v1.6.csv"
    pacific_crl_file: Optional[Union[str, Path]] = "CHS_PAC_CRLs_v1.2.txt"

    # ── Augmented HURDAT2 source (the augmented_hurricane_database output) ──────
    # Per basin: an explicit CSV path (absolute, or relative to input_dir). None
    # auto-locates the newest augmented_hurdat2_<basin>_*.csv under
    # ``ahd_outputs_dir`` (a link to the sibling ahd module's outputs).
    atlantic_hurdat_file: Optional[Union[str, Path]] = None
    pacific_hurdat_file: Optional[Union[str, Path]] = None
    ahd_outputs_dir: Optional[Union[str, Path]] = None

    # ── GKF / selection parameters (defaults match the CHS MATLAB) ─────────────
    k_size: float = 200.0          # distance Gaussian kernel size (km)
    dir_kernel: float = 30.0       # directional (heading) Gaussian kernel size (deg)
    max_dist: float = 600.0        # storm-selection cutoff distance (km)
    max_cp: float = 1005.0         # drop fixes with central pressure above this (hPa)
    ref_pressure: float = 1013.0   # sea-level reference for the deficit dp = 1013 - Cp
    # First season counted in the rate. None -> the entire HURDAT record (each
    # basin's first season). Otherwise clamped up to the basin's record start.
    start_year: Optional[int] = 1938
    end_year: Optional[int] = None # last season; None -> max year present in the data
    min_dp: float = 8.0            # overall intensity floor (hPa) for the rate
    # Intensity-bin deficit thresholds (hPa): Low [min_dp, dp_low), Med [dp_low,
    # dp_med), High [dp_med, inf).
    dp_low: float = 28.0
    dp_med: float = 48.0

    # ── SRR-within-a-radius variant (SRR_<R>km), off by default ────────────────
    # A second variant of the SRR results only (not DSRR): SRR_<R>km = SRR · (2·R),
    # i.e. the rate (storms/km/yr) times the 2R-km diameter, giving the expected
    # storms / year within ``srr_radius_km`` of each CRL (TC/yr). Written to and
    # plotted in separate ``srr_<R>km`` folders.
    srr_radial: bool = False
    srr_radius_km: float = 200.0

    # ── Per-CRL selected-TC maps (optional, off by default) ────────────────────
    plot_selection: bool = False     # one annual map per CRL
    plot_monthly: bool = False       # one map per CRL and month (Jan-Dec); high volume
    plot_dir: Optional[Union[str, Path]] = None       # None -> output_dir / "plots"
    plot_jobs: Optional[int] = None                   # None/0 -> auto, 1 -> serial
    # Natural Earth basemap resolution for the maps: "10m", "50m", or "110m".
    basemap_resolution: str = "50m"

    @field_validator("basins", mode="before")
    @classmethod
    def _expand_basins(cls, v):
        if isinstance(v, str):
            v = [v]
        out: List[str] = []
        for b in v:
            b = str(b).strip().lower()
            if b == "both":
                out.extend(BASINS)
            else:
                out.append(b)
        seen = set()
        ordered = [b for b in BASINS if b in out and not (b in seen or seen.add(b))]
        unknown = sorted(set(out) - set(BASINS))
        if unknown:
            raise ValueError(f"Unknown basin(s) {unknown}; expected {BASINS} or 'both'.")
        if not ordered:
            raise ValueError("No basins selected.")
        return ordered

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _as_path(cls, v):
        return Path(v)

    @property
    def raw_dir(self) -> Path:
        """Raw source inputs (CRL files, basemap cache)."""
        return Path(self.input_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        """Derived / preprocessed inputs."""
        return Path(self.input_dir) / "processed"

    def crl_file_for(self, basin: str) -> Optional[Union[str, Path]]:
        """Operator-supplied CRL CSV for a basin (None disables the basin)."""
        return {"atlantic": self.atlantic_crl_file,
                "pacific": self.pacific_crl_file}.get(basin)

    def hurdat_file_for(self, basin: str) -> Optional[Union[str, Path]]:
        """Operator-pinned augmented-HURDAT CSV for a basin (None -> auto-locate)."""
        return {"atlantic": self.atlantic_hurdat_file,
                "pacific": self.pacific_hurdat_file}.get(basin)
