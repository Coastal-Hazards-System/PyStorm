"""config - configuration model for the coastal_storm_hydrograph (CSH) module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A single validated ``CSHConfig`` carries the operator options from the launcher to
the orchestrator. Paths are coerced to ``Path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator


class CSHConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # ── Mode (which hydrograph to build) ───────────────────────────────────────
    # "surge"      : storm-surge (water-level) hydrograph. The implemented method.
    # "wave"       : wave-height (Hs) hydrograph. PLACEHOLDER (not yet implemented).
    # "surge_wave" : joint surge + wave, evaluated synoptically with the lag between
    #                the surge peak and the wave peak. PLACEHOLDER (not yet implemented).
    mode: str = "surge"

    # ── I/O layout (CyHAN raw / processed / outputs convention) ────────────────
    input_dir: Path = Path("data/inputs")
    output_dir: Path = Path("data/outputs")

    # Save-point metadata: id, lat, lon, depth (m, positive DOWN; ground elevation
    # above NAVD88 = -depth). Relative to the raw inputs dir unless absolute.
    staid_file: Union[str, Path] = "CTXCS_staID.csv"
    # Per-save-point surge matrices (cols = synthetic TCs, rows = 15-min steps,
    # m NAVD88). ``{sp}`` is replaced by the 5-digit save-point id (e.g. 03911).
    surge_file_glob: str = "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv"
    # Matching timestamp matrix (one column per TC). Used only to confirm the step.
    time_file: Union[str, Path] = "CTXCS_TP_SYN_Tides_0_SLC_0_time.csv"
    # Restrict to these save-point ids (ints); None -> every SP found by the glob.
    save_points: Optional[List[int]] = None

    # ── Data semantics ─────────────────────────────────────────────────────────
    dt_hours: float = 0.25          # time step (15 min)
    dry_value: float = -99999.0     # sentinel: point is dry (water below ground)
    # The staID depth column is positive-DOWN, so ground elevation (m NAVD88) is
    # -depth. Set False if a future file already supplies ground elevation.
    depth_is_positive_down: bool = True

    # ── Unit-hydrograph construction ───────────────────────────────────────────
    # Shape model. "double_norm" (default): canonical shape over dimensionless time
    # s = tau/D, reconstructed from two parameters (peak, duration); the most accurate
    # and parameter-efficient model in the whitepaper comparison. "amplitude": legacy
    # peak-only shape over physical time (leaves duration variability; baseline only).
    method: str = "double_norm"
    min_wet_samples: int = 5        # skip a storm with fewer wet (above-ground) samples
    # Half-width (h) of the peak-aligned day-of-storm window. None -> auto from the
    # data (max wet half-extent), capped at ``max_window_hours``.
    window_hours: Optional[float] = None
    max_window_hours: float = 72.0
    aggregate: str = "mean"         # "mean" or "median" across a point's storms

    # ── Parametric limb fit (separate rising / falling curves) ─────────────────
    parametric: bool = True         # fit generalized-Gaussian rising/falling limbs

    # ── Actual duration (time above a physical threshold) ──────────────────────
    # Threshold elevation z0 = max(ground, MHHW) + offset: "offset above ground" for
    # overland points and "offset above MHHW" for overwater points. The actual duration
    # (time the surge exceeds z0) converts to the equivalent width via the canonical
    # level-width function. MHHW is in m NAVD88; None -> all points treated as overland
    # (no MHHW available, as in the no-tide CTXS run). Per-point MHHW is a future input.
    actual_duration_offset_m: float = 0.30
    mhhw_navd88: Optional[float] = None

    # ── Scaled-hydrograph examples ─────────────────────────────────────────────
    # Reconstruct example hydrographs at each point for target PEAK elevations
    # (m NAVD88). "auto" -> the point's observed median and max peak. A list of
    # floats applies the same peaks to every point. None -> none written.
    scale_peaks: Optional[Union[str, List[float]]] = "auto"

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots: bool = True

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def _as_path(cls, v):
        return Path(v)

    @field_validator("mode", mode="before")
    @classmethod
    def _mode(cls, v):
        v = str(v).strip().lower()
        if v not in ("surge", "wave", "surge_wave"):
            raise ValueError("mode must be 'surge', 'wave', or 'surge_wave'")
        return v

    @field_validator("aggregate", mode="before")
    @classmethod
    def _agg(cls, v):
        v = str(v).strip().lower()
        if v not in ("mean", "median"):
            raise ValueError("aggregate must be 'mean' or 'median'")
        return v

    @field_validator("method", mode="before")
    @classmethod
    def _method(cls, v):
        v = str(v).strip().lower()
        if v not in ("double_norm", "amplitude"):
            raise ValueError("method must be 'double_norm' or 'amplitude'")
        return v

    @property
    def raw_dir(self) -> Path:
        return Path(self.input_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.input_dir) / "processed"
