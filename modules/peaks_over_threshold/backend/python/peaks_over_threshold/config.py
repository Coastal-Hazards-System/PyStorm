"""config - pydantic models for the POT job request.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Defines the configuration surface for a single Peaks-Over-Threshold extraction
and for the optional upstream NTR-pipeline stages. Validated on construction;
downstream code may assume types and ranges are honoured.

Public API
----------
  POTConfig                  POT extraction run request
  PreprocessConfig           download -> detrend -> ntr pipeline request
  Stage                      Literal of the four selectable stages
"""

from pathlib import Path
from typing  import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The four stages a run may select, in canonical execution order.
Stage = Literal["download", "detrend", "ntr", "pot"]
STAGE_ORDER: Tuple[str, ...] = ("download", "detrend", "ntr", "pot")


class POTConfig(BaseModel):
    """Top-level POT extraction job request.

    A run reads a time-series CSV at ``input_csv`` (columns ``datetime_col``
    and ``value_col``), iterates a percentile threshold upward until the event
    rate matches ``target_events_per_year`` within ``tolerance``, and writes
    the converged peaks CSV + a diagnostic plot.
    """

    model_config = ConfigDict(frozen=True)

    # ── I/O ────────────────────────────────────────────────────────────────
    input_csv:    Path
    output_dir:   Path
    plots_dir:    Path
    datetime_col: str = "Date Time"
    value_col:    str = "Storm Surge"

    # ── Display ────────────────────────────────────────────────────────────
    units:  str = "m"
    vdatum: str = ""

    # ── Event independence ────────────────────────────────────────────────
    interevent_hours: float = Field(default=48.0, gt=0.0)
    method: Literal["hydrograph", "peak_gap"] = "hydrograph"

    # ── Threshold-search target / iteration ───────────────────────────────
    target_events_per_year: float = Field(default=10.0, gt=0.0)
    tolerance:              float = Field(default=0.25, gt=0.0)
    start_percentile:       float = Field(default=75.0, ge=0.0, lt=100.0)
    step_size:              float = Field(default=0.01, gt=0.0)
    max_iter:               Optional[int] = None  # None ⇒ derive from band

    @field_validator("method", mode="before")
    @classmethod
    def _normalize_method(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            # v1 used "hydrograph" or "peaks"; canonicalize to "peak_gap".
            if v == "peaks":
                v = "peak_gap"
        return v


class PreprocessConfig(BaseModel):
    """Optional upstream pipeline that builds the POT input from NOAA data.

    Stages run in canonical order (download -> detrend -> ntr); only those
    listed in ``stages`` execute. Raw NOAA CSVs land in ``raw_dir`` (one folder
    per gauge); the detrended water level and the NTR series land in
    ``processed_dir``. The ``"pot"`` stage, if present, is handled by the
    module orchestrator using ``POTConfig`` and is not configured here.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    # ── Stage selection ────────────────────────────────────────────────────
    stages: List[Stage] = Field(default_factory=lambda: ["pot"])

    # ── Station / paths ────────────────────────────────────────────────────
    station_id:    str
    raw_dir:       Path
    processed_dir: Path
    plots_dir:     Path

    # ── Download (NOAA Tides & Currents) ───────────────────────────────────
    start_year:     int  = Field(default=1900)
    end_year:       int  = Field(default=2025)
    datum:          str  = "MSL"
    time_zone:      str  = "GMT"
    download_units: str  = "metric"      # NOAA API units: "metric" or "english"
    tide_interval:  str  = "h"           # hourly tide predictions (match WL grid)

    # ── Source column names (NOAA defaults) ────────────────────────────────
    datetime_col:   str = "Date Time"
    wl_value_col:   str = "Water Level"
    tide_value_col: str = "Prediction"

    # ── Detrend ────────────────────────────────────────────────────────────
    detrend_method: Literal["midpoint", "ordinary"] = "midpoint"
    # NTDE bounds may be fractional calendar years (e.g. 2012.42).
    ntde_start:     float = 2012
    ntde_end:       float = 2016
    # Override the fitted sea-level slope (value-units/yr, e.g. +0.0048 m/yr).
    # None → fit the slope from the record by least squares (default).
    detrend_slope:  Optional[float] = None

    # ── Display ────────────────────────────────────────────────────────────
    units:  str = "m"
    vdatum: str = "MSL"

    @field_validator("stages", mode="before")
    @classmethod
    def _normalize_stages(cls, v):
        if isinstance(v, str):
            v = [v]
        seen, ordered = set(v), []
        for s in STAGE_ORDER:          # enforce canonical order, drop dups
            if s in seen:
                ordered.append(s)
        return ordered or ["pot"]
