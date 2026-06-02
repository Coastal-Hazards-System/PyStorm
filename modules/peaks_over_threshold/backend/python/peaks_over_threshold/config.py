"""config — pydantic models for the POT job request.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Defines the configuration surface for a single Peaks-Over-Threshold extraction.
Validated on construction; downstream code may assume types and ranges are
honoured.

Public API
----------
  POTConfig                  top-level POT run request
"""

from pathlib import Path
from typing  import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
