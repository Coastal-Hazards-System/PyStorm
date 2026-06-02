"""config — pydantic models for the PST job request.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Defines the configuration surface for a single PST run. Values are validated
on construction; downstream code may assume types and ranges are honoured.

Public API
----------
  BootstrapConfig            inner config block for the bootstrap stage
  PSTConfig                  top-level run request

Both models are immutable (frozen) after construction.
"""

from pathlib import Path
from typing  import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BootstrapConfig(BaseModel):
    """Parameters for the truncated-noise bootstrap of POT exceedances."""

    model_config = ConfigDict(frozen=True)

    distribution: Literal["gaussian", "uniform"] = "gaussian"
    truncation:   Tuple[float, float]            = (-1.0, 1.0)

    @field_validator("truncation")
    @classmethod
    def _ordered_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if not (lo < hi):
            raise ValueError(f"truncation lower bound must be < upper bound; got ({lo}, {hi})")
        return v


class PSTConfig(BaseModel):
    """Top-level PST job request.

    A run consumes a POT CSV at ``input_csv`` (column ``storm_column``) and
    writes the ensemble + hazard-curve tables to ``output_dir`` and the
    hazard-curve plot to ``plots_dir``.
    """

    model_config = ConfigDict(frozen=True)

    # ── I/O ────────────────────────────────────────────────────────────────
    input_csv:   Path
    output_dir:  Path
    plots_dir:   Path
    storm_column: str = "value"

    # ── Population statistics ──────────────────────────────────────────────
    record_length_years: float = Field(gt=0)

    # ── Bootstrap / Monte Carlo ───────────────────────────────────────────
    num_simulations: int        = Field(default=1000, gt=0)
    random_seed:     Optional[int] = None
    bootstrap:       BootstrapConfig = Field(default_factory=BootstrapConfig)

    # ── GPD threshold selection (Quantile-Delta-Method WMSE search) ───────
    threshold_min_percentile: float = Field(default=20.0, ge=0.0, le=100.0)
    threshold_max_percentile: float = Field(default=80.0, ge=0.0, le=100.0)
    n_threshold_candidates:   int   = Field(default=50,   gt=1)

    # ── GPD shape clipping (Luceño bounds, retained from v1) ───────────────
    shape_clip_low:  float = -0.5
    shape_clip_high: float =  0.33

    # ── Plotting ───────────────────────────────────────────────────────────
    y_axis_label: str = "Response Magnitude"

    @field_validator("threshold_max_percentile")
    @classmethod
    def _max_above_min(cls, v, info):
        lo = info.data.get("threshold_min_percentile")
        if lo is not None and not (lo < v):
            raise ValueError(
                f"threshold_max_percentile ({v}) must be > threshold_min_percentile ({lo})"
            )
        return v
