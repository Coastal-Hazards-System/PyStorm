"""config - pydantic models for the PST job request.

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


class PlotSeriesConfig(BaseModel):
    """Per-series visibility toggles for the hazard-curve plot (all on)."""

    model_config = ConfigDict(frozen=True)

    empirical_below: bool = True   # empirical WPP below the GPD threshold
    empirical_above: bool = True   # empirical WPP above the GPD threshold
    gpd_mean:        bool = True   # GPD best-estimate (mean) curve
    gpd_cl:          bool = True   # GPD 10-90% confidence-limit band
    gpd_threshold:   bool = True   # GPD threshold cross (horizontal + vertical)


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
    # record_length_years None -> auto: n_pot / events_per_year (POT trims to
    # exactly events_per_year × effective_duration peaks, so this inverts to the
    # effective duration). events_per_year must match the POT target rate.
    record_length_years: Optional[float] = Field(default=None, gt=0)
    events_per_year:     float = Field(default=10.0, gt=0)

    # ── Bootstrap / Monte Carlo ───────────────────────────────────────────
    num_simulations: int        = Field(default=1000, gt=0)
    random_seed:     Optional[int] = None
    bootstrap:       BootstrapConfig = Field(default_factory=BootstrapConfig)

    # ── GPD location selection (Quantile Delta Optimization, QDO-WMSE) ────
    # Selection band = empirical PERCENTILES of the POT values (data quantiles),
    # not of the magnitude range - robust to outliers. μ is scanned/selected
    # between these. The count floor (min_exceedances) is the principled upper
    # cap; the ceiling percentile is a secondary, interpretable guardrail.
    threshold_min_percentile: float = Field(default=50.0, ge=0.0, le=100.0)
    threshold_max_percentile: float = Field(default=95.0, ge=0.0, le=100.0)
    n_threshold_candidates:   int   = Field(default=50,   gt=1)
    # Legacy WMSE tolerance - DIAGNOSTIC ONLY. WMSE no longer gates selection
    # (it minimizes in the over-fit sparse tail), so this only affects the
    # reference line on the QDO diagnostics plot.
    wmse_tolerance:           float = Field(default=0.05, ge=0.0)
    # Minimum exceedances a candidate μ must retain to be selectable - guards
    # against over-fitting the sparse tail (absolute-magnitude WMSE → 0 there).
    min_exceedances:          int   = Field(default=30,   ge=1)
    # GPD-location selection method:
    #   "wmse"      (DEFAULT) - WMSE-tolerance set + tie-break (see wmse_tolerance).
    #   "stability" (opt-in)  - flat-ξ threshold-stability plateau (robust
    #                           ξ-dispersion), lower-clip guarded.
    #   "mrl"       (opt-in)  - automated mean-residual-life (Langousis 2016,
    #                           eqs 4-6): lowest in-band threshold where the
    #                           mean-excess curve goes linear.
    #   "gof"       (opt-in)  - Choulakian-Stephens failure-to-reject: lowest
    #                           in-band threshold where the GPD fit is not
    #                           rejected by the A²/W² EDF test (Langousis §2.3).
    gpd_selection:    Literal["wmse", "stability", "mrl", "gof"] = "wmse"
    # GPD fit estimator (selection + hazard ensemble): "mle" (default) or "mom"
    # (method of moments - closed-form, more robust for small/quantized samples).
    gpd_fit_method:   Literal["mle", "mom"] = "mle"
    # GoF-method knobs ("gof" only): EDF statistic and significance level.
    gof_statistic:    Literal["ad", "cvm"] = "ad"
    gof_significance: float = Field(default=0.05, gt=0.0, lt=1.0)
    #   gpd_tiebreak     - arbiter within the chosen set: "stability" (min robust
    #                      ξ-dispersion, ties → lowest μ) or "lowest_mu".
    #   stability_window - ± candidates for the robust (MAD) ξ-dispersion.
    #   stability_tol    - ξ-dispersion tolerance defining the stability plateau.
    gpd_tiebreak:     Literal["stability", "lowest_mu"] = "stability"
    stability_window: int   = Field(default=3, ge=1)
    stability_tol:    float = Field(default=0.02, ge=0.0)

    # ── GPD shape clipping (Luceño bounds, retained from v1) ───────────────
    shape_clip_low:  float = -0.5
    shape_clip_high: float =  0.33

    # ── Plotting ───────────────────────────────────────────────────────────
    y_axis_label: str = "Response Magnitude"
    plot_series:  PlotSeriesConfig = Field(default_factory=PlotSeriesConfig)
    plot_threshold_diagnostics: bool = True   # render the QDO μ-selection plot

    @field_validator("threshold_max_percentile")
    @classmethod
    def _max_above_min(cls, v, info):
        lo = info.data.get("threshold_min_percentile")
        if lo is not None and not (lo < v):
            raise ValueError(
                f"threshold_max_percentile ({v}) must be > threshold_min_percentile ({lo})"
            )
        return v
