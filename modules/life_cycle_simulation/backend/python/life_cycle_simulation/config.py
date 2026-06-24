"""config - configuration model for the life_cycle_simulation module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A single validated ``LCSConfig`` carries the operator options from the launcher
to the orchestrator. Paths are coerced to ``Path``; ``crl_ids`` is normalized to
a list of ints; ``day_method`` is lower-cased and checked. Only tropical cyclones
(``storm_type='tc'``) are implemented; ``etc`` is a scaffolded placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Intensity strata, in the SRR file's column order. "all" is the union; "low",
# "med", "high" partition it (their annual rates sum to the "all" rate).
STRATA = ("low", "med", "high")

# Visualization-suite figure keys, in display order. Defined here (not in plots.py)
# so config validation stays free of the matplotlib import chain. "all" expands to
# this tuple; see life_cycle_simulation.plots for what each figure shows.
PLOT_KEYS = ("annual_fan", "annual_heatmap", "annual_violin", "cumulative",
             "count_dist", "seasonality", "waiting_times", "clustering", "diagnostic")


class LCSConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # ── Storm type (which cyclone class) ───────────────────────────────────────
    # "tc"  : tropical cyclones (the implemented Monte-Carlo, driven by the SCA SRR).
    # "etc" : extratropical cyclones. PLACEHOLDER (not yet implemented): the same
    #         Poisson / stratum / day-of-year machinery would run on an ETC SRR.
    storm_type: str = "tc"

    # ── SRR source (the storm_climatology_analysis output) ─────────────────────
    # input_csv : the per-CRL annual + monthly SRR table in TC/km/yr, i.e.
    #             srr_<basin>_<v>.csv (NOT the srr_<R>km variant, which is already
    #             multiplied by the diameter). Annual SRR sets the Poisson rate; the
    #             three stratum SRRs set the intensity split.
    # daily_csv : the companion continuous daily SRR table srr_daily_<basin>_<v>.csv
    #             (long form crl_id,lat,lon,doy,...). None auto-locates it next to
    #             input_csv. Used by day_method="daily" to place each TC in the year.
    input_csv: Optional[Union[str, Path]] = None
    daily_csv: Optional[Union[str, Path]] = None
    # selection_csv : the SCA per-CRL selected-TC table selection_<basin>_<v>.csv,
    # used to CALIBRATE the correlation parameters from each CRL's historical annual
    # counts. None auto-locates it next to input_csv (only read when correlation=True).
    selection_csv: Optional[Union[str, Path]] = None

    # ── Site and footprint ─────────────────────────────────────────────────────
    # crl_ids   : one CRL id, or several (a list); one synthetic catalog per CRL.
    # radius_km : radius of influence either side of the CRL. The Poisson rate is
    #             lambda = SRR(all) * (2 * radius_km): the rate density (TC/km/yr)
    #             times the 2R-km along-coast band -> expected TC/yr at the CRL.
    crl_ids: List[int] = [844]
    radius_km: float = 200.0

    # ── Simulation size ────────────────────────────────────────────────────────
    sim_years: int = 100           # length of each synthetic life cycle (years)
    n_realizations: int = 1000     # independent realizations of that life cycle

    # ── Day-of-occurrence model ────────────────────────────────────────────────
    # "daily"   : draw the day-of-year from the smooth per-stratum daily SRR curve
    #             (needs daily_csv). The physically richer option.
    # "monthly" : draw the calendar month from the per-stratum monthly SRR, then a
    #             day uniformly within that month. Uses only input_csv.
    day_method: str = "daily"

    # Random seed for reproducibility. None -> a fresh nondeterministic stream.
    # Each CRL draws from an independent sub-stream derived from (seed, crl_id).
    seed: Optional[int] = 12345

    # ── Serial correlation + clustering of annual counts (off by default) ───────
    # correlation=False keeps the independent Poisson(lambda) baseline exactly.
    # When True, the annual rate is modulated by a persistent latent climate state
    # and/or made overdispersed, so active and quiet years cluster:
    #     S_y      = ar_phi * S_{y-1} + sqrt(1-ar_phi^2) * eps_y   (AR(1), N(0,1))
    #     lambda_y = lambda * exp(ar_beta*S_y - ar_beta^2/2)       (mean-preserving)
    #     N(y)     ~ Poisson(lambda_y * G_y),  G_y ~ Gamma(mean 1, var overdispersion)
    # ar_beta drives the year-to-year memory (lag-1 autocorrelation); overdispersion
    # lifts the count variance (Fano = 1 + lambda*overdispersion). The annual mean rate
    # is preserved, so the catalog still matches the SRR.
    #
    # When correlation=True, each parameter left as None is CALIBRATED from the CRL's
    # historical annual counts (the SCA selection): overdispersion = (Fano-1)/mean and
    # ar_beta/ar_phi from the lag-1/lag-2 count autocorrelation. Set a number to
    # override that estimate. A sparse, low-rate CRL typically calibrates to ~0
    # (Poisson), which is the statistically appropriate result.
    correlation: bool = False
    ar_phi: Optional[float] = None         # AR(1) persistence [0, 1); None -> calibrate
    ar_beta: Optional[float] = None        # log-rate sensitivity (>=0); None -> calibrate
    overdispersion: Optional[float] = None  # rate-multiplier variance (>=0); None -> calibrate
    # regional_pool_km : when set, the auto-calibration pools every CRL within this
    # great-circle distance (km) of the target, so the basin/regional clustering
    # signal is estimated from many CRLs instead of one sparse record. None (default)
    # calibrates from the target CRL alone. Ignored where parameters are overridden.
    regional_pool_km: Optional[float] = None
    # regional_pool_sigma_km : optional Gaussian distance taper for the pool. None
    # (default) weights every pooled CRL uniformly (a hard cutoff at regional_pool_km);
    # a value applies w = exp(-d^2 / (2 sigma^2)) so CRLs nearer the target count more,
    # fading toward the edge (e.g. sigma = regional_pool_km / 2). Needs regional_pool_km.
    regional_pool_sigma_km: Optional[float] = None

    # ── Event sequencing ───────────────────────────────────────────────────────
    # Add the chronological event timeline to the catalog: a continuous event_time
    # (years), a per-realization chronological order (seq), and the inter-arrival
    # waiting time from the previous event (wait_yr).
    sequencing: bool = True

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: Path = Path("data/outputs")

    # ── Visualization suite (optional, off by default) ─────────────────────────
    # make_plots is the master switch; plots selects which figures to render.
    # "all" expands to every figure; otherwise list any subset of the keys in
    # plots.PLOT_KEYS (annual_fan, annual_heatmap, annual_violin, cumulative,
    # count_dist, seasonality, waiting_times, diagnostic) to toggle individually.
    make_plots: bool = False
    # validate_default so the "all" sentinel expands to PLOT_KEYS even when unset.
    plots: List[str] = Field(default=["all"], validate_default=True)
    plot_dir: Optional[Union[str, Path]] = None    # None -> output_dir / "plots"

    @field_validator("storm_type", mode="before")
    @classmethod
    def _storm_type(cls, v):
        v = str(v).strip().lower()
        if v not in ("tc", "etc"):
            raise ValueError("storm_type must be 'tc' or 'etc'")
        return v

    @field_validator("crl_ids", mode="before")
    @classmethod
    def _as_id_list(cls, v):
        if v is None:
            raise ValueError("crl_ids must name at least one CRL id.")
        if isinstance(v, (int, str)):
            v = [v]
        out = [int(x) for x in v]
        if not out:
            raise ValueError("crl_ids must name at least one CRL id.")
        return out

    @field_validator("day_method", mode="before")
    @classmethod
    def _day_method(cls, v):
        v = str(v).strip().lower()
        if v not in ("daily", "monthly"):
            raise ValueError("day_method must be 'daily' or 'monthly'")
        return v

    @field_validator("radius_km", mode="before")
    @classmethod
    def _positive_radius(cls, v):
        if float(v) <= 0.0:
            raise ValueError("radius_km must be positive.")
        return float(v)

    @field_validator("sim_years", "n_realizations", mode="before")
    @classmethod
    def _positive_count(cls, v):
        if int(v) <= 0:
            raise ValueError("sim_years and n_realizations must be positive.")
        return int(v)

    @field_validator("ar_phi", mode="before")
    @classmethod
    def _ar_phi_range(cls, v):
        if v is None:
            return None
        v = float(v)
        if not (0.0 <= v < 1.0):
            raise ValueError("ar_phi must be in [0, 1) or None (auto-calibrate).")
        return v

    @field_validator("ar_beta", "overdispersion", mode="before")
    @classmethod
    def _nonneg_or_none(cls, v):
        if v is None:
            return None
        if float(v) < 0.0:
            raise ValueError("ar_beta and overdispersion must be >= 0 or None.")
        return float(v)

    @field_validator("regional_pool_km", "regional_pool_sigma_km", mode="before")
    @classmethod
    def _pool_km_or_none(cls, v):
        if v is None:
            return None
        if float(v) <= 0.0:
            raise ValueError("regional pool distances must be > 0 or None.")
        return float(v)

    @field_validator("plots", mode="before")
    @classmethod
    def _expand_plots(cls, v):
        if isinstance(v, str):
            v = [v]
        out: List[str] = []
        for k in v:
            k = str(k).strip().lower()
            if k == "all":
                out.extend(PLOT_KEYS)
            else:
                out.append(k)
        unknown = sorted(set(out) - set(PLOT_KEYS))
        if unknown:
            raise ValueError(f"Unknown plot key(s) {unknown}; expected "
                             f"{list(PLOT_KEYS)} or 'all'.")
        # De-duplicate, preserving the canonical display order.
        return [k for k in PLOT_KEYS if k in set(out)]

    @field_validator("output_dir", mode="before")
    @classmethod
    def _as_path(cls, v):
        return Path(v)
