"""config - configuration model for the joint_distribution_model module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A single validated ``JDMConfig`` carries the operator options from the launcher to
the orchestrator. JDM consumes the storm_climatology_analysis (SCA) outputs (the
per-CRL TC selection table and the DSRR arrays) and characterizes, per CRL and per
intensity bin, the joint distribution of TC parameters [Heading, Dp, Rmax, Vt].
Only tropical cyclones are implemented; ``etc`` is a scaffolded placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator

# Canonical basins. Pacific is scaffolded but off by default (no Pacific SCA
# selection/DSRR outputs exist yet).
BASINS = ("atlantic", "pacific")

# Intensity bins, by SCA bin name, mapped to the CHS intensity labels. The three
# non-"all" bins partition the All bin by central-pressure deficit Dp = 1013 - Cp.
INTENSITY_BINS = ("all", "high", "med", "low")
INTENSITY_LABEL = {"all": "All", "high": "HI", "med": "MI", "low": "LI"}

# JPM joint-distribution parameters, in copula/output column order.
PARAM_NAMES = ("Hd", "Dp", "Rmax", "Vt")


class JDMConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # ── Storm type ─────────────────────────────────────────────────────────────
    # "tc" : tropical cyclones (implemented). "etc" : placeholder (not implemented).
    storm_type: str = "tc"

    # Which basins to characterize. Atlantic only for now (Pacific SCA outputs do
    # not exist yet); the dispatch is basin-aware for when they do.
    basins: List[str] = ["atlantic"]

    # ── I/O layout ─────────────────────────────────────────────────────────────
    input_dir: Path = Path("data/inputs")
    output_dir: Path = Path("data/outputs")

    # ── SCA source (the storm_climatology_analysis outputs) ────────────────────
    # JDM reads two SCA products per basin: the per-CRL selected-TC table
    # (selection_<basin>_<v>.csv) and the DSRR arrays (dsrr_<basin>_<v>.npz). None
    # auto-locates the newest matching files under ``sca_outputs_dir``; set an
    # explicit path (absolute, or relative to input_dir) to pin one.
    sca_outputs_dir: Optional[Union[str, Path]] = None
    atlantic_selection_file: Optional[Union[str, Path]] = None
    atlantic_dsrr_file: Optional[Union[str, Path]] = None
    pacific_selection_file: Optional[Union[str, Path]] = None
    pacific_dsrr_file: Optional[Union[str, Path]] = None

    # ── Adjustment + intensity binning ─────────────────────────────────────────
    ref_pressure: float = 1013.0   # deficit reference: Dp = ref_pressure - Cp (hPa)
    # Which selection Cp column drives Dp. "cp_gauss" (the default) is SCA's
    # representative-point Cp (the Gaussian distance-weight x deficit fix), which
    # aligns JDM's intensity bin with SCA's SRR stratum for each storm; "cp_mindist"
    # is the Cp at the closest-approach fix.
    cp_source: str = "cp_gauss"
    start_year: int = 1938         # drop selected TCs before this season (pre-1938)
    # Intensity-bin deficit thresholds (hPa): All [min_dp, inf), HI [dp_med, inf),
    # MI [dp_low, dp_med), LI [min_dp, dp_low). Match SCA strata.
    min_dp: float = 8.0
    dp_low: float = 28.0
    dp_med: float = 48.0
    # Physical clips applied after the distance-weighted adjustment.
    vt_clip: tuple = (1.0, 152.0)      # forward translation speed (km/h)
    rmax_clip: tuple = (8.0, 200.0)    # radius of maximum winds (km)

    # ── Dp marginal bootstrap ──────────────────────────────────────────────────
    # Parametric jitter bootstrap for the Dp Weibull confidence limits. n_boot is
    # the per-CRL resample count (default 10000); the heaviest step, so it is
    # tunable and parallelized. None/0 jobs = auto (cores-1), 1 = serial.
    n_boot: int = 10000
    seed: Optional[int] = 12345
    n_jobs: Optional[int] = None

    # ── Per-CRL diagnostic plots (optional, off by default) ────────────────────
    make_plots: bool = False
    plot_dir: Optional[Union[str, Path]] = None    # None -> output_dir / "plots"

    @field_validator("storm_type", mode="before")
    @classmethod
    def _storm_type(cls, v):
        v = str(v).strip().lower()
        if v not in ("tc", "etc"):
            raise ValueError("storm_type must be 'tc' or 'etc'")
        return v

    @field_validator("cp_source", mode="before")
    @classmethod
    def _cp_source(cls, v):
        v = str(v).strip().lower()
        if v not in ("cp_mindist", "cp_gauss"):
            raise ValueError("cp_source must be 'cp_mindist' or 'cp_gauss'")
        return v

    @field_validator("basins", mode="before")
    @classmethod
    def _expand_basins(cls, v):
        if isinstance(v, str):
            v = [v]
        out: List[str] = []
        for b in v:
            b = str(b).strip().lower()
            out.extend(BASINS if b == "both" else [b])
        ordered = [b for b in BASINS if b in out]
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
        """Raw source inputs."""
        return Path(self.input_dir) / "raw"

    def selection_file_for(self, basin: str) -> Optional[Union[str, Path]]:
        """Operator-pinned selection CSV for a basin (None -> auto-locate)."""
        return {"atlantic": self.atlantic_selection_file,
                "pacific": self.pacific_selection_file}.get(basin)

    def dsrr_file_for(self, basin: str) -> Optional[Union[str, Path]]:
        """Operator-pinned DSRR npz for a basin (None -> auto-locate)."""
        return {"atlantic": self.atlantic_dsrr_file,
                "pacific": self.pacific_dsrr_file}.get(basin)
