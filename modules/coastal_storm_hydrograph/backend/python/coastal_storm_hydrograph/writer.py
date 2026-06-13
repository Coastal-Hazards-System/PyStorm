"""writer - output writers for the coastal_storm_hydrograph (CSH) module.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from coastal_storm_hydrograph.hydrograph import (
    UnitHydrograph, scale_to_peak, width_stats, actual_durations,
    threshold_depth, is_overwater,
)


def write_unit_hydrograph(uh: UnitHydrograph, path) -> Path:
    """Per-save-point canonical shape CSV.

    Double-normalized: columns ``s_dimensionless`` (= tau/D), ``u_empirical``,
    ``u_parametric``. Amplitude (legacy): ``tau_hours`` instead of ``s_dimensionless``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xcol = "s_dimensionless" if uh.dimensionless else "tau_hours"
    out = pd.DataFrame({xcol: uh.grid, "u_empirical": uh.u})
    if uh.fit is not None:
        out["u_parametric"] = uh.fit.u_param
    out.to_csv(path, index=False)
    return path


def write_scaled_hydrograph(uh: UnitHydrograph, peak_elev: float, path, *,
                            equiv_width: Optional[float] = None,
                            actual_duration: Optional[float] = None,
                            offset_m: float = 0.30, mhhw: Optional[float] = None,
                            parametric: bool = False) -> Path:
    """Scaled hydrograph CSV (tau h, elevation m NAVD88, surge above ground) for a
    target peak elevation and timescale (equivalent width, or an actual duration to
    convert via the canonical level-width)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tau, elev = scale_to_peak(uh, peak_elev, equiv_width=equiv_width,
                              actual_duration=actual_duration, offset_m=offset_m,
                              mhhw=mhhw, parametric=parametric)
    pd.DataFrame({"tau_hours": tau, "elevation_m_navd88": elev,
                  "surge_above_ground_m": elev - uh.ground_elev}).to_csv(path, index=False)
    return path


def write_parameters(uhs: Sequence[UnitHydrograph], save_points, path, *,
                     offset_m: float = 0.30, mhhw: float = None) -> Path:
    """Summary CSV: one row per save point (geometry, storms, peaks, widths, duration, fit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {p.sp_id: p for p in save_points}
    rows = []
    for uh in uhs:
        sp = by_id.get(uh.sp_id)
        pk = uh.peaks
        ws = width_stats(uh)
        corr = (float(np.corrcoef(uh.peaks, uh.equiv_widths)[0, 1])
                if uh.n_storms > 2 else np.nan)
        ad = actual_durations(uh, offset_m=offset_m, mhhw=mhhw) if uh.dimensionless else None
        ad_med = float(np.median(ad[ad > 0])) if ad is not None and np.any(ad > 0) else np.nan
        row = {
            "sp_id": uh.sp_id,
            "lat": getattr(sp, "lat", np.nan),
            "lon": getattr(sp, "lon", np.nan),
            "ground_elev_m_navd88": round(uh.ground_elev, 4),
            "overwater": is_overwater(uh.ground_elev, mhhw),
            "method": uh.method,
            "n_storms": uh.n_storms,
            "peak_min_m": round(float(np.min(pk)), 4),
            "peak_median_m": round(float(np.median(pk)), 4),
            "peak_max_m": round(float(np.max(pk)), 4),
            "equiv_width_p25_h": round(ws["p25"], 3),
            "equiv_width_median_h": round(ws["p50"], 3),
            "equiv_width_p75_h": round(ws["p75"], 3),
            "actual_dur_offset_m": offset_m,
            "actual_dur_threshold_depth_m": round(threshold_depth(uh.ground_elev, mhhw, offset_m), 3),
            "actual_dur_median_h": round(ad_med, 3) if np.isfinite(ad_med) else np.nan,
            "corr_peak_width": round(corr, 3) if np.isfinite(corr) else np.nan,
            "aggregate": uh.aggregate,
        }
        if uh.fit is not None:
            row.update({
                "sigma_rise": round(uh.fit.sigma_rise, 4),
                "p_rise": round(uh.fit.p_rise, 4),
                "sigma_fall": round(uh.fit.sigma_fall, 4),
                "p_fall": round(uh.fit.p_fall, 4),
                "fit_rmse": round(uh.fit.rmse, 5),
            })
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("sp_id").reset_index(drop=True)
    out.to_csv(path, index=False)
    return path
