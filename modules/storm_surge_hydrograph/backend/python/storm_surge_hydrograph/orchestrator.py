"""Orchestrator for the storm_surge_hydrograph (SSH) module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Per save point: load the synthetic-TC surge matrix, build the peak-aligned unit
hydrograph (optionally with a rising/falling parametric fit), write the normalized
table and example scaled hydrographs, and plot a diagnostic figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from storm_surge_hydrograph.config import SSHConfig
from storm_surge_hydrograph import io, writer
from storm_surge_hydrograph.hydrograph import build_unit_hydrograph, fit_limbs


@dataclass
class SavePointResult:
    sp_id: int
    n_storms: int
    ground_elev: float
    median_equiv_width: float
    unit_path: Path
    scaled_paths: List[Path] = field(default_factory=list)
    plot_path: Optional[Path] = None
    ensemble_plot_path: Optional[Path] = None


@dataclass
class SSHResult:
    params_path: Optional[Path] = None
    results: Dict[int, SavePointResult] = field(default_factory=dict)


class SSHOrchestrator:
    def __init__(self, config: SSHConfig) -> None:
        self.cfg = config

    def _resolve_in(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.cfg.raw_dir / p

    def _scale_peaks_for(self, uh) -> List[float]:
        sp = self.cfg.scale_peaks
        if sp is None:
            return []
        if isinstance(sp, str) and sp.lower() == "auto":
            return [float(np.median(uh.peaks)), float(np.max(uh.peaks))]
        return [float(x) for x in sp]

    def run(self) -> SSHResult:
        cfg = self.cfg
        out = cfg.output_dir
        staid = io.load_staid(self._resolve_in(cfg.staid_file),
                              depth_is_positive_down=cfg.depth_is_positive_down)
        pts = io.discover_save_points(cfg.raw_dir, staid, cfg.surge_file_glob,
                                      only=cfg.save_points)
        if not pts:
            print("[ssh] no save points found; check input paths.")
            return SSHResult()
        dt = io.confirm_time_step(self._resolve_in(cfg.time_file), cfg.dt_hours)
        if abs(dt - cfg.dt_hours) > 1e-6:
            print(f"[ssh] note: timestamps imply dt={dt:.3f} h (config {cfg.dt_hours}); "
                  f"using {cfg.dt_hours} h.")
        print(f"[ssh] {len(pts)} save points; dt={cfg.dt_hours*60:.0f} min")

        result = SSHResult()
        uhs = []
        for p in pts:
            surge = io.load_surge_matrix(p.surge_path)
            uh = build_unit_hydrograph(
                surge, sp_id=p.sp_id, ground_elev=p.ground_elev, dt_hours=cfg.dt_hours,
                dry_value=cfg.dry_value, min_wet_samples=cfg.min_wet_samples,
                window_hours=cfg.window_hours, max_window_hours=cfg.max_window_hours,
                aggregate=cfg.aggregate, method=cfg.method)
            if uh is None:
                print(f"[ssh] SP{p.sp_id:05d}: no wet storms; skipped.")
                continue
            if cfg.parametric:
                uh.fit = fit_limbs(uh.grid, uh.u)
            uhs.append(uh)

            from storm_surge_hydrograph.hydrograph import width_stats, actual_durations
            wstat = width_stats(uh)
            ad = actual_durations(uh, offset_m=cfg.actual_duration_offset_m,
                                  mhhw=cfg.mhhw_navd88) if uh.dimensionless else None
            ad_med = float(np.median(ad[ad > 0])) if ad is not None and np.any(ad > 0) else float("nan")
            unit_path = writer.write_unit_hydrograph(
                uh, out / f"unit_hydrograph_SP{p.sp_id:05d}.csv")
            # Peak-scaling examples at the median equivalent width; plus a width envelope
            # (P25/P50/P75) at the median peak, since the timescale is an independent input.
            scaled_paths = []
            for P in self._scale_peaks_for(uh):
                sp_path = out / "scaled" / f"hydrograph_SP{p.sp_id:05d}_peak{P:.2f}m.csv"
                scaled_paths.append(writer.write_scaled_hydrograph(
                    uh, P, sp_path, equiv_width=wstat["p50"]))
            if uh.dimensionless:
                Pmed = float(np.median(uh.peaks))
                for tag, W in (("p25", wstat["p25"]), ("p50", wstat["p50"]), ("p75", wstat["p75"])):
                    wp = out / "scaled" / f"hydrograph_SP{p.sp_id:05d}_widthenv_{tag}.csv"
                    scaled_paths.append(writer.write_scaled_hydrograph(
                        uh, Pmed, wp, equiv_width=W))

            plot_path = ensemble_plot_path = None
            if cfg.plots:
                try:
                    from storm_surge_hydrograph import plots
                    plot_path = plots.plot_save_point(
                        uh, out / "plots" / f"SSH_SP{p.sp_id:05d}.png",
                        lat=p.lat, lon=p.lon, scale_peaks=self._scale_peaks_for(uh) or None)
                    ensemble_plot_path = plots.plot_aligned_ensemble(
                        uh, out / "plots" / f"SSH_ensemble_SP{p.sp_id:05d}.png",
                        lat=p.lat, lon=p.lon)
                except RuntimeError as exc:
                    print(f"[ssh] SP{p.sp_id:05d}: plot skipped ({exc})")

            fittxt = (f" fit rmse={uh.fit.rmse:.3f}" if uh.fit else "")
            adtxt = f", actual_dur={ad_med:.1f} h" if np.isfinite(ad_med) else ""
            print(f"[ssh] SP{p.sp_id:05d}: {uh.n_storms:3d} storms, ground={uh.ground_elev:+.2f} m, "
                  f"W_eq={wstat['p50']:.1f} h [{wstat['p25']:.1f}, {wstat['p75']:.1f}]{adtxt}{fittxt}")
            result.results[p.sp_id] = SavePointResult(
                sp_id=p.sp_id, n_storms=uh.n_storms, ground_elev=uh.ground_elev,
                median_equiv_width=wstat["p50"], unit_path=unit_path,
                scaled_paths=scaled_paths, plot_path=plot_path,
                ensemble_plot_path=ensemble_plot_path)

        if uhs:
            result.params_path = writer.write_parameters(
                uhs, pts, out / "ssh_parameters.csv",
                offset_m=cfg.actual_duration_offset_m, mhhw=cfg.mhhw_navd88)
            print(f"[ssh] wrote unit hydrographs + parameters -> {out}")
        return result
