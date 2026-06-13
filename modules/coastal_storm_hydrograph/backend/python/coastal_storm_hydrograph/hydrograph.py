"""hydrograph - unit / scalable storm-surge hydrographs (the CSH method core).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A scalable hydrograph reconstructs the surge time history from a fixed dimensionless
shape and a few per-storm scaling parameters. The peak-aligned, amplitude-normalized
surge ensemble still varies strongly in width, nearly independent of the peak, so
normalizing by peak alone blurs the shape. The whitepaper comparison shows that DOUBLE
NORMALIZATION (scaling time by a per-storm timescale as well as amplitude) collapses
the ensemble about ninefold and reconstructs storms best, at two physical parameters
(peak A and a timescale).

The timescale is the EQUIVALENT WIDTH

    W = (integral of a) / peak = integral of n          (units of time)

i.e. the width of the peak-height rectangle with the same area as the surge. A
companion comparison (analysis/timescale_comparison.py) shows the equivalent width
gives a marginally tighter collapse and lower reconstruction error than the full width
at half maximum (FWHM) or a second-moment width, and is always defined. It is NOT the
total wet duration; it is a characteristic width, close to the FWHM.

For one save point with ground elevation G:

  * surge above ground a(tau) = E(tau) - G (dry = 0), peak A = max(a) at the peak,
    normalized n(tau) = a/A, time tau relative to the peak.
  * equivalent width W = integral of n (h).
  * double-normalized: the canonical shape C(s) is the ensemble mean of n over the
    DIMENSIONLESS time s = tau / W; C(0) = 1.

A hydrograph for a target peak elevation P and equivalent width W is
E(tau) = G + (P - G) * C(tau / W).

ACTUAL DURATION. A physically meaningful duration is the time the water surface stays
above a threshold elevation z0 = max(ground, MHHW) + offset (offset = 0.30 m): "0.30 m
above ground" for overland points and "0.30 m above MHHW" for overwater points. It
relates to the equivalent width through the canonical level-width function Phi(f),
the width of C at height fraction f:

    actual_duration T = W * Phi(f),    f = (z0 - ground) / (P - ground),

so an observed duration converts to an equivalent width via W = T / Phi(f), given the
peak. For a threshold set as a fixed fraction of the peak, Phi(f) is constant; for a
fixed depth, f (hence the ratio) depends on the peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NormalizedStorm:
    tau: np.ndarray          # time relative to peak (h)
    n: np.ndarray            # normalized surge-above-ground (peak = 1)
    peak_elev: float         # peak water-surface elevation (m NAVD88)
    peak_surge: float        # peak surge above ground (m) = peak_elev - ground
    equiv_width: float       # W = area/peak (h); the time scale for double normalization


@dataclass
class LimbFit:
    sigma_rise: float
    p_rise: float
    sigma_fall: float
    p_fall: float
    rmse: float
    u_param: np.ndarray = field(repr=False)


@dataclass
class UnitHydrograph:
    sp_id: int
    ground_elev: float
    grid: np.ndarray          # shape grid: dimensionless s (double_norm) or tau hours (amplitude)
    u: np.ndarray             # canonical shape on `grid` (peak = 1)
    n_storms: int
    peaks: np.ndarray         # per-storm peak elevations used (m NAVD88)
    equiv_widths: np.ndarray  # per-storm equivalent widths W (h)
    window: float             # half-width of `grid` (dimensionless or hours)
    aggregate: str
    method: str               # "double_norm" or "amplitude"
    stack: np.ndarray = field(default=None, repr=False)   # (n_storms, n_grid) normalized
    fit: Optional[LimbFit] = None

    @property
    def dimensionless(self) -> bool:
        return self.method == "double_norm"

    @property
    def tau(self) -> np.ndarray:
        """Back-compatible alias for the shape grid."""
        return self.grid


def _contiguous_valid(col: np.ndarray) -> Tuple[int, int]:
    """First and last non-NaN index (inclusive); (-1, -1) if all NaN."""
    valid = np.flatnonzero(~np.isnan(col))
    if valid.size == 0:
        return -1, -1
    return int(valid[0]), int(valid[-1])


def width_at_level(grid: np.ndarray, u: np.ndarray, level: float) -> float:
    """Width of a peak-1 shape on ``grid`` at height ``level`` (0 if never reached)."""
    if level <= 0:
        return float(grid[-1] - grid[0])
    if level >= float(np.max(u)):
        return 0.0
    k = int(np.argmax(u))
    li = k
    while li > 0 and u[li] >= level:
        li -= 1
    t_left = grid[0] if u[li] >= level else float(
        np.interp(level, [u[li], u[li + 1]], [grid[li], grid[li + 1]]))
    ri = k
    while ri < len(u) - 1 and u[ri] >= level:
        ri += 1
    t_right = grid[-1] if u[ri] >= level else float(
        np.interp(level, [u[ri], u[ri - 1]], [grid[ri], grid[ri - 1]]))
    return float(t_right - t_left)


def normalize_storm(
    col: np.ndarray, ground_elev: float, *, dt_hours: float, dry_value: float,
    min_wet_samples: int,
) -> Optional[NormalizedStorm]:
    """Normalize one storm column to a peak-aligned unit shape, or None if too dry."""
    i0, i1 = _contiguous_valid(col)
    if i0 < 0:
        return None
    seg = col[i0:i1 + 1].astype(float)
    dry = seg == dry_value
    a = np.where(dry, 0.0, seg - ground_elev)
    a = np.where(np.isnan(a), 0.0, a)
    a = np.clip(a, 0.0, None)
    wet = a > 0.0
    if int(wet.sum()) < min_wet_samples:
        return None
    A = float(a.max())
    if A <= 0.0:
        return None
    k = int(np.argmax(a))
    tau = (np.arange(seg.size) - k) * dt_hours
    n = a / A
    W = float(np.sum(n) * dt_hours)                  # equivalent width (area/peak, h)
    return NormalizedStorm(tau=tau, n=n, peak_elev=A + ground_elev, peak_surge=A,
                           equiv_width=max(W, 1e-6))


def _auto_window(values: List[Tuple[np.ndarray, np.ndarray]], step: float,
                 cap: float) -> float:
    """Largest wet pre/post half-extent across (x, n) pairs, capped, step-aligned."""
    half = 0.0
    for x, n in values:
        wet = np.flatnonzero(n > 0.0)
        if wet.size == 0:
            continue
        half = max(half, float(-x[wet[0]]), float(x[wet[-1]]))
    half = min(half, cap) if half > 0 else cap
    return float(np.ceil(half / step) * step)


def build_unit_hydrograph(
    surge: np.ndarray, *, sp_id: int, ground_elev: float, dt_hours: float,
    dry_value: float, min_wet_samples: int, window_hours: Optional[float],
    max_window_hours: float, aggregate: str, method: str = "double_norm",
) -> Optional[UnitHydrograph]:
    """Build a save point's unit hydrograph (double-normalized by default).

    ``method`` is "double_norm" (canonical shape over dimensionless time s = tau/W) or
    "amplitude" (legacy: shape over physical time tau).
    """
    storms: List[NormalizedStorm] = []
    for c in range(surge.shape[1]):
        ns = normalize_storm(surge[:, c], ground_elev, dt_hours=dt_hours,
                             dry_value=dry_value, min_wet_samples=min_wet_samples)
        if ns is not None:
            storms.append(ns)
    if not storms:
        return None
    peaks = np.array([s.peak_elev for s in storms])
    widths = np.array([s.equiv_width for s in storms])

    if method == "double_norm":
        ds = 0.05
        pairs = [(s.tau / s.equiv_width, s.n) for s in storms]
        S = min(_auto_window(pairs, ds, cap=max_window_hours), 8.0)
        n_steps = int(round(S / ds))
        grid = np.arange(-n_steps, n_steps + 1) * ds
        stack = np.array([np.interp(grid, x, n, left=0.0, right=0.0) for x, n in pairs])
        window = S
    else:
        step = dt_hours
        pairs = [(s.tau, s.n) for s in storms]
        W = float(window_hours) if window_hours else _auto_window(pairs, step, max_window_hours)
        n_steps = int(round(W / step))
        grid = np.arange(-n_steps, n_steps + 1) * step
        stack = np.array([np.interp(grid, x, n, left=0.0, right=0.0) for x, n in pairs])
        window = W

    u = np.median(stack, axis=0) if aggregate == "median" else stack.mean(axis=0)
    u = np.clip(u, 0.0, 1.0)
    mid = grid.size // 2
    if u[mid] > 0:
        u = u / u[mid]
    return UnitHydrograph(sp_id=sp_id, ground_elev=ground_elev, grid=grid, u=u,
                          n_storms=len(storms), peaks=peaks, equiv_widths=widths,
                          window=window, aggregate=aggregate, method=method, stack=stack)


# ── Parametric limbs: generalized Gaussian U = exp(-0.5 (|x|/sigma)^p) ──────────

def _gen_gauss(x: np.ndarray, sigma: float, p: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * np.power(np.abs(x) / sigma, p))


def _fit_one_limb(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit one limb (x >= 0, y in [0,1], y(0)=1) -> (sigma, p)."""
    from scipy.optimize import curve_fit
    step = float(x[1] - x[0]) if x.size > 1 else 1.0
    below = np.flatnonzero(y <= 0.5)
    sigma0 = float(x[below[0]]) / 1.1774 if below.size else max(float(x.max()), 1.0)
    sigma0 = max(sigma0, abs(step), 1e-3)
    try:
        popt, _ = curve_fit(_gen_gauss, x, y, p0=[sigma0, 2.0],
                            bounds=([1e-3, 0.3], [np.inf, 12.0]), maxfev=20000)
        return float(popt[0]), float(popt[1])
    except Exception:                                         # noqa: BLE001
        return sigma0, 2.0


def fit_limbs(grid: np.ndarray, u: np.ndarray) -> LimbFit:
    """Fit separate generalized-Gaussian rising and falling limbs to the shape."""
    rise = grid <= 0
    fall = grid >= 0
    sr, pr = _fit_one_limb(-grid[rise], u[rise])
    sf, pf = _fit_one_limb(grid[fall], u[fall])
    u_param = np.where(grid <= 0, _gen_gauss(grid, sr, pr), _gen_gauss(grid, sf, pf))
    rmse = float(np.sqrt(np.mean((u - u_param) ** 2)))
    return LimbFit(sigma_rise=sr, p_rise=pr, sigma_fall=sf, p_fall=pf,
                   rmse=rmse, u_param=u_param)


def parametric_curve(grid: np.ndarray, fit: LimbFit) -> np.ndarray:
    """Evaluate the fitted piecewise parametric shape on ``grid``."""
    return np.where(grid <= 0, _gen_gauss(grid, fit.sigma_rise, fit.p_rise),
                    _gen_gauss(grid, fit.sigma_fall, fit.p_fall))


def width_stats(uh: UnitHydrograph) -> dict:
    """Equivalent-width summary (h): median and P25/P50/P75 envelope."""
    w = uh.equiv_widths
    return {"median": float(np.median(w)),
            "p25": float(np.percentile(w, 25)),
            "p50": float(np.percentile(w, 50)),
            "p75": float(np.percentile(w, 75))}


# ── Actual duration (time above a threshold) <-> equivalent width ───────────────
# The threshold elevation is z0 = max(ground, MHHW) + offset (offset 0.30 m): 0.30 m
# above ground for overland points (ground >= MHHW, or MHHW unknown) and 0.30 m above
# MHHW for overwater points (ground < MHHW). The threshold DEPTH above ground is then
# d = max(0, MHHW - ground) + offset, and the height fraction on the unit shape is
# f = d / (peak - ground). The canonical level-width Phi(f) gives actual_duration =
# W * Phi(f), so W = actual_duration / Phi(f). These are meaningful for the
# double-normalized canonical (grid in dimensionless s).

def is_overwater(ground_elev: float, mhhw: Optional[float]) -> bool:
    """A point is overwater when its ground sits below MHHW (normally wet)."""
    return mhhw is not None and ground_elev < float(mhhw)


def threshold_depth(ground_elev: float, mhhw: Optional[float], offset_m: float = 0.30) -> float:
    """Depth above ground of the threshold z0 = max(ground, MHHW) + offset."""
    extra = max(0.0, float(mhhw) - ground_elev) if mhhw is not None else 0.0
    return extra + float(offset_m)


def canonical_level_width(uh: UnitHydrograph, f: float) -> float:
    """Phi(f): width of the canonical shape at height fraction f (in grid units)."""
    return width_at_level(uh.grid, uh.u, f)


def actual_duration_from_equiv_width(uh: UnitHydrograph, equiv_width: float,
                                     peak_surge: float, *, offset_m: float = 0.30,
                                     mhhw: Optional[float] = None) -> float:
    """Actual duration (h) above the threshold, given an equivalent width and peak."""
    d = threshold_depth(uh.ground_elev, mhhw, offset_m)
    f = d / float(peak_surge) if peak_surge > 0 else 1.0
    return float(equiv_width * canonical_level_width(uh, f))


def equiv_width_from_actual_duration(uh: UnitHydrograph, actual_duration: float,
                                     peak_surge: float, *, offset_m: float = 0.30,
                                     mhhw: Optional[float] = None) -> float:
    """Equivalent width (h) implied by an observed actual duration and peak."""
    d = threshold_depth(uh.ground_elev, mhhw, offset_m)
    f = d / float(peak_surge) if peak_surge > 0 else 1.0
    phi = canonical_level_width(uh, f)
    if phi <= 0:
        raise ValueError(f"peak does not exceed the threshold (f={f:.2f}); cannot convert")
    return float(actual_duration / phi)


def actual_durations(uh: UnitHydrograph, *, offset_m: float = 0.30,
                     mhhw: Optional[float] = None) -> np.ndarray:
    """Per-storm actual duration (h) above the threshold, from the stored stack."""
    if uh.stack is None:
        raise RuntimeError("unit hydrograph has no stored storm stack")
    d = threshold_depth(uh.ground_elev, mhhw, offset_m)
    out = np.zeros(uh.n_storms)
    for i in range(uh.n_storms):
        A = uh.peaks[i] - uh.ground_elev
        f = d / A if A > 0 else 1.0
        w = width_at_level(uh.grid, uh.stack[i], f)
        out[i] = w * uh.equiv_widths[i] if uh.dimensionless else w
    return out


def scale_to_peak(uh: UnitHydrograph, peak_elev: float, *,
                  equiv_width: Optional[float] = None,
                  actual_duration: Optional[float] = None,
                  offset_m: float = 0.30, mhhw: Optional[float] = None,
                  parametric: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the unit hydrograph to a target PEAK elevation and timescale.

    The timescale may be given as an ``equiv_width`` (h) or as an ``actual_duration``
    (h, time above the 0.30 m threshold), which is converted via the canonical shape and
    the peak. If neither is given, the point's median equivalent width is used. Returns
    (tau_hours, elevation_m_NAVD88): E = ground + C(tau/W) * (peak - ground). For the
    legacy amplitude method the grid is already in hours and the timescale is ignored.
    """
    u = parametric_curve(uh.grid, uh.fit) if (parametric and uh.fit) else uh.u
    if uh.dimensionless:
        if actual_duration is not None:
            W = equiv_width_from_actual_duration(
                uh, actual_duration, float(peak_elev) - uh.ground_elev,
                offset_m=offset_m, mhhw=mhhw)
        elif equiv_width is not None:
            W = float(equiv_width)
        else:
            W = float(np.median(uh.equiv_widths))
        tau = uh.grid * W
    else:
        tau = uh.grid
    elev = uh.ground_elev + u * (float(peak_elev) - uh.ground_elev)
    return tau, elev
