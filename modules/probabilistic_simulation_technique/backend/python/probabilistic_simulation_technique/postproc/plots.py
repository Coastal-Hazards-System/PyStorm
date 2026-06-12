"""plots — hazard-curve and QDO-diagnostic renderers for PST results.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Two renderers share one cohesive, publication-grade palette:

  HazardCurvePlotter      empirical WPP below/above μ (distinct colors), the GPD
                          best-estimate curve, the 10/90% confidence-limit band,
                          and the μ cross (horizontal at μ, vertical at λ_μ) on a
                          reverse-oriented log-AER x-axis. Each series toggles.
  plot_qdo_diagnostics    the QDO GPD-location selection: WMSE(μ),
                          GPD shape ξ(μ), and exceedance count, over the full
                          evaluated candidate range with the selection band
                          shaded and the chosen μ marked.

μ is the GPD location (the threshold optimized for the fit), distinct from the
POT extraction threshold u (rate λ_u = λ).
"""

from pathlib import Path
from typing  import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── Canonical Wave Maker design palette (see pystorm_common) ────────────────
from pystorm_common import (
    WAVE_MAKER, EMPIRICAL, EMPHASIS, EMPH_DARK, BAND, REF_DASH, GRID, INK, MUTED, C,
    style_ax, save_figure,
)

_C = {
    "below":   EMPIRICAL,        # empirical bulk (< μ)
    "above":   EMPHASIS,         # empirical exceedances (> μ) — coral
    "gpd":     WAVE_MAKER,       # GPD mean / WMSE — hero cyan
    "mu":      REF_DASH,         # μ cross / reference dashes
    "select":  EMPH_DARK,        # selected μ marker
    "shape":   C["sea_green"],   # GPD shape ξ
    "count":   C["amber"],       # exceedance count
    "band":    BAND,             # selection-band / CL fill (Wave Maker tint)
    "grid":    GRID,
    "spine":   INK,
    "muted":   MUTED,
}


def _on(series, name: str) -> bool:
    """True if series toggle ``name`` is enabled (default on).

    Accepts a PlotSeriesConfig-like object (attribute access), a dict, or None.
    """
    if series is None:
        return True
    if isinstance(series, dict):
        return bool(series.get(name, True))
    return bool(getattr(series, name, True))


class HazardCurvePlotter:
    """Render a PST hazard curve onto a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes; the caller owns the figure lifecycle.
    lambda_val : float
        Sample intensity λ_u (events/yr). Retained for downstream consumers
        that annotate the curve with the population rate.
    """

    def __init__(self, ax, lambda_val: float = 1.0) -> None:
        self.ax         = ax
        self.lambda_val = lambda_val

    def plot_hazard_curve(
        self,
        empirical_below: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        empirical_above: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_curve:       Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_cb:          Optional[Tuple[np.ndarray, np.ndarray]] = None,
        gpd_threshold:   Optional[float]                         = None,
        threshold_aer:   Optional[float]                         = None,
        series=None,
        ylabel:        str                                     = "Response Magnitude",
        output_path:   Optional[Path]                          = None,
    ) -> None:
        """Render the hazard curve. See module docstring for the series.

        Parameters
        ----------
        empirical_below / empirical_above : (aer, resp) or None
            Empirical WPP below / above μ.
        gpd_curve : (aer_gpd, resp_gpd) or None
            GPD best-estimate (mean) above μ.
        gpd_cb : (cb10, cb90) or None
            Confidence-limit responses; share x with ``gpd_curve``.
        gpd_threshold : float or None
            μ (horizontal line). threshold_aer : λ_μ (vertical line).
        series : PlotSeriesConfig | dict | None
            Per-series toggles; missing entries default to on.
        ylabel : str
            Y-axis label.
        output_path : pathlib.Path or None
            If given, the figure is saved at 150 DPI.
        """
        if empirical_below is not None and _on(series, "empirical_below"):
            aer_b, resp_b = empirical_below
            self.ax.scatter(aer_b, resp_b, color=_C["below"], s=14, zorder=2,
                            alpha=0.75, linewidths=0, label="Empirical < μ (WPP)")

        if empirical_above is not None and _on(series, "empirical_above"):
            aer_a, resp_a = empirical_above
            self.ax.scatter(aer_a, resp_a, color=_C["above"], s=34, zorder=5,
                            edgecolors="white", linewidths=0.5,
                            label="Empirical > μ (WPP)")

        if gpd_curve is not None and _on(series, "gpd_mean"):
            aer_gpd, resp_gpd = gpd_curve
            self.ax.plot(aer_gpd, resp_gpd, color=_C["gpd"], linewidth=2.2,
                         zorder=4, label="GPD mean")

        if gpd_cb is not None and gpd_curve is not None and _on(series, "gpd_cl"):
            cb10, cb90 = gpd_cb
            self.ax.fill_between(gpd_curve[0], cb10, cb90, color=_C["band"],
                                 linewidth=0, zorder=1,
                                 label="GPD 10–90% CLs")

        if gpd_threshold is not None and _on(series, "gpd_threshold"):
            self.ax.axhline(gpd_threshold, linestyle="--", color=_C["mu"],
                            linewidth=1.4, zorder=3,
                            label=f"GPD location μ = {gpd_threshold:.2f}")
            if threshold_aer is not None:
                # Vertical companion at μ's exceedance rate (λ_μ).
                self.ax.axvline(threshold_aer, linestyle="--", color=_C["mu"],
                                linewidth=1.4, zorder=3)

        self.ax.set_xscale("log")
        self.ax.set_xlim(1e-3, 10)
        self.ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
        self.ax.set_xlabel("Annual Exceedance Rate (AER, yr⁻¹)", fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title("PyStorm — Probabilistic Simulation Technique (PST)",
                          fontsize=13, fontweight="bold")
        style_ax(self.ax)
        self.ax.grid(True, which="both", color=_C["grid"], linewidth=0.6, alpha=0.7)
        leg = self.ax.legend(frameon=True, framealpha=0.95, edgecolor=_C["grid"],
                             fontsize=9)
        leg.get_frame().set_linewidth(0.8)
        self.ax.invert_xaxis()  # high-rate events sit on the right

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_figure(self.ax.figure, output_path)


def _shade_runs(ax, x, mask, **kw) -> None:
    """Shade contiguous True runs of ``mask`` over ``x`` with axvspan."""
    if not np.any(mask):
        return
    idx    = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for grp in np.split(idx, splits):
        ax.axvspan(float(x[grp[0]]), float(x[grp[-1]]), **kw)


def _plot_mrl_diagnostics(qdo, output_path: Path, ylabel: str = "Response") -> None:
    """Render the automated MEAN-RESIDUAL-LIFE (MRL) selection diagnostics.

    Three stacked panels over the order-statistic threshold u:
      1. mean excess e(u) with the weighted-least-squares line fit from μ* up
         (linear where a GPD holds — the basis of the method).
      2. weighted MSE of that fit vs the start threshold — μ* is its lowest
         in-band local minimum.
      3. # exceedances above u, with the selectability floor.
    """
    u    = qdo.mrl_u
    e    = qdo.mrl_excess
    wm   = qdo.mrl_wmse
    A, B = qdo.mrl_slope, qdo.mrl_intercept
    ustar = qdo.best_threshold
    floor = getattr(qdo, "min_exceedances", 0) or 0
    n     = u.size + 10                              # i_max_e = n - 10
    cnt   = (n - (np.arange(u.size) + 1)).astype(float)   # excesses above each u_i

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    fig.patch.set_facecolor("white")
    band_kw = dict(color=_C["band"], alpha=0.6, zorder=0)
    sel_kw  = dict(color=_C["select"], linestyle="--", linewidth=1.6, zorder=4)
    for ax in (a0, a1, a2):
        ax.axvspan(qdo.band_lo, qdo.band_hi, **band_kw)
        ax.axvline(ustar, **sel_kw)

    # (1) mean-excess curve + the fitted line from μ* upward.
    a0.plot(u, e, "-o", color=_C["shape"], ms=3.5, lw=1.4, label="mean excess e(u)")
    a0.axvspan(qdo.band_lo, qdo.band_hi, label="selection band", **band_kw)
    fitmask = u >= ustar
    a0.plot(u[fitmask], A * u[fitmask] + B, color=_C["gpd"], lw=1.8,
            label=f"WLS fit (A={A:+.3f} → ξ={A/(1+A):+.3f})")
    a0.plot([], [], **sel_kw, label=f"selected μ* = {ustar:.3f}")
    a0.set_ylabel(f"mean excess e(u)")
    a0.set_title("PyStorm — automated mean-residual-life (μ) selection [mrl]",
                 fontsize=12, fontweight="bold")
    style_ax(a0)
    leg = a0.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    leg.get_frame().set_linewidth(0.8)

    # (2) weighted MSE of the linear fit; μ* is its lowest in-band local minimum.
    a1.plot(u, wm, "-o", color=_C["count"], ms=3.5, lw=1.4, label="fit weighted MSE")
    a1.set_ylabel("fit weighted MSE")
    style_ax(a1)
    leg1 = a1.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    leg1.get_frame().set_linewidth(0.8)

    # (3) data richness, with the selectability floor.
    a2.plot(u, cnt, "-o", color=_C["below"], ms=3.5, lw=1.4)
    if floor and floor > 0:
        a2.axhline(floor, color=_C["muted"], linestyle=":", linewidth=1.3,
                   label=f"min exceedances = {floor}")
        a2.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    a2.set_ylabel("# exceedances > u")
    a2.set_xlabel(f"threshold u  ({ylabel})", fontsize=11)
    style_ax(a2)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_gof_diagnostics(qdo, output_path: Path, ylabel: str = "Response") -> None:
    """Render the GoF "failure-to-reject" (Choulakian-Stephens) diagnostics.

    Three stacked panels over the candidate μ:
      1. EDF statistic (A²/W²) vs its critical value; μ* is the LOWEST μ where
         the statistic falls below the critical line (GPD not rejected).
      2. GPD shape ξ(μ) — context.
      3. # exceedances(μ), with the selectability floor.
    """
    mu    = qdo.candidates
    stat  = getattr(qdo, "gof_stat", np.full_like(mu, np.nan))
    crit  = getattr(qdo, "gof_crit", np.full_like(mu, np.nan))
    stt   = getattr(qdo, "gof_statistic", "ad").upper()
    alpha = getattr(qdo, "gof_significance", 0.05)
    floor = getattr(qdo, "min_exceedances", 0) or 0
    accepted = np.isfinite(stat) & np.isfinite(crit) & (stat <= crit)

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    fig.patch.set_facecolor("white")
    band_kw = dict(color=_C["band"], alpha=0.6, zorder=0)
    sel_kw  = dict(color=_C["select"], linestyle="--", linewidth=1.6, zorder=4)
    acc_kw  = dict(color=_C["below"], alpha=0.18, zorder=0)
    for ax in (a0, a1, a2):
        ax.axvspan(qdo.band_lo, qdo.band_hi, **band_kw)
        _shade_runs(ax, mu, accepted, **acc_kw)
        ax.axvline(qdo.best_threshold, **sel_kw)

    # (1) EDF statistic vs critical value; not-rejected = stat <= crit (shaded).
    a0.plot(mu, stat, "-o", color=_C["gpd"], ms=4, lw=1.6, label=f"{stt} statistic")
    a0.plot(mu, crit, "-",  color=_C["above"], lw=1.6, label=f"critical ({stt}, α={alpha:g})")
    a0.axvspan(qdo.band_lo, qdo.band_hi, label="selection band", **band_kw)
    a0.plot([], [], **acc_kw, label="not rejected (stat ≤ crit)")
    a0.plot([], [], **sel_kw, label=f"selected μ* = {qdo.best_threshold:.3f}")
    a0.set_ylabel(f"{stt} GoF statistic")
    a0.set_title("PyStorm — GoF failure-to-reject (μ) selection [gof]",
                 fontsize=12, fontweight="bold")
    style_ax(a0)
    leg = a0.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    leg.get_frame().set_linewidth(0.8)

    # (2) GPD shape ξ(μ) — context.
    a1.plot(mu, qdo.shape, "-o", color=_C["shape"], ms=4, lw=1.6)
    a1.axhline(0.0, color=_C["muted"], linewidth=0.9)
    a1.set_ylabel("GPD shape ξ")
    style_ax(a1)

    # (3) data richness.
    a2.plot(mu, qdo.n_exceed, "-o", color=_C["count"], ms=4, lw=1.6)
    if floor and floor > 0:
        a2.axhline(floor, color=_C["muted"], linestyle=":", linewidth=1.3,
                   label=f"min exceedances = {floor}")
        a2.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    a2.set_ylabel("# exceedances > μ")
    a2.set_xlabel(f"GPD location μ candidate  ({ylabel})", fontsize=11)
    style_ax(a2)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_qdo_diagnostics(qdo, output_path: Path, ylabel: str = "Response") -> None:
    """Render the QDO GPD-location selection diagnostics to ``output_path``.

    Adapts to ``qdo.selection_method``. Four stacked panels share the candidate-μ
    x-axis over the full evaluated range:

      1. WMSE(μ) — the objective. For the "wmse" method the accept tolerance is
         drawn (it gates); for "stability" it is shown as a diagnostic only.
      2. GPD shape ξ(μ) — the selected candidate set is highlighted.
      3. robust ξ-dispersion(μ) — the stability signal (↓ = flatter).
      4. # exceedances(μ) — data richness, with the selectability floor.

    Shaded: the selection band (cyan) and the EXCLUDED region (grey) — sparse
    (< floor exceedances) always, plus ξ pinned at the lower clip for the
    "stability" method (which guards against it).

    Parameters
    ----------
    qdo : QDOResult
        Diagnostics bundle from ``select_gpd_threshold_qdo``.
    output_path : pathlib.Path
        Destination PNG (saved at 150 DPI).
    ylabel : str
        Response-magnitude label for the μ (x) axis units.
    """
    method  = getattr(qdo, "selection_method", "wmse")
    if method == "mrl":
        return _plot_mrl_diagnostics(qdo, output_path, ylabel)
    if method == "gof":
        return _plot_gof_diagnostics(qdo, output_path, ylabel)

    mu      = qdo.candidates
    is_stab = (method == "stability")
    set_lbl = "stability plateau" if is_stab else "WMSE-tolerance set"

    fig, (a0, a1, a2, a3) = plt.subplots(4, 1, figsize=(9, 11.5), sharex=True)
    fig.patch.set_facecolor("white")

    band_kw = dict(color=_C["band"], alpha=0.6, zorder=0)
    sel_kw  = dict(color=_C["select"], linestyle="--", linewidth=1.6, zorder=4)
    excl_kw = dict(color=_C["muted"], alpha=0.14, zorder=0)

    floor   = getattr(qdo, "min_exceedances", 0) or 0
    sset    = getattr(qdo, "selected_set_idx", np.empty(0, dtype=int))
    clip_lo = getattr(qdo, "shape_clip_low", -np.inf)

    # EXCLUDED candidates: too sparse always; plus lower-clip-pinned ONLY for
    # the "stability" method (the "wmse" method does not guard against it).
    excluded = qdo.n_exceed < floor
    if is_stab:
        excluded = excluded | (np.isfinite(qdo.shape) & (qdo.shape <= clip_lo + 1e-3))

    for ax in (a0, a1, a2, a3):
        ax.axvspan(qdo.band_lo, qdo.band_hi, **band_kw)
        _shade_runs(ax, mu, excluded, **excl_kw)
        ax.axvline(qdo.best_threshold, **sel_kw)

    lam_mu_arr = getattr(qdo, "lambda_mu", None)
    lam_sel    = (float(lam_mu_arr[qdo.best_idx])
                  if lam_mu_arr is not None else float("nan"))

    # (1) WMSE — the gate for "wmse" (draw accept tolerance), diagnostic for "stability".
    wmse_lbl = "WMSE (diagnostic only)" if is_stab else "WMSE"
    a0.plot(mu, qdo.wmse, "-o", color=_C["gpd"], ms=4, lw=1.6, label=wmse_lbl)
    a0.axvspan(qdo.band_lo, qdo.band_hi, label="selection band", **band_kw)
    a0.plot([], [], **excl_kw,
            label="excluded (sparse" + (" / lower-clip)" if is_stab else ")"))
    a0.plot([], [], **sel_kw,
            label=f"selected μ = {qdo.best_threshold:.2f}  (λμ = {lam_sel:.2f}/yr)")
    if not is_stab and sset.size:
        # Accept ceiling: WMSE ≤ best + tol·(upper − best), upper = Tukey-robust
        # max in-band WMSE. The selector computes it; draw the value it used.
        ceiling = float(getattr(qdo, "wmse_ceiling", np.nan))
        tol     = float(getattr(qdo, "tol", 0.05))
        if np.isfinite(ceiling):
            a0.axhline(ceiling, color=_C["above"], linestyle="--", linewidth=1.3,
                       zorder=3, label=f"accept ceiling ({tol:.0%} of floor→robust-max)")
    a0.set_ylabel("WMSE")
    a0.set_title("PyStorm — QDO GPD-location (μ) selection diagnostics  "
                 f"[{method}]", fontsize=12, fontweight="bold")
    style_ax(a0)
    leg = a0.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    leg.get_frame().set_linewidth(0.8)

    # (2) GPD shape ξ(μ) — the candidate set the selection chose among.
    a1.plot(mu, qdo.shape, "-o", color=_C["shape"], ms=4, lw=1.6, label="ξ(μ)")
    if sset.size:
        a1.plot(mu[sset], qdo.shape[sset], "o", color=_C["select"], ms=6, zorder=5,
                label=f"{set_lbl} · pick: {getattr(qdo, 'tiebreak', '')}")
    a1.axhline(0.0, color=_C["muted"], linewidth=0.9)
    a1.set_ylabel("GPD shape ξ")
    style_ax(a1)
    if sset.size:
        leg1 = a1.legend(fontsize=8, frameon=True, framealpha=0.95,
                         edgecolor=_C["grid"])
        leg1.get_frame().set_linewidth(0.8)

    # (3) Stability signal: robust ξ-dispersion (scaled MAD; ↓ = stable).
    xi_disp = np.where(np.isfinite(qdo.shape_stability),
                       qdo.shape_stability, np.nan)
    a2.plot(mu, xi_disp, "-o", color=_C["shape"], ms=4, lw=1.6,
            label="robust ξ dispersion")
    if sset.size:
        a2.plot(mu[sset], xi_disp[sset], "o", color=_C["select"], ms=6, zorder=5,
                label=set_lbl)
        if is_stab:
            d_min    = float(np.nanmin(xi_disp[sset]))
            stab_tol = float(getattr(qdo, "stab_tol", 0.0))
            a2.axhline(d_min + stab_tol, color=_C["above"], linestyle="--",
                       linewidth=1.3, zorder=3, label=f"plateau tol (+{stab_tol:g})")
    a2.set_ylabel("robust ξ dispersion (↓ = stable)")
    style_ax(a2)
    leg2 = a2.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    leg2.get_frame().set_linewidth(0.8)

    # (4) Exceedance count, with the minimum-exceedance selectability floor.
    a3.plot(mu, qdo.n_exceed, "-o", color=_C["count"], ms=4, lw=1.6)
    if floor and floor > 0:
        a3.axhline(floor, color=_C["muted"], linestyle=":", linewidth=1.3,
                   label=f"min exceedances = {floor}")
        a3.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor=_C["grid"])
    a3.set_ylabel("# exceedances > μ")
    a3.set_xlabel(f"GPD location μ candidate  ({ylabel})", fontsize=11)
    style_ax(a3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    plt.close(fig)
