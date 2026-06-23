"""plots - per-CRL marginal-distribution diagnostic figures (WPP vs fit).

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

One figure per CRL: a 3x3 grid (intensity HI/MI/LI x parameter Dp/Rmax/Vt) of the
fitted marginal CDF against the empirical Weibull plotting positions. Post-processing
only; imports the shared palette / styling / writer from pystorm_common.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from joint_distribution_model.config import (
    INTENSITY_BINS, INTENSITY_LABEL, PARAM_NAMES,
)
from joint_distribution_model.marginals import weibull_cdf

_DEEP = "#1F4E79"        # deep-ocean blue for fitted curves / scatter

# Column index of each parameter in the adjusted [Hd, Dp, Rmax, Vt] bin arrays.
_COL = {"Dp": 1, "Rmax": 2, "Vt": 3}
_INTENSITIES = ("high", "med", "low")
_PARAMS = ("Dp", "Rmax", "Vt")
_STRATUM_COLOR = {"high": "#FF2020", "med": "#E6A100", "low": "#21C521"}
# Fixed, full-range x-axes per parameter so each bin's (truncated) marginal is shown
# in context across the whole parameter range, not zoomed into its narrow band.
_XLIM = {"Dp": (0.0, 100.0), "Rmax": (0.0, 250.0), "Vt": (0.0, 160.0)}
_XLABEL = {"Dp": "Dp (hPa)", "Rmax": "Rmax (km)", "Vt": "Vt (km/h)"}


def _fitted_cdf(rec, x):
    """Fitted non-exceedance CDF of a marginal record over grid ``x``."""
    dist = rec["dist"]
    p1, p2 = rec["p1"], rec["p2"]
    if not (np.isfinite(p1) and np.isfinite(p2)):
        return np.full_like(x, np.nan)
    if dist == "weibull_trunc":
        lo, hi = rec["trunc_lo"], rec["trunc_hi"]
        flo, fhi = weibull_cdf(lo, p1, p2), weibull_cdf(hi, p1, p2)
        return np.clip((weibull_cdf(x, p1, p2) - flo) / (fhi - flo), 0.0, 1.0)
    if dist == "lognorm":                              # p1=mu, p2=sigma (log space)
        from scipy.stats import norm
        with np.errstate(divide="ignore"):
            return norm.cdf((np.log(np.where(x > 0, x, np.nan)) - p1) / p2)
    if dist == "norm":                                 # p1=mu, p2=sigma
        from scipy.stats import norm
        return norm.cdf((x - p1) / p2)
    return np.full_like(x, np.nan)


def plot_crl_marginals(crl_id, bins, records, boot_extra, *, basin, out_dir) -> int:
    """Write the 3x3 marginal-diagnostic figure for one CRL. Returns 1, or 0 if empty.

    Raises RuntimeError if matplotlib is unavailable so the orchestrator can skip.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                            # matplotlib not installed
        raise RuntimeError(f"matplotlib unavailable ({exc})")
    from pystorm_common import save_figure, style_ax

    rec_at = {(r["intensity"], r["param"]): r for r in records}
    fig, axes = plt.subplots(4, 3, figsize=(13, 14))

    # ── Row 0: the shared body Dp fit (the actual Weibull fit, on all Dp >= dp_low),
    #          the union of HI+MI shown in context. Rmax/Vt have no All-bin fit.
    ax = axes[0, 0]
    rb = rec_at.get(("all", "Dp_body"))
    lo = rb["trunc_lo"] if rb else 28.0
    x1 = _XLIM["Dp"][1]
    if rb is not None and np.isfinite(rb["p1"]) and np.isfinite(rb["p2"]):
        A, k, hi = rb["p1"], rb["p2"], rb["trunc_hi"]
        xg = np.linspace(1e-6, x1, 400)
        flo, fhi = weibull_cdf(lo, A, k), weibull_cdf(hi, A, k)
        ax.plot(xg, np.clip((weibull_cdf(xg, A, k) - flo) / (fhi - flo), 0, 1),
                color=_DEEP, lw=1.6, label="weibull")
    dp_all = bins["all"][:, 1] if bins["all"].size else np.empty(0)
    body = np.sort(dp_all[dp_all >= lo])
    if body.size:
        emp = (np.arange(1, body.size + 1)) / (body.size + 1)
        ax.plot(body, emp, "o", ms=3, mfc="#9FB1BA", mec="k", mew=0.3,
                ls="none", label="WPP")
    ax.set_xlim(0, x1); ax.set_ylim(0, 1)
    ax.set_xlabel(_XLABEL["Dp"]); ax.set_ylabel("All  Dp>=28\nnon-exceedance")
    ax.set_title("Dp  (body fit)")
    ax.legend(frameon=False, fontsize=7, loc="lower right"); style_ax(ax)
    axes[0, 1].axis("off"); axes[0, 2].axis("off")

    # ── Rows 1-3: HI / MI / LI marginals (truncated/parametric fits vs the bin WPP).
    for ri, inten in enumerate(_INTENSITIES, start=1):
        d = bins[inten]
        for ci, param in enumerate(_PARAMS):
            ax = axes[ri, ci]
            x0, x1 = _XLIM[param]
            # Fitted marginal across the FULL fixed range (a truncated Weibull reads
            # as 0 below its band, rising within it, and 1 above).
            rec = rec_at.get((inten, param))
            if rec is not None:
                xg = np.linspace(max(x0, 1e-6), x1, 400)
                ax.plot(xg, _fitted_cdf(rec, xg), color=_DEEP,
                        lw=1.6, label=rec["dist"])
            # Empirical Weibull plotting positions (non-exceedance).
            vals = d[:, _COL[param]] if d.size else np.empty(0)
            vals = np.sort(vals[np.isfinite(vals)])
            n = vals.size
            if n:
                emp = (np.arange(1, n + 1)) / (n + 1)
                ax.plot(vals, emp, "o", ms=3, mfc=_STRATUM_COLOR[inten],
                        mec="k", mew=0.3, ls="none", label="WPP")
            ax.set_xlim(x0, x1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(_XLABEL[param])
            if ci == 0:
                ax.set_ylabel(f"{INTENSITY_LABEL[inten]}\nnon-exceedance")
            if ri == 1:
                ax.set_title(param)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            style_ax(ax)

    fig.suptitle(f"PyStorm-JDM  Marginal distributions  -  {basin.capitalize()} "
                 f"CRL {int(crl_id):04d}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, Path(out_dir) / f"CHS_{basin.capitalize()}_CRL_{int(crl_id):04d}.png",
                close=True)
    return 1


def plot_crl_copula(crl_id, bins, copula, *, basin, out_dir) -> int:
    """Write the per-CRL meta-Gaussian copula figure. Returns 1, or 0 if matplotlib
    is unavailable -> RuntimeError so the orchestrator can skip.

    Top: a Gaussian-copula rho heatmap per intensity (All/HI/MI/LI) over
    [Hd, Dp, Rmax, Vt]. Bottom: an All-bin pairs scatter-matrix (lower triangle
    bivariate scatter, diagonal histograms, upper triangle the Kendall tau / rho).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                            # matplotlib not installed
        raise RuntimeError(f"matplotlib unavailable ({exc})")
    from pystorm_common import save_figure, RAMP, WAVE_MAKER

    P = PARAM_NAMES
    fig = plt.figure(figsize=(12, 14))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.2], hspace=0.30)

    # ── Top: Gaussian-copula rho heatmaps per intensity ────────────────────────
    top = outer[0].subgridspec(1, len(INTENSITY_BINS), wspace=0.40)
    im = None
    for j, b in enumerate(INTENSITY_BINS):
        ax = fig.add_subplot(top[0, j])
        rho = np.asarray(copula[b]["rho"], float)
        im = ax.imshow(rho, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(4)); ax.set_xticklabels(P, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(P, fontsize=7)
        ax.set_title(f"{INTENSITY_LABEL[b]} rho", fontsize=10)
        for r in range(4):
            for c in range(4):
                v = rho[r, c]
                ax.text(c, r, f"{v:.2f}" if np.isfinite(v) else "-",
                        ha="center", va="center", fontsize=6,
                        color="white" if (np.isfinite(v) and abs(v) > 0.55) else "black")
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            "Gaussian copula rho", fontsize=8)

    # ── Bottom: All-bin pairs scatter-matrix ───────────────────────────────────
    alld = np.asarray(bins["all"], float)
    tau = np.asarray(copula["all"]["tau"], float)
    rho = np.asarray(copula["all"]["rho"], float)
    bot = outer[1].subgridspec(4, 4, wspace=0.08, hspace=0.08)
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(bot[i, j])
            if i == j:                                  # marginal histogram
                col = alld[:, i][np.isfinite(alld[:, i])] if alld.size else np.empty(0)
                if col.size:
                    ax.hist(col, bins=20, color=RAMP[200], edgecolor=_DEEP, linewidth=0.4)
                ax.set_yticks([])
            elif i > j:                                 # bivariate scatter
                if alld.size:
                    ax.scatter(alld[:, j], alld[:, i], s=4, alpha=0.35,
                               color=WAVE_MAKER, edgecolors="none")
            else:                                       # upper triangle: tau / rho
                t, r = tau[i, j], rho[i, j]
                ax.text(0.5, 0.5,
                        f"tau={t:.2f}\nrho={r:.2f}" if np.isfinite(t) else "-",
                        ha="center", va="center", fontsize=9, transform=ax.transAxes)
                ax.axis("off")
            if i == 3 and i >= j:
                ax.set_xlabel(P[j], fontsize=8)
            if j == 0 and i >= j:
                ax.set_ylabel(P[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"PyStorm-JDM  Copula [Hd, Dp, Rmax, Vt]  -  {basin.capitalize()} "
        f"CRL {int(crl_id):04d}\n(top: Gaussian rho per intensity;  "
        f"bottom: All-bin pairs, tau/rho in the upper triangle)",
        fontsize=11, fontweight="bold")
    save_figure(fig, Path(out_dir) / f"CHS_{basin.capitalize()}_CRL_{int(crl_id):04d}.png",
                close=True)
    return 1
