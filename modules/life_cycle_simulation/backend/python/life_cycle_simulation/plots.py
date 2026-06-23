"""plots - the per-CRL visualization suite for the life-cycle simulation.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Eight individually selectable figures, all derived from the synthetic TC catalog
(realization x year counts) one CRL at a time. Post-processing only; imports the
shared palette / styling / writer from pystorm_common.

Figures (keys in PLOT_KEYS):
  annual_fan      - TCs/year median + percentile bands vs simulation year
  annual_heatmap  - year x count, color = fraction of realizations
  annual_violin   - per-year count distribution (year-binned if long)
  cumulative      - ensemble running-total trajectories + 5-95% envelope
  count_dist      - TCs/year pmf vs Poisson, and per-realization total counts
  seasonality     - monthly + day-of-year occurrence by stratum vs driving SRR
  waiting_times   - inter-event waiting time and time-to-first-TC distributions
  diagnostic      - the three-panel quick QC (count / stratum / seasonality)

Stationarity note: every year is i.i.d. Poisson(lambda), so the per-year views
(fan / heatmap / violin) are flat across years by construction - a stationarity and
Monte-Carlo-convergence check. The cumulative view shows the year-over-year growth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from life_cycle_simulation.calendar365 import MONTHS, MONTH_START_DOY, NDOY
from life_cycle_simulation.config import PLOT_KEYS, STRATA

from pystorm_common import (
    save_figure, style_ax, WAVE_MAKER, WAVE_MAKER_CMAP, EMPH_DARK, RAMP,
)

# Stratum colors echo the SCA traffic-light scheme (High red, Med gold, Low green).
_STRATUM_COLOR = {"low": "#21C521", "med": "#E6A100", "high": "#FF2020"}
_DEEP = "#1F4E79"            # deep-ocean blue for median / reference lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mpl():
    """Import matplotlib (Agg) or raise RuntimeError so the caller can skip plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # matplotlib not installed
        raise RuntimeError(f"matplotlib unavailable ({exc})")
    return plt


def _count_matrix(catalog: pd.DataFrame, n_real: int, n_years: int) -> np.ndarray:
    """TC count per (realization, year): an [R, Y] integer matrix, zeros included."""
    counts = np.zeros((n_real, n_years), dtype=np.int64)
    if len(catalog):
        r = catalog["realization"].to_numpy() - 1
        y = catalog["year"].to_numpy() - 1
        np.add.at(counts, (r, y), 1)
    return counts


def _poisson_pmf(lam: float, k: np.ndarray) -> np.ndarray:
    """Poisson(lambda) pmf at non-negative integers k."""
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    return np.exp(-lam) * lam ** k / fact


def _nbinom_pmf(k: np.ndarray, r: float, p: float) -> np.ndarray:
    """Negative-Binomial pmf P(X=k) = C(k+r-1, k) p^r (1-p)^k, r real (via lgamma)."""
    k = np.asarray(k, float)
    log_c = np.array([math.lgamma(ki + r) - math.lgamma(r) - math.lgamma(ki + 1.0)
                      for ki in k])
    return np.exp(log_c + r * math.log(p) + k * np.log1p(-p))


def _plot_count_refs(ax, counts: np.ndarray, lam: float, k: np.ndarray) -> None:
    """Overlay BOTH count references on an annual-count histogram axis, for diagnosis.

    Always draws Poisson(lambda), the independent baseline; when the counts are
    overdispersed (Fano > ~1) also draws a Negative Binomial matched to the realized
    mean and Fano factor. Both are clearly labeled, so the simulated histogram sits on
    Poisson under the baseline and on the Negative Binomial under clustering, and the
    gap between the two curves shows the overdispersion.
    """
    ax.plot(k, _poisson_pmf(lam, k), "o--", color=EMPH_DARK, markersize=4,
            label=f"Poisson({lam:.3f})")
    flat = counts.reshape(-1).astype(float)
    mean = flat.mean() if flat.size else 0.0
    fano = float(flat.var() / mean) if mean > 0 else 1.0
    if fano > 1.01 and mean > 0:                     # overdispersed -> Negative Binomial
        r = mean / (fano - 1.0)
        p = r / (r + mean)
        ax.plot(k, _nbinom_pmf(k, r, p), "s-", color="#8159C9", markersize=4,
                label=f"NegBin(mean={mean:.3f}, Fano={fano:.2f})")


def _suptitle(fig, subtitle: str, name: str) -> None:
    """Brand the figure: PyStorm-LCS prefix on the top-level title (comment std 4.6)."""
    fig.suptitle(f"PyStorm-LCS  {subtitle}  -  {name}", fontweight="bold")


# ---------------------------------------------------------------------------
# Plot #1 family: per-year activity across the ensemble
# ---------------------------------------------------------------------------

def plot_annual_fan(counts, lam, *, subtitle, out_path) -> Path:
    """TCs/year vs simulation year: the per-year mean and 25-75% / 5-95% bands.

    The per-year *marginal* is stationary, so the mean and bands are flat across
    years whether or not the counts are serially correlated; clustering shows in the
    ACF (the clustering figure), not here. The mean (per-year average across
    realizations) is shown rather than the median, which is a degenerate 0 with
    interpolation spikes for sparse, low-rate integer counts.
    """
    plt = _mpl()
    n_years = counts.shape[1]
    years = np.arange(1, n_years + 1)
    p05, p25, p75, p95 = np.percentile(counts, [5, 25, 75, 95], axis=0)
    mean_y = counts.mean(axis=0)                    # smooth per-year mean (~ lambda)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(years, p05, p95, color=RAMP[100], label="5-95%")
    ax.fill_between(years, p25, p75, color=RAMP[200], label="25-75%")
    ax.plot(years, mean_y, color=_DEEP, linewidth=1.6, label="per-year mean")
    ax.axhline(lam, color=EMPH_DARK, linestyle="--", linewidth=1.2,
               label=f"lambda={lam:.3f}")
    ax.set_xlim(1, n_years)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("TCs per year")
    ax.set_title("Per-year count: stationary marginal across years "
                 "(serial correlation shows in the ACF, not here)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    style_ax(ax)
    _suptitle(fig, subtitle, "annual TC count (ensemble bands)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, Path(out_path), close=True)


def plot_annual_heatmap(counts, *, subtitle, out_path) -> Path:
    """Year x TC-count grid, colored by the fraction of realizations at each count."""
    plt = _mpl()
    n_real, n_years = counts.shape
    kmax = max(1, int(counts.max()))
    # Column-normalized: freq[k, y] = fraction of realizations with k TCs in year y.
    freq = np.zeros((kmax + 1, n_years))
    for k in range(kmax + 1):
        freq[k] = (counts == k).mean(axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    x_edges = np.arange(0.5, n_years + 1.5)
    y_edges = np.arange(-0.5, kmax + 1.5)
    mesh = ax.pcolormesh(x_edges, y_edges, freq, cmap=WAVE_MAKER_CMAP, shading="flat")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("fraction of realizations")
    ax.set_yticks(np.arange(0, kmax + 1))
    ax.set_xlabel("simulation year")
    ax.set_ylabel("TCs per year")
    ax.set_title("Per-year count distribution (each column sums to 1)")
    _suptitle(fig, subtitle, "annual TC count (heatmap)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, Path(out_path), close=True)


def plot_annual_violin(counts, *, subtitle, out_path) -> Path:
    """Per-year TC-count distribution as violins (years pooled into bins if long)."""
    plt = _mpl()
    n_real, n_years = counts.shape
    nbins = min(n_years, 20)
    edges = np.linspace(0, n_years, nbins + 1).astype(int)
    data, pos = [], []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        data.append(counts[:, lo:hi].ravel().astype(float))
        pos.append((lo + hi) / 2.0 + 0.5)          # bin-center year (1-based)
    width = max(1.0, 0.8 * n_years / nbins)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    parts = ax.violinplot(data, positions=pos, widths=width, showmedians=True,
                          showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(RAMP[200])
        body.set_edgecolor(_DEEP)
        body.set_alpha(0.85)
    if "cmedians" in parts:
        parts["cmedians"].set_color(_DEEP)
    ax.set_xlim(0, n_years + 1)
    ax.set_ylim(bottom=-0.3)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("TCs per year")
    binned = "" if nbins == n_years else f" (pooled into {nbins} year-bins)"
    ax.set_title(f"Per-year count distribution{binned}")
    style_ax(ax)
    _suptitle(fig, subtitle, "annual TC count (violins)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, Path(out_path), close=True)


# ---------------------------------------------------------------------------
# Ensemble accumulation and marginal distributions
# ---------------------------------------------------------------------------

def plot_cumulative(counts, lam, *, subtitle, out_path) -> Path:
    """Running-total TCs vs year: a sample of realization lines + 5-95% envelope."""
    plt = _mpl()
    n_real, n_years = counts.shape
    cum = np.cumsum(counts, axis=1)
    years = np.arange(1, n_years + 1)
    p05, p50, p95 = np.percentile(cum, [5, 50, 95], axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    nshow = min(n_real, 100)                        # thin spaghetti of realizations
    for i in range(nshow):
        ax.plot(years, cum[i], color=WAVE_MAKER, linewidth=0.5, alpha=0.15)
    ax.fill_between(years, p05, p95, color=RAMP[100], label="5-95%")
    ax.plot(years, p50, color=_DEEP, linewidth=1.8, label="median")
    ax.plot(years, lam * years, color=EMPH_DARK, linestyle="--", linewidth=1.2,
            label=f"expected lambda*yr ({lam:.3f}/yr)")
    ax.set_xlim(1, n_years)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("cumulative TCs")
    ax.set_title(f"Ensemble accumulation ({nshow} of {n_real:,} realizations shown)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax)
    _suptitle(fig, subtitle, "cumulative TC count")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, Path(out_path), close=True)


def plot_count_distributions(counts, lam, *, subtitle, out_path) -> Path:
    """Two marginals: TCs/year pmf vs the reference (Poisson or NegBin), and the
    per-realization total counts vs a Normal fit to the realized totals."""
    plt = _mpl()
    n_real, n_years = counts.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # ── Per-year count vs the count reference (Poisson, or NegBin if overdispersed) ─
    ax = axes[0]
    per_year = counts.ravel()
    kmax = max(1, int(per_year.max()))
    edges = np.arange(-0.5, kmax + 1.5)
    ax.hist(per_year, bins=edges, density=True, color=RAMP[200],
            edgecolor=_DEEP, linewidth=0.6, label="simulated")
    k = np.arange(0, kmax + 1)
    _plot_count_refs(ax, counts, lam, k)
    ax.set_xlabel("TCs per year")
    ax.set_ylabel("probability")
    ax.set_title("Annual count")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    # ── Per-realization total over the full life cycle (Normal fit to the totals) ──
    ax = axes[1]
    totals = counts.sum(axis=1)
    # Fit the Normal to the realized totals so it widens correctly when the annual
    # counts are overdispersed / serially correlated (else it matches lambda*Y).
    mu = float(totals.mean())
    sd = max(float(totals.std()), 1e-12)
    tmin, tmax = int(totals.min()), int(totals.max())
    bins = np.arange(tmin - 0.5, tmax + 1.5)
    ax.hist(totals, bins=bins, density=True, color=RAMP[200],
            edgecolor=_DEEP, linewidth=0.6, label="simulated")
    xx = np.linspace(tmin, tmax, 200)
    ax.plot(xx, np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
            color=EMPH_DARK, linewidth=1.4,
            label=f"Normal(mu={mu:.1f}, sd={sd:.1f})")
    ax.set_xlabel(f"total TCs over {n_years} yr")
    ax.set_ylabel("probability")
    ax.set_title("Per-realization total")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    _suptitle(fig, subtitle, "count distributions")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, Path(out_path), close=True)


# ---------------------------------------------------------------------------
# Seasonality and waiting times
# ---------------------------------------------------------------------------

def plot_seasonality(catalog, srr, p, *, subtitle, out_path) -> Path:
    """Monthly stacked-by-stratum occurrence + day-of-year histogram vs driving SRR."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    n = len(catalog)

    # ── Monthly occurrence, stacked by stratum, vs driving monthly SRR ──────────
    ax = axes[0]
    months = np.arange(1, 13)
    bottom = np.zeros(12)
    for s in STRATA:
        cnt = np.array([((catalog["intensity"] == s) & (catalog["month"] == m)).sum()
                        for m in months], dtype=float)
        ax.bar(months, cnt, bottom=bottom, color=_STRATUM_COLOR[s],
               edgecolor="white", linewidth=0.3, label=s.capitalize())
        bottom += cnt
    # Driving day-of-year shape = the STRATUM-WEIGHTED mix of the per-stratum daily
    # pmfs, sum_s p_s f_s(d), which is what the simulation draws (and equals the
    # overall daily SRR normalized). An equal-weight sum would not match the catalog.
    drive_doy = (np.asarray(p, float)[:, None] * srr.doy_pmf).sum(axis=0)
    if drive_doy.sum() > 0:
        drive_doy = drive_doy / drive_doy.sum()
        month_of_doy = np.repeat(months, np.diff(np.append(MONTH_START_DOY, NDOY + 1)))
        drive_month = np.array([drive_doy[month_of_doy == m].sum() for m in months])
        ax.plot(months, drive_month * n, color=_DEEP, linewidth=1.6, marker="o",
                markersize=3, label="driving SRR")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_ylabel("synthetic TCs")
    ax.set_title("Monthly occurrence by stratum")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_ax(ax)

    # ── Day-of-year occurrence vs the driving seasonal shape ───────────────────
    ax = axes[1]
    if n:
        hist, _ = np.histogram(catalog["doy"], bins=np.arange(1, NDOY + 2))
        ax.bar(np.arange(1, NDOY + 1), hist / hist.sum(), width=1.0,
               color=RAMP[100], label="simulated")
    if drive_doy.sum() > 0:
        ax.plot(np.arange(1, NDOY + 1), drive_doy, color=_DEEP, linewidth=1.4,
                label="driving SRR")
    ax.set_xticks(MONTH_START_DOY)
    ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_xlim(1, NDOY)
    ax.set_ylabel("probability")
    ax.set_title("Day-of-year occurrence")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    _suptitle(fig, subtitle, "seasonality")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, Path(out_path), close=True)


def plot_waiting_times(catalog, lam, *, n_realizations, subtitle, out_path) -> Path:
    """Inter-event waiting time and time-to-first-TC distributions vs Exponential."""
    plt = _mpl()
    # Continuous event time in years: whole years plus the fractional day-of-year.
    df = catalog[["realization", "year", "doy"]].copy()
    df["t"] = (df["year"] - 1) + (df["doy"] - 1) / float(NDOY)
    df = df.sort_values(["realization", "t"])
    t = df["t"].to_numpy()
    r = df["realization"].to_numpy()
    dt = np.diff(t)
    inter = dt[r[1:] == r[:-1]]                     # drop cross-realization gaps
    first = df.groupby("realization")["t"].min().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    rate = max(lam, 1e-12)
    tmax = float(np.percentile(inter, 99)) if inter.size else 1.0
    xx = np.linspace(0, max(tmax, 1e-3), 200)

    # ── Inter-event waiting time (years between consecutive TCs) ────────────────
    ax = axes[0]
    if inter.size:
        ax.hist(inter, bins=40, range=(0, tmax), density=True, color=RAMP[200],
                edgecolor=_DEEP, linewidth=0.5, label="simulated")
    ax.plot(xx, rate * np.exp(-rate * xx), color=EMPH_DARK, linewidth=1.4,
            label=f"Exp(mean={1.0 / rate:.2f} yr)")
    ax.set_xlabel("years between TCs")
    ax.set_ylabel("probability density")
    ax.set_title("Inter-event waiting time")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    # ── Time to the first TC of a realization (observed; quiet realizations omit) ─
    ax = axes[1]
    if first.size:
        fmax = float(np.percentile(first, 99))
        ax.hist(first, bins=40, range=(0, max(fmax, 1e-3)), density=True,
                color=RAMP[200], edgecolor=_DEEP, linewidth=0.5,
                label=f"simulated ({first.size:,} of {n_realizations:,})")
        xf = np.linspace(0, max(fmax, 1e-3), 200)
        ax.plot(xf, rate * np.exp(-rate * xf), color=EMPH_DARK, linewidth=1.4,
                label=f"Exp(mean={1.0 / rate:.2f} yr)")
    ax.set_xlabel("years to first TC")
    ax.set_ylabel("probability density")
    ax.set_title("Time to first TC")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    _suptitle(fig, subtitle, "waiting times")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, Path(out_path), close=True)


# ---------------------------------------------------------------------------
# Quick three-panel diagnostic (count / stratum / seasonality)
# ---------------------------------------------------------------------------

def plot_diagnostic(catalog, counts, srr, *, lam, p, subtitle, out_path) -> Path:
    """Three-panel QC: annual count vs its reference, the stratum split, seasonality."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # ── Panel 1: annual count vs the reference (Poisson, or NegBin if overdispersed) ─
    ax = axes[0]
    per_year = counts.ravel()
    kmax = max(1, int(per_year.max()))
    ax.hist(per_year, bins=np.arange(-0.5, kmax + 1.5), density=True, color=RAMP[200],
            edgecolor=_DEEP, linewidth=0.6, label="simulated")
    k = np.arange(0, kmax + 1)
    _plot_count_refs(ax, counts, lam, k)
    ax.set_xlabel("TCs per year")
    ax.set_ylabel("probability")
    ax.set_title("Annual count")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    # ── Panel 2: intensity split, simulated vs SRR ratios ──────────────────────
    ax = axes[1]
    x = np.arange(len(STRATA))
    sim_frac = np.array([(catalog["intensity"] == s).mean() if len(catalog) else 0.0
                         for s in STRATA])
    ax.bar(x - 0.2, p, width=0.4, color="#9FB1BA", label="SRR ratio")
    ax.bar(x + 0.2, sim_frac, width=0.4,
           color=[_STRATUM_COLOR[s] for s in STRATA], label="simulated")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STRATA])
    ax.set_ylabel("fraction of TCs")
    ax.set_title("Intensity stratum")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    # ── Panel 3: day-of-year occurrence vs driving seasonal shape ──────────────
    ax = axes[2]
    if len(catalog):
        hist, _ = np.histogram(catalog["doy"], bins=np.arange(1, NDOY + 2))
        ax.bar(np.arange(1, NDOY + 1), hist / hist.sum(), width=1.0,
               color=RAMP[100], label="simulated")
    drive = (np.asarray(p, float)[:, None] * srr.doy_pmf).sum(axis=0)   # stratum-weighted
    if drive.sum() > 0:
        ax.plot(np.arange(1, NDOY + 1), drive / drive.sum(), color=_DEEP,
                linewidth=1.4, label="driving SRR")
    ax.set_xticks(MONTH_START_DOY)
    ax.set_xticklabels(MONTHS, fontsize=8)
    ax.set_xlim(1, NDOY)
    ax.set_ylabel("probability")
    ax.set_title("Day-of-year occurrence")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    _suptitle(fig, subtitle, "diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, Path(out_path), close=True)


def plot_clustering(counts, lam, *, subtitle, out_path) -> Path:
    """Serial-correlation diagnostics: annual-count ACF and sample trajectories.

    Left: the lag-k autocorrelation of annual counts (lag-1 quantifies year-to-year
    memory), with the ~95% no-correlation band. Right: a sample of per-realization
    annual-count trajectories with the ensemble mean and lambda, and the Fano factor
    (variance/mean; > 1 indicates overdispersion). Both are flat/zero under the
    independent-Poisson baseline.
    """
    plt = _mpl()
    R, Y = counts.shape
    mean = counts.mean()
    fano = float(counts.var() / mean) if mean > 0 else 1.0
    maxlag = max(1, min(12, Y - 1))
    lags = np.arange(1, maxlag + 1)
    acf = np.array([
        (np.corrcoef(counts[:, :-L].ravel().astype(float),
                     counts[:, L:].ravel().astype(float))[0, 1]
         if counts[:, :-L].std() > 0 and counts[:, L:].std() > 0 else 0.0)
        for L in lags])
    band = 1.96 / np.sqrt(max(counts.size, 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # ── ACF of annual counts ───────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(lags, acf, color=RAMP[200], edgecolor=_DEEP, linewidth=0.5)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.axhline(band, color=EMPH_DARK, linestyle="--", linewidth=1.0,
               label="95% band (no corr)")
    ax.axhline(-band, color=EMPH_DARK, linestyle="--", linewidth=1.0)
    ax.set_xticks(lags)
    ax.set_xlabel("lag (years)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"Annual-count ACF  (lag-1 = {acf[0]:+.2f})")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    # ── Sample annual-count trajectories (clustering is visible as runs) ────────
    ax = axes[1]
    years = np.arange(1, Y + 1)
    for i in range(min(R, 15)):
        ax.plot(years, counts[i], color=WAVE_MAKER, linewidth=0.7, alpha=0.5)
    ax.plot(years, counts.mean(axis=0), color=_DEEP, linewidth=2.0, label="ensemble mean")
    ax.axhline(lam, color=EMPH_DARK, linestyle="--", linewidth=1.2,
               label=f"lambda={lam:.3f}")
    ax.set_xlim(1, Y)
    ax.set_xlabel("simulation year")
    ax.set_ylabel("TCs per year")
    ax.set_title(f"Sample trajectories  (Fano = {fano:.2f})")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)

    _suptitle(fig, subtitle, "clustering / serial correlation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, Path(out_path), close=True)


# ---------------------------------------------------------------------------
# Suite dispatcher
# ---------------------------------------------------------------------------

def render_suite(catalog: pd.DataFrame, summary: pd.DataFrame, srr, *,
                 lam: float, p, sim_years: int, n_realizations: int,
                 plots: Sequence[str], out_dir, tag: str) -> List[Path]:
    """Render the selected figures for one CRL; returns the written paths.

    ``plots`` is a subset of PLOT_KEYS. Figures that need events (seasonality,
    waiting_times) are skipped with a note when the catalog is empty. Raises
    RuntimeError (via the guarded import) only if matplotlib is missing.
    """
    _mpl()                                          # fail fast if no matplotlib
    wanted = [k for k in PLOT_KEYS if k in set(plots)]
    out_dir = Path(out_dir)
    counts = _count_matrix(catalog, n_realizations, sim_years)
    subtitle = (f"CRL {srr.crl_id}  ({srr.lat:.2f}, {srr.lon:.2f})  -  "
                f"{n_realizations:,} x {sim_years} yr")
    has_events = len(catalog) > 0

    def path(key):
        return out_dir / f"lcs_{key}_{tag}.png"

    paths: List[Path] = []
    for key in wanted:
        if key in ("seasonality", "waiting_times") and not has_events:
            print(f"[lcs]     {key}: skipped (no synthetic TCs to plot)")
            continue
        if key == "annual_fan":
            paths.append(plot_annual_fan(counts, lam, subtitle=subtitle, out_path=path(key)))
        elif key == "annual_heatmap":
            paths.append(plot_annual_heatmap(counts, subtitle=subtitle, out_path=path(key)))
        elif key == "annual_violin":
            paths.append(plot_annual_violin(counts, subtitle=subtitle, out_path=path(key)))
        elif key == "cumulative":
            paths.append(plot_cumulative(counts, lam, subtitle=subtitle, out_path=path(key)))
        elif key == "count_dist":
            paths.append(plot_count_distributions(counts, lam, subtitle=subtitle,
                                                  out_path=path(key)))
        elif key == "seasonality":
            paths.append(plot_seasonality(catalog, srr, p, subtitle=subtitle, out_path=path(key)))
        elif key == "waiting_times":
            paths.append(plot_waiting_times(catalog, lam, n_realizations=n_realizations,
                                            subtitle=subtitle, out_path=path(key)))
        elif key == "clustering":
            paths.append(plot_clustering(counts, lam, subtitle=subtitle, out_path=path(key)))
        elif key == "diagnostic":
            paths.append(plot_diagnostic(catalog, counts, srr, lam=lam, p=p,
                                         subtitle=subtitle, out_path=path(key)))
    return paths
