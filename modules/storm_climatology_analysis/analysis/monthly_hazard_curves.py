"""monthly_hazard_curves - split an annual hazard curve into 12 monthly curves by SRR.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

A hazard curve is a rate times a tail probability: the annual exceedance rate of a
response level r is AER_annual(r) = lambda_annual * P(R > r). The SCA monthly SRR
partitions the annual storm rate across the calendar (the twelve monthly rates sum
to the annual rate), so each month carries a fraction

    f_m = SRR_m / SRR_all          (dimensionless; sum_m f_m = 1)

of the annual storms. Assuming a month's storms draw from the same per-storm
response distribution as the full-year population, P(R > r) is unchanged and the
monthly hazard curve is the annual curve with its rate axis scaled,

    AER_m(r) = f_m * AER_annual(r),

i.e. on a (magnitude, AER) plot the curve shifts straight down by log10(f_m), and
the mean return interval (MRI = 1 / AER) is multiplied by 1 / f_m. The response
magnitudes are unchanged. Because sum_m f_m = 1, the twelve monthly curves sum back
to the annual curve at every magnitude (a conservative decomposition).

This tool reads an annual hazard-curve CSV (a PST output: AER plus one or more
magnitude columns such as BE, or CB10/CB90) and an SCA SRR table, then writes the
Annual + 12 monthly curves (long form) and a combined plot for one CRL.

Caveat: pure rate-scaling assumes the intensity mix is seasonally homogeneous. If a
month is relatively richer or poorer in intense storms, scale instead per intensity
stratum using the per-stratum monthly columns the SRR table also carries
(srr*_low_<Mon>, srr*_med_<Mon>, srr*_high_<Mon>) against per-stratum hazard curves.

Run
---
    python analysis/monthly_hazard_curves.py \
        --hc  <annual_hc>.csv  [--cb <annual_cb>.csv] \
        --srr <srr_or_srr200km>.csv  --crl 74  [--out <dir>] [--no-plot]
    # --hc  : annual best-estimate curve (columns: AER, BE)
    # --cb  : optional confidence band (columns: AER, CB10, CB90) to shade
    # --srr : SCA srr_<basin>_<v>.csv or srr_<R>km_<basin>_<v>.csv (the 2R cancels)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "backend" / "python", ROOT.parents[1] / "common" / "python"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from storm_climatology_analysis.gkf import MONTHS                  # noqa: E402

AER_COL = "AER"                       # frequency column in the PST hazard-curve tables
DEFAULT_OUT = ROOT / "data" / "outputs" / "monthly_hazard_curves"


# ---------------------------------------------------------------------------
# SRR monthly fractions
# ---------------------------------------------------------------------------

def detect_all_col(columns) -> str:
    """Find the annual all-bin SRR column (``srr_all`` or ``srr200km_all``).

    The annual all-bin rate is the unique column ending in ``_all``; the all-bin
    monthly columns are that name plus a ``_<Mon>`` suffix (e.g. ``srr200km_all_Oct``).
    """
    for c in columns:
        if c.endswith("_all"):
            return c
    raise ValueError("SRR file has no '<prefix>_all' column; expected an SCA SRR table.")


def monthly_fractions(srr_csv: Path, crl_id: int) -> tuple[dict, float, str]:
    """Per-month fraction f_m = SRR_m / SRR_all for one CRL, plus the annual rate.

    Returns
    -------
    fractions : dict   {month: f_m} over the twelve calendar months
    srr_all   : float  annual all-bin rate for the CRL (file units)
    all_col   : str    the detected annual all-bin column name
    """
    df = pd.read_csv(srr_csv)
    if "crl_id" not in df.columns:
        raise ValueError(f"{Path(srr_csv).name} has no 'crl_id' column.")
    df = df.set_index("crl_id")
    if crl_id not in df.index:
        raise KeyError(f"CRL {crl_id} not in {Path(srr_csv).name} "
                       f"(ids {int(df.index.min())}..{int(df.index.max())}).")
    all_col = detect_all_col(df.columns)
    row = df.loc[crl_id]
    srr_all = float(row[all_col])
    if srr_all <= 0:
        raise ValueError(f"CRL {crl_id} has SRR_all = {srr_all}; cannot form fractions.")
    fractions = {m: float(row[f"{all_col}_{m}"]) / srr_all for m in MONTHS}
    return fractions, srr_all, all_col


# ---------------------------------------------------------------------------
# Hazard-curve loading and scaling
# ---------------------------------------------------------------------------

def load_hazard_curve(hc_csv: Path, cb_csv: Path | None = None,
                      aer_col: str = AER_COL) -> tuple[pd.DataFrame, list]:
    """Load the annual hazard curve (and optional band), dropping blank-AER rows.

    Returns the merged frame (AER plus magnitude columns) and the list of magnitude
    column names. The PST tables pad the fixed AER grid with trailing all-blank rows
    beyond the valid reporting band; those are dropped.
    """
    hc = pd.read_csv(hc_csv)
    if aer_col not in hc.columns:
        raise ValueError(f"{Path(hc_csv).name} has no '{aer_col}' column.")
    if cb_csv is not None:
        cb = pd.read_csv(cb_csv)
        if aer_col not in cb.columns:
            raise ValueError(f"{Path(cb_csv).name} has no '{aer_col}' column.")
        hc = hc.merge(cb, on=aer_col, how="outer")
    hc = hc.dropna(subset=[aer_col]).sort_values(aer_col, ascending=False)
    value_cols = [c for c in hc.columns if c != aer_col]
    return hc.reset_index(drop=True), value_cols


def scale_to_monthly(hc: pd.DataFrame, fractions: dict,
                     aer_col: str = AER_COL) -> pd.DataFrame:
    """Long-form Annual + 12 monthly curves: each block is the annual curve with
    its AER scaled by the block's fraction (Annual = 1.0). Magnitudes unchanged."""
    blocks = []
    for label, f in [("Annual", 1.0)] + [(m, fractions[m]) for m in MONTHS]:
        b = hc.copy()
        b.insert(0, "month", label)
        b.insert(1, "fraction", f)
        b[aer_col] = b[aer_col] * f
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_long(long_df: pd.DataFrame, path: Path) -> Path:
    """Write the long-form Annual + monthly curves CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(path, index=False)
    return path


def plot_curves(long_df: pd.DataFrame, value_col: str, crl_id: int, *,
                aer_col: str = AER_COL, band: tuple | None = None,
                title_base: str = "", out_path: Path) -> Path | None:
    """Magnitude vs AER (log) for the Annual curve and the 12 monthly curves.

    ``band`` is an optional (lo_col, hi_col) pair shaded for the Annual curve only.
    Returns the path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # matplotlib not installed
        print(f"[sca] plot skipped (matplotlib unavailable: {exc})")
        return None
    from pystorm_common import save_figure, style_ax

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    cmap = plt.get_cmap("twilight", len(MONTHS))   # cyclic map for Jan..Dec

    ann = long_df[long_df["month"] == "Annual"].dropna(subset=[value_col])
    if band is not None:
        lo, hi = band
        a = ann.dropna(subset=[lo, hi])
        ax.fill_betweenx(a[aer_col], a[lo], a[hi], color="#A2E3EB", alpha=0.45,
                         label="Annual 10-90% band", zorder=1)
    for i, m in enumerate(MONTHS):
        sub = long_df[long_df["month"] == m].dropna(subset=[value_col])
        ax.plot(sub[value_col], sub[aer_col], color=cmap(i), linewidth=1.3,
                label=m, zorder=2)
    ax.plot(ann[value_col], ann[aer_col], color="k", linewidth=2.4,
            label="Annual", zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("Response magnitude")
    ax.set_ylabel("Annual exceedance rate (1/yr)   [MRI = 1/AER]")
    ax.set_title(f"PyStorm-SCA  Monthly hazard curves  -  CRL {crl_id}")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    style_ax(ax)
    return save_figure(fig, Path(out_path), close=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Scale an annual hazard curve into 12 monthly curves by the SCA "
                    "monthly SRR fractions for a CRL.")
    p.add_argument("--hc", required=True, type=Path,
                   help="Annual best-estimate hazard-curve CSV (columns: AER, BE).")
    p.add_argument("--cb", type=Path,
                   help="Optional confidence-band CSV (columns: AER, CB10, CB90).")
    p.add_argument("--srr", required=True, type=Path,
                   help="SCA SRR table (srr_<basin>_<v>.csv or srr_<R>km_<basin>_<v>.csv).")
    p.add_argument("--crl", required=True, type=int, help="CRL id to weight by.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default {DEFAULT_OUT}).")
    p.add_argument("--value-col", default="BE",
                   help="Magnitude column to plot as the curve (default BE).")
    p.add_argument("--no-plot", action="store_true", help="Skip the combined plot.")
    args = p.parse_args(argv)

    fractions, srr_all, all_col = monthly_fractions(args.srr, args.crl)
    hc, value_cols = load_hazard_curve(args.hc, args.cb)

    print(f"[sca] CRL {args.crl}: {all_col} = {srr_all:.6g} (from {args.srr.name})")
    print(f"[sca] monthly fractions f_m = SRR_m / SRR_all (sum = "
          f"{sum(fractions.values()):.6f}):")
    for m in MONTHS:
        f = fractions[m]
        mri_mult = (1.0 / f) if f > 0 else float("inf")
        print(f"        {m}: f={f:.6f}   AER x {f:.4g}   MRI x {mri_mult:.4g}")

    long_df = scale_to_monthly(hc, fractions)
    base = args.hc.stem
    csv_path = write_long(long_df, args.out / f"{base}_crl{args.crl}_monthly.csv")
    print(f"[sca] wrote {len(MONTHS)} monthly + Annual curves -> {csv_path}")

    if not args.no_plot:
        value_col = args.value_col if args.value_col in value_cols else value_cols[0]
        band = ("CB10", "CB90") if {"CB10", "CB90"} <= set(value_cols) else None
        png = plot_curves(long_df, value_col, args.crl, band=band,
                          out_path=args.out / f"{base}_crl{args.crl}_monthly.png")
        if png:
            print(f"[sca] wrote plot -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
