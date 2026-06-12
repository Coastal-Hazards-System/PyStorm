"""Comparison of unit-hydrograph shape models for the SSH module.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The peak-aligned, amplitude-normalized surge ensemble still varies strongly in
DURATION (FWHM coefficient of variation ~0.8, weakly correlated with peak), so a
single peak-scaled shape blurs the crest and tails. This script tests four shape
models and three duration-input strategies on the real CTXS data and writes the
metrics and figures used in the module whitepaper.

Models (each defines how to reconstruct a storm's normalized shape from a few
per-storm scalars; scored by leave-one-out (LOO) reconstruction RMSE so we measure
generalization, not fit):

  M0  amplitude-only      params: A            n_hat(tau) = T(tau)              [baseline]
  M1  double-normalized   params: A, D         n_hat(tau) = C(tau / D)          [time + amp]
  M2  duration clusters   params: A, cluster   n_hat(tau) = T_k(tau)            [K archetypes]
  M3  functional PCA      params: A, c_1..c_m  n_hat(tau) = mu + sum c_k phi_k  [m modes]

where A is peak surge above ground, D = area/peak (equivalent duration), T/C/T_k/mu
are LOO means, and phi_k are LOO principal modes. Duration strategies for M1 when D
is not known a priori: the point's median D, a regression of log D on log A, and a
P25/P50/P75 family (reported as an envelope).

Run:  python analysis/shape_model_comparison.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "python"))
from storm_surge_hydrograph import io, hydrograph as H   # noqa: E402

DT = 0.25
MIN_WET = 5
MIN_STORMS = 40                      # points with enough storms for fair PCA/clustering
PHYS = np.arange(-30.0, 30.0 + DT / 2, DT)        # physical reconstruction grid (h)
SGRID = np.arange(-15.0, 15.0 + 0.05 / 2, 0.05)   # dimensionless grid s = tau / D


@dataclass
class Storm:
    tau: np.ndarray
    n: np.ndarray
    A: float            # peak surge above ground (m)
    D: float            # equivalent duration = area / peak (h)
    n_phys: np.ndarray  # n resampled on PHYS (peak-aligned, hours)


def _load_storms(sp_id: int) -> List[Storm]:
    raw = ROOT / "data" / "inputs" / "raw"
    staid = io.load_staid(raw / "CTXCS_staID.csv")
    pts = io.discover_save_points(raw, staid, "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv",
                                  only=[sp_id])
    surge = io.load_surge_matrix(pts[0].surge_path)
    g = pts[0].ground_elev
    out: List[Storm] = []
    for c in range(surge.shape[1]):
        ns = H.normalize_storm(surge[:, c], g, dt_hours=DT, dry_value=-99999.0,
                               min_wet_samples=MIN_WET)
        if ns is None:
            continue
        D = float(np.sum(ns.n) * DT)                       # area / peak (since n=a/A)
        if D <= 0:
            continue
        n_phys = np.interp(PHYS, ns.tau, ns.n, left=0.0, right=0.0)
        out.append(Storm(tau=ns.tau, n=ns.n, A=ns.peak_surge, D=D, n_phys=n_phys))
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _canonical_dimensionless(storms: List[Storm], exclude: int) -> np.ndarray:
    acc = np.zeros_like(SGRID)
    k = 0
    for j, s in enumerate(storms):
        if j == exclude:
            continue
        acc += np.interp(SGRID, s.tau / s.D, s.n, left=0.0, right=0.0)
        k += 1
    return acc / max(k, 1)


def _loo_mean_phys(stack: np.ndarray, i: int) -> np.ndarray:
    n = stack.shape[0]
    return (stack.sum(axis=0) - stack[i]) / (n - 1)


def evaluate_point(sp_id: int) -> dict:
    storms = _load_storms(sp_id)
    n = len(storms)
    stack = np.array([s.n_phys for s in storms])           # (n, PHYS)
    A = np.array([s.A for s in storms])
    D = np.array([s.D for s in storms])

    # Collapse spread: across-storm STD on each grid.
    spread_amp = float(np.array([s.n_phys for s in storms]).std(axis=0).mean())
    cano_all = _canonical_dimensionless(storms, exclude=-1)
    dn = np.array([np.interp(SGRID, s.tau / s.D, s.n, left=0, right=0) for s in storms])
    spread_dn = float(dn.std(axis=0).mean())

    # Duration clusters (K=3) by log-D terciles.
    qedges = np.quantile(np.log(D), [1 / 3, 2 / 3])
    cluster = np.digitize(np.log(D), qedges)

    # LOO reconstruction RMSE per model.
    rm = {k: [] for k in ("M0", "M1", "M2_k3", "M3_1", "M3_2", "M3_3", "M1_medD", "M1_regD")}
    Dmed = float(np.median(D))
    # regression log D ~ log A
    b1, b0 = np.polyfit(np.log(A), np.log(D), 1)
    for i, s in enumerate(storms):
        # M0 amplitude-only
        T = _loo_mean_phys(stack, i)
        rm["M0"].append(_rmse(s.n_phys, T))
        # M1 double-norm with true D
        C = _canonical_dimensionless(storms, exclude=i)
        rec = np.interp(PHYS, SGRID * s.D, C, left=0.0, right=0.0)
        rm["M1"].append(_rmse(s.n_phys, rec))
        # M1 with median D / regression D (D not known a priori)
        rec_med = np.interp(PHYS, SGRID * Dmed, C, left=0.0, right=0.0)
        rm["M1_medD"].append(_rmse(s.n_phys, rec_med))
        Dreg = float(np.exp(b0 + b1 * np.log(s.A)))
        rec_reg = np.interp(PHYS, SGRID * Dreg, C, left=0.0, right=0.0)
        rm["M1_regD"].append(_rmse(s.n_phys, rec_reg))
        # M2 cluster template (LOO within cluster)
        same = (cluster == cluster[i])
        same[i] = False
        Tk = stack[same].mean(axis=0) if same.any() else T
        rm["M2_k3"].append(_rmse(s.n_phys, Tk))
        # M3 FPCA (LOO mean + modes; storm described by its own projection scores)
        others = np.delete(stack, i, axis=0)
        mu = others.mean(axis=0)
        U, S, Vt = np.linalg.svd(others - mu, full_matrices=False)
        for m, key in ((1, "M3_1"), (2, "M3_2"), (3, "M3_3")):
            phi = Vt[:m]
            scores = (s.n_phys - mu) @ phi.T
            rec_pca = mu + scores @ phi
            rm[key].append(_rmse(s.n_phys, rec_pca))

    # FPCA variance explained (all storms) for reporting.
    mu_all = stack.mean(axis=0)
    _, Sv, _ = np.linalg.svd(stack - mu_all, full_matrices=False)
    var = Sv ** 2 / np.sum(Sv ** 2)

    res = {"sp_id": sp_id, "n_storms": n,
           "spread_amp_only": round(spread_amp, 4), "spread_double_norm": round(spread_dn, 4),
           "D_median_h": round(Dmed, 2), "D_cv": round(float(D.std() / D.mean()), 2),
           "corr_A_D": round(float(np.corrcoef(A, D)[0, 1]), 2),
           "pca_var1": round(float(var[0]), 3), "pca_var2": round(float(var[1]), 3),
           "pca_var3": round(float(var[2]), 3)}
    for k, v in rm.items():
        res[f"rmse_{k}"] = round(float(np.mean(v)), 4)
    return res


def make_figures(df: pd.DataFrame, rep_sp: int, outdir: Path) -> None:
    """Whitepaper figures: collapse, RMSE-vs-params, duration, and FPCA modes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                          # noqa: BLE001
        print("matplotlib unavailable; skipping figures")
        return
    storms = _load_storms(rep_sp)
    A = np.array([s.A for s in storms]); D = np.array([s.D for s in storms])
    stack = np.array([s.n_phys for s in storms])

    # Fig 1: collapse (amplitude-only vs double-normalized ensemble).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in storms:
        a1.plot(s.tau, s.n, color="0.7", lw=0.5)
    a1.plot(PHYS, stack.mean(0), "b", lw=2)
    a1.set_xlim(-30, 30); a1.set_title(f"Amplitude-only (spread={df.spread_amp_only.mean():.3f})")
    a1.set_xlabel("Time relative to peak (h)"); a1.set_ylabel("normalized surge"); a1.grid(alpha=0.3)
    dn = np.array([np.interp(SGRID, s.tau / s.D, s.n, left=0, right=0) for s in storms])
    for row in dn:
        a2.plot(SGRID, row, color="0.7", lw=0.5)
    a2.plot(SGRID, dn.mean(0), "b", lw=2)
    a2.set_xlim(-6, 6); a2.set_title(f"Double-normalized (spread={df.spread_double_norm.mean():.3f})")
    a2.set_xlabel("dimensionless time s = tau / D"); a2.grid(alpha=0.3)
    fig.suptitle(f"CHS — SSH shape collapse (SP{rep_sp:05d}, n={len(storms)})", fontweight="bold")
    fig.tight_layout(); fig.savefig(outdir / "fig_collapse.png", dpi=120); plt.close(fig)

    # Fig 2: LOO reconstruction RMSE vs parameter count.
    items = [("amplitude-only", 1, df.rmse_M0.mean()), ("cluster K=3", 2, df.rmse_M2_k3.mean()),
             ("FPCA 1 mode", 2, df.rmse_M3_1.mean()), ("double-norm (A,D)", 2, df.rmse_M1.mean()),
             ("FPCA 2 modes", 3, df.rmse_M3_2.mean()), ("FPCA 3 modes", 4, df.rmse_M3_3.mean())]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p, r in items:
        ax.scatter(p, r, s=60); ax.annotate(name, (p, r), textcoords="offset points",
                                            xytext=(6, 4), fontsize=8)
    ax.set_xlabel("number of per-storm parameters"); ax.set_ylabel("LOO reconstruction RMSE")
    ax.set_title("CHS — SSH shape models: accuracy vs parameters", fontweight="bold")
    ax.grid(alpha=0.3); ax.set_xticks([1, 2, 3, 4])
    fig.tight_layout(); fig.savefig(outdir / "fig_rmse_vs_params.png", dpi=120); plt.close(fig)

    # Fig 3: duration distribution and peak-duration scatter.
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
    b1.hist(D, bins=25, color="#3a7"); b1.set_xlabel("equivalent duration D = area/peak (h)")
    b1.set_ylabel("storms"); b1.set_title(f"Duration distribution (CV={D.std()/D.mean():.2f})"); b1.grid(alpha=0.3)
    b2.scatter(A, D, s=14, alpha=0.6)
    bb1, bb0 = np.polyfit(np.log(A), np.log(D), 1)
    xs = np.linspace(A.min(), A.max(), 50); b2.plot(xs, np.exp(bb0 + bb1 * np.log(xs)), "r--")
    b2.set_xlabel("peak surge A (m)"); b2.set_ylabel("duration D (h)")
    b2.set_title(f"Peak vs duration (corr={np.corrcoef(A, D)[0,1]:.2f}: nearly independent)"); b2.grid(alpha=0.3)
    fig.suptitle(f"CHS — SSH duration (SP{rep_sp:05d})", fontweight="bold")
    fig.tight_layout(); fig.savefig(outdir / "fig_duration.png", dpi=120); plt.close(fig)

    # Fig 4: FPCA mean + first modes.
    mu = stack.mean(0); _, Sv, Vt = np.linalg.svd(stack - mu, full_matrices=False)
    var = Sv ** 2 / np.sum(Sv ** 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(PHYS, mu, "k", lw=2, label="mean")
    for m in range(2):
        ax.plot(PHYS, mu + 2 * Sv[m] / np.sqrt(len(storms)) * Vt[m],
                lw=1.2, label=f"mode {m+1} (+, {var[m]*100:.0f}% var)")
    ax.set_xlim(-30, 30); ax.set_xlabel("Time relative to peak (h)"); ax.set_ylabel("normalized surge")
    ax.set_title(f"CHS — SSH functional PCA modes (SP{rep_sp:05d})", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "fig_fpca_modes.png", dpi=120); plt.close(fig)
    print(f"wrote 4 figures -> {outdir}")


def main() -> None:
    staid = io.load_staid(ROOT / "data" / "inputs" / "raw" / "CTXCS_staID.csv")
    pts = io.discover_save_points(ROOT / "data" / "inputs" / "raw", staid,
                                  "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv")
    rows = []
    for p in pts:
        storms = _load_storms(p.sp_id)
        if len(storms) < MIN_STORMS:
            print(f"SP{p.sp_id:05d}: {len(storms)} storms (< {MIN_STORMS}); skipped for model scoring")
            continue
        rows.append(evaluate_point(p.sp_id))
        print(f"SP{p.sp_id:05d}: scored ({rows[-1]['n_storms']} storms)")
    df = pd.DataFrame(rows)
    outdir = ROOT / "data" / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "shape_model_metrics.csv", index=False)

    # Parameter counts for the reconstruction-error summary.
    params = {"rmse_M0": 1, "rmse_M1": 2, "rmse_M2_k3": 2, "rmse_M3_1": 2,
              "rmse_M3_2": 3, "rmse_M3_3": 4, "rmse_M1_medD": 1, "rmse_M1_regD": 1}
    print("\n=== mean over scored save points ===")
    print(f"collapse spread: amplitude-only={df.spread_amp_only.mean():.3f} "
          f"double-norm={df.spread_double_norm.mean():.3f}")
    print(f"duration CV={df.D_cv.mean():.2f}  corr(A,D)={df.corr_A_D.mean():.2f}  "
          f"PCA var [m1,m2,m3]={df.pca_var1.mean():.2f},{df.pca_var2.mean():.2f},{df.pca_var3.mean():.2f}")
    print("LOO reconstruction RMSE (normalized shape):")
    for k in ("M0", "M2_k3", "M3_1", "M1", "M3_2", "M3_3"):
        print(f"  {k:8s} (params={params['rmse_'+k]}): {df['rmse_'+k].mean():.4f}")
    print("Duration strategy for M1 (params=1, D not from the storm):")
    for k in ("M1_medD", "M1_regD"):
        print(f"  {k:8s}: {df['rmse_'+k].mean():.4f}   (vs true-D M1={df.rmse_M1.mean():.4f})")
    print(f"\nwrote {outdir / 'shape_model_metrics.csv'}")
    if not df.empty:
        rep = int(df.sort_values("n_storms").iloc[-1].sp_id)   # most-sampled point
        make_figures(df, rep, outdir)


if __name__ == "__main__":
    main()
