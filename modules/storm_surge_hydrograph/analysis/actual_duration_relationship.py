"""Relationship between actual duration (time above threshold) and equivalent width.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

The actual duration is the time the surge stays above z0 = max(ground, MHHW) + 0.30 m
(here 0.30 m above ground; all CTXS points are overland). It relates to the equivalent
width W through the canonical level-width Phi(f), actual_duration = W * Phi(f) with
f = threshold_depth / (peak above ground). Because f depends on the peak, the
duration/W ratio is peak-dependent (not a constant), so converting an observed duration
back to W needs the peak: W = duration / Phi(f). This script reports, per overland save
point, the duration distribution, the duration/W ratio, and how accurately the canonical
conversion recovers each storm's equivalent width.

Run:  python analysis/actual_duration_relationship.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "python"))
from storm_surge_hydrograph import io, hydrograph as H   # noqa: E402

DT = 0.25
MIN_WET = 5
MIN_STORMS = 40
OFFSET = 0.30
MHHW = None             # overland (no MHHW in the no-tide CTXS run)


def main() -> None:
    raw = ROOT / "data" / "inputs" / "raw"
    staid = io.load_staid(raw / "CTXCS_staID.csv")
    pts = io.discover_save_points(raw, staid, "CTXCS_TP_SYN_Tides_0_SLC_0_surge_SP{sp}.csv")
    rows = []
    rep = None
    for p in pts:
        surge = io.load_surge_matrix(p.surge_path)
        uh = H.build_unit_hydrograph(surge, sp_id=p.sp_id, ground_elev=p.ground_elev,
                                     dt_hours=DT, dry_value=-99999.0, min_wet_samples=MIN_WET,
                                     window_hours=None, max_window_hours=72.0,
                                     aggregate="mean", method="double_norm")
        if uh is None or uh.n_storms < MIN_STORMS:
            continue
        T = H.actual_durations(uh, offset_m=OFFSET, mhhw=MHHW)
        W = uh.equiv_widths
        A = uh.peaks - uh.ground_elev
        ok = T > 0
        T, W, A = T[ok], W[ok], A[ok]
        # Canonical conversion: recover W from the observed duration and peak.
        W_pred = np.array([H.equiv_width_from_actual_duration(uh, t, a, offset_m=OFFSET, mhhw=MHHW)
                           for t, a in zip(T, A)])
        ratio = T / W
        rows.append({
            "sp_id": p.sp_id, "n_used": int(ok.sum()),
            "dur_median_h": round(float(np.median(T)), 2),
            "dur_cv": round(float(T.std() / T.mean()), 2),
            "ratio_T_over_W_median": round(float(np.median(ratio)), 3),
            "ratio_cv": round(float(ratio.std() / ratio.mean()), 3),
            "corr_T_W": round(float(np.corrcoef(T, W)[0, 1]), 3),
            "W_recover_rmse_h": round(float(np.sqrt(np.mean((W_pred - W) ** 2))), 3),
            "W_recover_rel": round(float(np.mean(np.abs(W_pred - W) / W)), 3),
        })
        if rep is None or uh.n_storms > rep[0]:
            rep = (uh.n_storms, p, uh, T, W, A, W_pred)
        print(f"SP{p.sp_id:05d}: scored ({int(ok.sum())} storms)")

    df = pd.DataFrame(rows)
    outdir = ROOT / "data" / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "actual_duration_metrics.csv", index=False)
    print("\n=== actual duration (0.30 m above ground) vs equivalent width ===")
    print(df.to_string(index=False))
    print(f"\nmean: dur/W ratio={df.ratio_T_over_W_median.mean():.2f} "
          f"(CV {df.ratio_cv.mean():.2f}), corr(T,W)={df.corr_T_W.mean():.2f}, "
          f"W recovery rel.err={df.W_recover_rel.mean()*100:.1f}%")

    # Figure: duration vs W (peak-colored) and conversion accuracy.
    if rep is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:                                      # noqa: BLE001
            return
        _, p, uh, T, W, A, W_pred = rep
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
        sc = a1.scatter(W, T, c=A, cmap="viridis", s=16)
        a1.set_xlabel("equivalent width W (h)"); a1.set_ylabel("actual duration > 0.30 m (h)")
        a1.set_title(f"SP{p.sp_id:05d}: duration vs W (color = peak)")
        a1.grid(alpha=0.3); fig.colorbar(sc, ax=a1, label="peak above ground (m)")
        lim = [0, max(W.max(), T.max()) * 1.05]
        a2.plot(lim, lim, "k--", lw=1, label="1:1")
        a2.scatter(W, W_pred, s=16, alpha=0.7)
        a2.set_xlabel("true W (h)"); a2.set_ylabel("W recovered from duration + peak (h)")
        a2.set_title(f"Canonical conversion (rel.err "
                     f"{np.mean(np.abs(W_pred-W)/W)*100:.1f}%)")
        a2.grid(alpha=0.3); a2.legend()
        fig.suptitle("CHS — SSH actual duration vs equivalent width", fontweight="bold")
        fig.tight_layout(); fig.savefig(outdir / "fig_actual_duration.png", dpi=120)
        plt.close(fig)
        print(f"wrote {outdir / 'fig_actual_duration.png'}")


if __name__ == "__main__":
    main()
