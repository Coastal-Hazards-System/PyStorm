"""One-off benchmark - times the α/β sweep on real data with all

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

optimisations on, comparing sequential vs parallel.

Honors SCOPE in the launcher (local applies the bbox filter; regional uses
all nodes and storms).

Run from venv:
  python scripts/_bench_ab_sweep.py
"""

import sys
import time
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MODULE_ROOT / "backend" / "python"))
sys.path.insert(0, str(_MODULE_ROOT))

# Mirror the launcher's config so we exercise the real path.
# noinspection PyUnresolvedReferences
from run_reduced_storm_suite import CONFIG, SCOPE, _build_bbox_config  # noqa: E402

import numpy as np  # noqa: E402

# noinspection PyUnresolvedReferences
from reduced_storm_suite.geo.bbox_filter import apply_bbox_filter  # noqa: E402
# noinspection PyUnresolvedReferences
from reduced_storm_suite.workflows.rss_selection import (  # noqa: E402
    _load_pipeline_data, _load_forced_indices, _remap_forced_indices,
)
# noinspection PyUnresolvedReferences
from reduced_storm_suite.sampling.pca import reduce_output  # noqa: E402
# noinspection PyUnresolvedReferences
from reduced_storm_suite.workflows._ab_sweep import run_ab_sweep  # noqa: E402
# noinspection PyUnresolvedReferences
from reduced_storm_suite.config.defaults import RSS_SELECTION_DEFAULTS  # noqa: E402


def main() -> None:
    cfg = dict(RSS_SELECTION_DEFAULTS)
    cfg.update(CONFIG)
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / SCOPE / "fixed")

    if SCOPE == "local":
        print("[bench] Applying bbox filter ...")
        bbox = _build_bbox_config()
        res = apply_bbox_filter(bbox, cfg["h5_path"], cfg["output_dir"])
        cfg["bbox_node_col_indices"] = res["node_col_indices"]
        forced_orig = _load_forced_indices(cfg)
        bbox_arr = np.asarray(res["storm_indices"], dtype=int)
        cfg["bbox_storm_indices"] = (
            np.union1d(bbox_arr, forced_orig) if forced_orig is not None else bbox_arr
        )
    else:
        print("[bench] Regional scope - skipping bbox filter.")
        forced_orig = _load_forced_indices(cfg)

    print("[bench] Loading data ...")
    X_full, Y, HC, _, _ = _load_pipeline_data(cfg)
    forced = _remap_forced_indices(cfg, forced_orig)

    X = X_full[:, cfg["x_select_columns"]]
    k = (len(forced) if forced is not None else 0) + cfg["k_additional"]
    print(f"[bench] X={X.shape}  Y={Y.shape}  HC={HC.shape}  k={k}  "
          f"forced={len(forced) if forced is not None else 0}")

    print("[bench] PCA ...")
    Y_r, _ = reduce_output(Y, cfg["pca_variance_threshold"])
    print(f"[bench] Y_r={Y_r.shape}")

    grid = cfg["alpha_beta_grid"]
    sample = cfg.get("ab_search_node_sample")

    if sample is not None and sample < Y.shape[1]:
        rng = np.random.default_rng(cfg["random_seed"])
        sel = rng.choice(Y.shape[1], size=int(sample), replace=False); sel.sort()
        Y_ab, HC_ab = Y[:, sel], HC[sel, :]
    else:
        Y_ab, HC_ab = Y, HC

    common = dict(
        X=X, Y=Y_ab, Y_r=Y_r, HC_bench=HC_ab, tbl_aer=cfg["TBL_AER"],
        k=k, seed=cfg["random_seed"], forced=forced,
        dry_thr=cfg["dry_threshold"], min_wet=cfg.get("min_wet_storms", 2),
        dsw_method=cfg.get("dsw_method", 1),
    )

    print(f"\n[bench] Running α/β sweep - sequential "
          f"(workers=1, nodes={Y_ab.shape[1]}/{Y.shape[1]}) ...")
    t0 = time.perf_counter()
    seq = run_ab_sweep(grid, workers=1, **common)
    seq_t = time.perf_counter() - t0
    print(f"[bench]   sequential: {seq_t:.2f} s  "
          f"({len(grid)} pts → {seq_t/len(grid):.2f} s/pt)")

    print(f"\n[bench] Running α/β sweep - parallel (workers=auto) ...")
    t0 = time.perf_counter()
    par = run_ab_sweep(grid, workers=None, **common)
    par_t = time.perf_counter() - t0
    print(f"[bench]   parallel:   {par_t:.2f} s  "
          f"→ {seq_t/par_t:.2f}× speedup over sequential")

    diffs = [abs(s["score"] - p["score"]) for s, p in zip(seq, par)]
    print(f"\n[bench] Max score divergence seq vs par: {max(diffs):.3e}  "
          f"(should be ~0)")


if __name__ == "__main__":
    main()
