"""_ab_sweep - parallel α/β grid sweep for RSS selection.

Author : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Each (α, β) iteration is independent: it builds a joint matrix, runs
k-medoids with the same forced indices and seed, then evaluates HC metrics.
Results are deterministic per (α, β) given the same inputs/seed - parallel
execution does not change numerics.

Two execution modes:
  - ``workers <= 1`` runs sequentially in-process (no spawn overhead).
  - ``workers > 1`` uses ``ProcessPoolExecutor`` with module-level globals
    initialised once per worker to avoid re-pickling Y / HC_bench per job.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from reduced_storm_suite.sampling.joint_matrix import build_joint_matrix
from reduced_storm_suite.sampling.kmedoids import select_kmedoids
from reduced_storm_suite.weights.dsw import evaluate_hc_metrics


# ---------------------------------------------------------------------------
# Worker-process state (populated by _init_worker via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(X, Y, Y_r, HC_bench, tbl_aer, k, seed, forced,
                 dry_thr, min_wet, dsw_method):
    """Stash heavy arrays + scalars as module globals on the worker process."""
    _W["X"]          = X
    _W["Y"]          = Y
    _W["Y_r"]        = Y_r
    _W["HC_bench"]   = HC_bench
    _W["tbl_aer"]    = tbl_aer
    _W["k"]          = k
    _W["seed"]       = seed
    _W["forced"]     = forced
    _W["dry_thr"]    = dry_thr
    _W["min_wet"]    = min_wet
    _W["dsw_method"] = dsw_method


def _evaluate_ab(ab):
    """Evaluate a single (α, β) - runs in worker or in-process."""
    alpha, beta = ab
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Z, _, _ = build_joint_matrix(_W["X"], _W["Y_r"], alpha, beta)
        idx = select_kmedoids(Z, _W["k"], _W["seed"],
                              forced_indices=_W["forced"])
        hc = evaluate_hc_metrics(
            _W["Y"][idx, :], _W["HC_bench"], _W["tbl_aer"],
            _W["dry_thr"], _W["min_wet"], dsw_method=_W["dsw_method"])
    score = abs(hc["mean_bias"]) + hc["mean_rmse"]
    return {"alpha": alpha, "beta": beta, **hc, "score": score}


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def run_ab_sweep(
    ab_grid,
    *,
    X, Y, Y_r, HC_bench, tbl_aer,
    k:          int,
    seed:       int,
    forced:     Optional[np.ndarray],
    dry_thr:    float,
    min_wet:    int,
    dsw_method: int,
    workers:    Optional[int] = None,
):
    """Run the α/β sweep, returning one result dict per grid point in order.

    workers : None → auto (min(cpu_count, len(grid))); 0 or 1 → sequential.
    """
    if workers is None:
        workers = min((os.cpu_count() or 1), len(ab_grid))
    workers = max(1, int(workers))

    init_args = (X, Y, Y_r, HC_bench, tbl_aer, k, seed, forced,
                 dry_thr, min_wet, dsw_method)

    if workers == 1 or len(ab_grid) == 1:
        _init_worker(*init_args)
        return [_evaluate_ab(ab) for ab in ab_grid]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        # map() preserves input order, matching the sequential print contract.
        return list(pool.map(_evaluate_ab, ab_grid))
