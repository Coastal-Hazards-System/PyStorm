"""PCA (POD) dimensionality reduction on the surge response matrix Y.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Engine contract: arrays in, arrays out.  No config, no I/O.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
import numpy as np


def _prepare_y(
    Y: np.ndarray,
    dry_strategy: str,
    min_wet_fraction: float = 0.2,
) -> np.ndarray:
    """Resolve NaN (dry-node markers) in Y per the selected strategy.

    Strategies
    ----------
    "drop_always_dry" (default)
        Drop columns (nodes) that are NaN for every storm — they contribute
        zero variance to PCA anyway. Remaining NaN entries are zero-filled.
        Strictly better than "zero" (smaller covariance matrix, identical
        components on the kept columns).
    "zero"
        Replace every NaN with 0.0. Preserves node count.
    "node_mean"
        Replace each NaN with the mean of the wet (non-NaN) values at that
        same node. Columns with no wet values are dropped.
    "wet_only"
        Drop any column that has at least one NaN. Aggressive — focuses PCA
        on the "always-wet" subset of nodes.
    "wet_ratio_floor"
        Drop any column wet for fewer than "min_wet_fraction" of storms,
        then zero-fill remaining NaN. A middle ground between
        "drop_always_dry" (keeps every node wet for ≥1 storm) and "wet_only"
        (keeps only nodes wet for every storm). Suppresses the zero-padding
        wedge in basin-wide runs without discarding all partially-wet nodes.

    Parameters
    ----------
    min_wet_fraction : float
        Only used by "wet_ratio_floor". Minimum fraction of storms (0–1) a
        node must be wet for to be retained.
    """
    if not np.isnan(Y).any():
        return Y

    if dry_strategy == "zero":
        return np.where(np.isnan(Y), 0.0, Y)

    if dry_strategy == "wet_only":
        keep = ~np.isnan(Y).any(axis=0)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"    PCA wet_only: dropped {n_drop:,} nodes "
                  f"with any NaN; kept {int(keep.sum()):,}")
        return Y[:, keep]

    if dry_strategy == "wet_ratio_floor":
        n_storms = Y.shape[0]
        wet_frac = (~np.isnan(Y)).sum(axis=0) / n_storms
        keep = wet_frac >= min_wet_fraction
        n_drop = int((~keep).sum())
        Y2 = Y[:, keep]
        n_filled = int(np.isnan(Y2).sum())
        if n_drop or n_filled:
            print(f"    PCA wet_ratio_floor (min_wet_fraction="
                  f"{min_wet_fraction:.2f}): dropped {n_drop:,} nodes wet for "
                  f"<{min_wet_fraction*100:.0f}% of storms; zero-filled "
                  f"{n_filled:,} remaining NaN ({int(keep.sum()):,} nodes kept)")
        return np.where(np.isnan(Y2), 0.0, Y2)

    if dry_strategy == "node_mean":
        all_nan = np.isnan(Y).all(axis=0)
        Y2 = Y[:, ~all_nan]
        col_mean = np.nanmean(Y2, axis=0)
        # column means are guaranteed non-NaN because we dropped all-NaN cols
        inds = np.where(np.isnan(Y2))
        Y2[inds] = np.take(col_mean, inds[1])
        if int(all_nan.sum()):
            print(f"    PCA node_mean: dropped {int(all_nan.sum()):,} "
                  f"always-dry nodes; imputed {len(inds[0]):,} NaN entries")
        return Y2

    # default: drop_always_dry
    all_nan = np.isnan(Y).all(axis=0)
    Y2 = Y[:, ~all_nan]
    n_dropped = int(all_nan.sum())
    n_filled = int(np.isnan(Y2).sum())
    if n_dropped or n_filled:
        print(f"    PCA drop_always_dry: dropped {n_dropped:,} all-NaN nodes; "
              f"zero-filled {n_filled:,} remaining NaN entries "
              f"({Y2.shape[1]:,} nodes kept)")
    return np.where(np.isnan(Y2), 0.0, Y2)


def reduce_output(
    Y: np.ndarray,
    variance_threshold: float = 0.95,
    dry_strategy: str = "drop_always_dry",
    min_wet_fraction: float = 0.2,
) -> tuple[np.ndarray, PCA]:
    """Compress the surge response matrix via PCA.

    Parameters
    ----------
    Y : ndarray [n_storms x m_nodes]
        Surge matrix. NaN entries are interpreted as dry / missing.
    variance_threshold : float
        Fraction of variance to retain (passed to sklearn PCA).
    dry_strategy : str
        How to handle NaN entries. See ``_prepare_y`` for options.
        Default "drop_always_dry": drop columns that are 100% NaN, then
        zero-fill remaining NaN. Matches the DSW "dry = 0 surge" convention.
    min_wet_fraction : float
        Only used when dry_strategy == "wet_ratio_floor". Minimum fraction of
        storms (0–1) a node must be wet for to be retained.

    Returns
    -------
    Y_r : [n_storms x r]  PCA score matrix
    pca : fitted sklearn PCA object
    """
    Y_prepared = _prepare_y(Y, dry_strategy, min_wet_fraction)
    if Y_prepared.shape[1] == 0:
        hint = (f"lower pca_min_wet_fraction (currently {min_wet_fraction:.2f})"
                if dry_strategy == "wet_ratio_floor"
                else "use 'wet_ratio_floor' with a modest pca_min_wet_fraction "
                     "(e.g. 0.2), 'node_mean', or 'drop_always_dry'")
        raise ValueError(
            f"PCA pca_dry_strategy={dry_strategy!r} removed every node "
            f"({Y.shape[1]:,} -> 0), leaving nothing to decompose. In a "
            f"basin-wide run almost every node is dry for some storm, so "
            f"'wet_only' (drop any node ever dry) keeps nothing. Fix: {hint}.")

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    Y_r = pca.fit_transform(Y_prepared)
    return Y_r, pca
