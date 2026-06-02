"""events — hydrograph and peak-gap segmenters (pure Python fallback).

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Public API
----------
  segment_hydrograph(values, times_sec, exceed_idx, interevent_sec) -> ndarray[int64]
      Group consecutive exceedances by time-gap > interevent_sec; per group
      keep argmax(values).
  segment_peak_gap(values, times_sec, exceed_idx, interevent_sec) -> ndarray[int64]
      Sequential filter: drop a sample whose preceding (chronological)
      exceedance is within `interevent_sec` AND has greater-or-equal value.
"""

import numpy as np


def segment_hydrograph(
    values:         np.ndarray,
    times_sec:      np.ndarray,
    exceed_idx:     np.ndarray,
    interevent_sec: float,
) -> np.ndarray:
    if exceed_idx.size == 0:
        return np.empty(0, dtype=np.int64)
    exceed_idx = np.asarray(exceed_idx, dtype=np.int64)
    t_exc      = times_sec[exceed_idx]
    gaps       = np.diff(t_exc)
    group_breaks = np.where(gaps > interevent_sec)[0] + 1
    group_starts = np.concatenate(([0], group_breaks))
    group_ends   = np.concatenate((group_breaks, [exceed_idx.size]))

    out = np.empty(group_starts.size, dtype=np.int64)
    for g, (s, e) in enumerate(zip(group_starts, group_ends)):
        idx_block       = exceed_idx[s:e]
        out[g]          = int(idx_block[np.argmax(values[idx_block])])
    return out


def segment_peak_gap(
    values:         np.ndarray,
    times_sec:      np.ndarray,
    exceed_idx:     np.ndarray,
    interevent_sec: float,
) -> np.ndarray:
    if exceed_idx.size == 0:
        return np.empty(0, dtype=np.int64)
    exceed_idx = np.asarray(exceed_idx, dtype=np.int64)
    keep = np.ones(exceed_idx.size, dtype=bool)
    for k in range(1, exceed_idx.size):
        dt = times_sec[exceed_idx[k]] - times_sec[exceed_idx[k - 1]]
        if dt < interevent_sec and values[exceed_idx[k]] <= values[exceed_idx[k - 1]]:
            keep[k] = False
    return exceed_idx[keep]
