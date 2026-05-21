#pragma once
/**
 * kmedoids_core.hpp
 * =================
 * Header-only k-medoids PAM with FastPAM1 SWAP optimisation.
 *
 * BUILD  – maximin (farthest-first) initialisation, with optional forced medoids.
 * SWAP   – FastPAM1: maintains nearest / second-nearest medoid distances so each
 *          candidate swap cost is O(n) instead of O(n·k).
 *
 * All inputs/outputs use raw pointers or std::vector; the pybind11 layer in
 * bindings.cpp handles numpy ↔ C++ conversions.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace kmedoids {

// ─── helpers ────────────────────────────────────────────────────────────────

inline double D_ij(const double* D, int n, int i, int j) {
    return D[static_cast<std::size_t>(i) * n + j];
}

/**
 * For every point, compute the index of the nearest and second-nearest medoid,
 * plus the distances d1 and d2.
 */
inline void compute_assignments(
    const double* D, int n,
    const std::vector<int>& medoids,
    std::vector<int>& nearest,      // [n]
    std::vector<double>& d1,        // [n]
    std::vector<int>& second,       // [n]
    std::vector<double>& d2         // [n]
) {
    int k = static_cast<int>(medoids.size());
    for (int j = 0; j < n; ++j) {
        double best1 = std::numeric_limits<double>::max();
        double best2 = std::numeric_limits<double>::max();
        int idx1 = -1, idx2 = -1;
        for (int mi = 0; mi < k; ++mi) {
            double dist = D_ij(D, n, j, medoids[mi]);
            if (dist < best1) {
                best2 = best1;  idx2 = idx1;
                best1 = dist;   idx1 = mi;
            } else if (dist < best2) {
                best2 = dist;   idx2 = mi;
            }
        }
        nearest[j] = idx1;
        d1[j]      = best1;
        second[j]  = idx2;
        d2[j]      = best2;
    }
}

// ─── BUILD: maximin initialisation ──────────────────────────────────────────

/**
 * Select k medoids using farthest-first traversal.
 * If forced_indices is non-empty, those are placed first and the remaining
 * (k - forced.size()) slots are filled greedily.
 */
inline std::vector<int> build(
    const double* D, int n, int k,
    uint64_t seed,
    const std::vector<int>& forced
) {
    std::vector<int> selected;
    selected.reserve(k);

    // Minimum distance from each point to the nearest selected medoid.
    std::vector<double> min_d(n, std::numeric_limits<double>::max());

    if (!forced.empty()) {
        for (int f : forced) selected.push_back(f);
        // Initialise min_d from forced medoids.
        for (int j = 0; j < n; ++j) {
            for (int f : forced) {
                double dist = D_ij(D, n, j, f);
                if (dist < min_d[j]) min_d[j] = dist;
            }
        }
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(0, n - 1);
        int first = dist(rng);
        selected.push_back(first);
        for (int j = 0; j < n; ++j)
            min_d[j] = D_ij(D, n, j, first);
    }

    if (static_cast<int>(selected.size()) >= k) {
        selected.resize(k);
        return selected;
    }

    while (static_cast<int>(selected.size()) < k) {
        // Pick the point farthest from any selected medoid.
        int nxt = 0;
        double best = -1.0;
        for (int j = 0; j < n; ++j) {
            if (min_d[j] > best) {
                best = min_d[j];
                nxt  = j;
            }
        }
        selected.push_back(nxt);
        // Update min_d.
        for (int j = 0; j < n; ++j) {
            double dist = D_ij(D, n, j, nxt);
            if (dist < min_d[j]) min_d[j] = dist;
        }
    }

    return selected;
}

// ─── SWAP: FastPAM1 ────────────────────────────────────────────────────────

/**
 * Full PAM with FastPAM1 swap optimisation.
 *
 * Returns the final medoid indices (sorted).
 */
inline std::vector<int> pam(
    const double* D, int n, int k,
    uint64_t seed,
    const std::vector<int>& forced
) {
    std::vector<int> medoids = build(D, n, k, seed, forced);

    // Which medoid indices are forced (index into medoids[] vector)?
    std::vector<bool> is_forced(k, false);
    if (!forced.empty()) {
        // forced entries occupy the first forced.size() slots after build().
        for (int i = 0; i < static_cast<int>(forced.size()) && i < k; ++i)
            is_forced[i] = true;
    }

    // Membership lookup: is point p a medoid?
    std::vector<bool> is_medoid(n, false);
    for (int m : medoids) is_medoid[m] = true;

    // Assignment arrays.
    std::vector<int>    nearest(n), second_nearest(n);
    std::vector<double> d1(n), d2(n);
    compute_assignments(D, n, medoids, nearest, d1, second_nearest, d2);

    bool improved = true;
    while (improved) {
        improved = false;

        // For each swappable medoid × each non-medoid candidate, compute
        // the cost delta using FastPAM1 logic.
        double best_delta = 0.0;
        int    best_mi    = -1;
        int    best_cand  = -1;

        for (int mi = 0; mi < k; ++mi) {
            if (is_forced[mi]) continue;

            for (int cand = 0; cand < n; ++cand) {
                if (is_medoid[cand]) continue;

                double delta = 0.0;
                for (int j = 0; j < n; ++j) {
                    double d_j_cand = D_ij(D, n, j, cand);
                    if (nearest[j] == mi) {
                        // j's nearest medoid is the one being removed.
                        // New cost = min(d(j, cand), d2[j]).
                        double new_cost = (d_j_cand < d2[j]) ? d_j_cand : d2[j];
                        delta += new_cost - d1[j];
                    } else {
                        // j's nearest medoid stays; check if candidate is closer.
                        if (d_j_cand < d1[j]) {
                            delta += d_j_cand - d1[j];
                        }
                        // else: no change for this point.
                    }
                }

                if (delta < best_delta - 1e-10) {
                    best_delta = delta;
                    best_mi    = mi;
                    best_cand  = cand;
                }
            }
        }

        if (best_mi >= 0) {
            // Perform the swap.
            is_medoid[medoids[best_mi]] = false;
            medoids[best_mi] = best_cand;
            is_medoid[best_cand] = true;

            // Recompute assignments.
            compute_assignments(D, n, medoids, nearest, d1, second_nearest, d2);
            improved = true;
        }
    }

    std::sort(medoids.begin(), medoids.end());
    return medoids;
}

} // namespace kmedoids
