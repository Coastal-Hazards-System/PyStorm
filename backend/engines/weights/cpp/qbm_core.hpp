#pragma once
/**
 * qbm_core.hpp
 * =============
 * Header-only C++ implementation of Quantile Bias Mapping (QBM)
 * post-DSW bias correction.
 *
 * Two modes:
 *   - AER mode (default): invert benchmark HC to remap AER positions.
 *   - Response mode (legacy): compute per-storm response bias, smooth,
 *     and subtract from surge values.
 *
 * Reuses interpolation primitives from dsw_core.hpp.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "dsw_core.hpp"  // for dsw::interp_linear, dsw::NaN, dsw::is_nan, dsw::is_finite

namespace qbm {

using dsw::NaN;
using dsw::is_nan;
using dsw::is_finite;

// ─── Log-linear interpolation with extrapolation option ──────────────────

/**
 * Interpolation in (log(x), y) space.
 * x_known must be positive.  Sorted internally.
 *
 * extrapolate=true  → clamp to boundary values
 * extrapolate=false → NaN outside range
 */
inline void interp_log_linear(
    const double* x_known, const double* y_known, int n,
    const double* x_query, int q,
    bool extrapolate,
    double* out
) {
    if (n < 2) {
        for (int i = 0; i < q; ++i) out[i] = NaN;
        return;
    }

    // Sort by log(x) ascending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return x_known[a] < x_known[b]; });

    std::vector<double> lx(n), ys(n);
    for (int i = 0; i < n; ++i) {
        lx[i] = std::log(x_known[order[i]]);
        ys[i] = y_known[order[i]];
    }

    for (int i = 0; i < q; ++i) {
        double lq = std::log(x_query[i]);
        if (lq < lx[0]) {
            out[i] = extrapolate ? ys[0] : NaN;
        } else if (lq > lx[n - 1]) {
            out[i] = extrapolate ? ys[n - 1] : NaN;
        } else {
            out[i] = dsw::interp_linear(lx.data(), ys.data(), n, lq);
        }
    }
}

// ─── Invert benchmark HC: surge → AER ───────────────────────────────────

/**
 * Given benchmark surge values and AER levels, return AER for query surges.
 * Interpolates in (surge, log-AER) space.
 * extrapolate=false → NaN outside benchmark surge range.
 */
inline void invert_hc_bench(
    const double* hc_surge, const double* tbl_aer, int n_aer,
    const double* query_surge, int q,
    bool extrapolate,
    double* out_aer
) {
    // Collect valid entries
    std::vector<double> s_valid, la_valid;
    s_valid.reserve(n_aer);
    la_valid.reserve(n_aer);
    for (int i = 0; i < n_aer; ++i) {
        if (is_finite(hc_surge[i]) && tbl_aer[i] > 0.0) {
            s_valid.push_back(hc_surge[i]);
            la_valid.push_back(std::log(tbl_aer[i]));
        }
    }

    if (s_valid.size() < 2) {
        for (int i = 0; i < q; ++i) out_aer[i] = NaN;
        return;
    }

    // Sort by surge ascending, deduplicate
    std::vector<int> order(s_valid.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return s_valid[a] < s_valid[b]; });

    std::vector<double> s_uniq, la_uniq;
    s_uniq.reserve(s_valid.size());
    la_uniq.reserve(s_valid.size());
    double prev_s = -std::numeric_limits<double>::infinity();
    for (int idx : order) {
        if (s_valid[idx] != prev_s) {
            s_uniq.push_back(s_valid[idx]);
            la_uniq.push_back(la_valid[idx]);
            prev_s = s_valid[idx];
        }
    }

    int nu = static_cast<int>(s_uniq.size());
    if (nu < 2) {
        for (int i = 0; i < q; ++i) out_aer[i] = NaN;
        return;
    }

    for (int i = 0; i < q; ++i) {
        double la = dsw::interp_linear(s_uniq.data(), la_uniq.data(), nu, query_surge[i]);
        if (is_nan(la)) {
            if (extrapolate) {
                if (query_surge[i] < s_uniq[0])
                    la = la_uniq[0];
                else
                    la = la_uniq[nu - 1];
                out_aer[i] = std::exp(la);
            } else {
                out_aer[i] = NaN;
            }
        } else {
            out_aer[i] = std::exp(la);
        }
    }
}

// ─── Monotonicity enforcement ────────────────────────────────────────────

inline void enforce_monotonicity_aer_nondecreasing(double* aer, int n) {
    for (int j = 1; j < n; ++j)
        if (aer[j] < aer[j - 1]) aer[j] = aer[j - 1];
}

inline void enforce_monotonicity_surge_nonincreasing(double* surge, int n) {
    for (int j = 1; j < n; ++j)
        if (surge[j] > surge[j - 1]) surge[j] = surge[j - 1];
}

// ─── Gaussian smoothing ─────────────────────────────────────────────────

inline void gaussian_smooth(const double* bias_raw, int n,
                            double win_frac, double* out) {
    if (n < 3) {
        for (int i = 0; i < n; ++i) out[i] = bias_raw[i];
        return;
    }
    double sigma = (win_frac * n) / 6.0;
    if (sigma < 0.5) sigma = 0.5;

    for (int i = 0; i < n; ++i) {
        double wsum = 0.0, vsum = 0.0;
        for (int j = 0; j < n; ++j) {
            double d = static_cast<double>(j - i) / sigma;
            double w = std::exp(-0.5 * d * d);
            wsum += w;
            vsum += w * bias_raw[j];
        }
        out[i] = vsum / wsum;
    }
}

// ─── Ramp endpoints ─────────────────────────────────────────────────────

inline void ramp_endpoints(const double* bias_smooth, const double* bias_raw,
                           int n, double ramp_frac, double* out) {
    for (int i = 0; i < n; ++i) out[i] = bias_smooth[i];

    int m = static_cast<int>(std::round(ramp_frac * n));
    if (m < 3) m = 3;
    if (m > n / 2) m = n / 2;

    // Left ramp: blend from raw[0] to smooth
    for (int i = 0; i < m; ++i) {
        double t = static_cast<double>(i) / (m - 1);
        double a = 3 * t * t - 2 * t * t * t;
        out[i] = (1.0 - a) * bias_raw[0] + a * bias_smooth[i];
    }
    out[0] = bias_raw[0];

    // Right ramp: blend from smooth to raw[-1]
    for (int i = 0; i < m; ++i) {
        double t = static_cast<double>(i) / (m - 1);
        double a = 3 * t * t - 2 * t * t * t;
        int idx = n - m + i;
        out[idx] = (1.0 - a) * bias_smooth[idx] + a * bias_raw[n - 1];
    }
    out[n - 1] = bias_raw[n - 1];
}


// ─── AER-mode: compute bias table ────────────────────────────────────────

/**
 * Compute AER-mode bias table for all nodes.
 *
 * For each node:
 *   1. Sort storms descending by surge, compute cum_aer_global = cumsum(DSW)
 *   2. cum_aer_corrected = H_bench^{-1}(surge) — invert benchmark
 *   3. Enforce monotonicity (non-decreasing AER)
 *   4. log_delta = log(cum_aer_global) - log(cum_aer_corrected) at overlap
 *   5. Map log_delta to 22 standard AERs via log-linear interpolation
 *
 * out_bias : [m x n_aer] row-major
 */
inline void compute_bias_aer(
    const double* Y_sub, int k, int m,
    const double* DSW_global,
    const double* HC_bench,
    const double* tbl_aer, int n_aer,
    double dry_thr,
    double* out_bias  // [m x n_aer]
) {
    // Zero-initialise output
    for (int i = 0; i < m * n_aer; ++i) out_bias[i] = 0.0;

    for (int node = 0; node < m; ++node) {
        // Extract column
        std::vector<int> valid_idx;
        valid_idx.reserve(k);
        for (int j = 0; j < k; ++j) {
            double r = Y_sub[j * m + node];
            double d = DSW_global[j];
            if (!is_nan(r) && !is_nan(d) && r > dry_thr)
                valid_idx.push_back(j);
        }
        if (static_cast<int>(valid_idx.size()) < 2) continue;

        // Sort descending by surge
        std::sort(valid_idx.begin(), valid_idx.end(),
                  [&](int a, int b) {
                      return Y_sub[a * m + node] > Y_sub[b * m + node];
                  });

        int nv = static_cast<int>(valid_idx.size());
        std::vector<double> surge_s(nv), cum_aer_g(nv);
        double cum = 0.0;
        for (int i = 0; i < nv; ++i) {
            surge_s[i] = Y_sub[valid_idx[i] * m + node];
            cum += DSW_global[valid_idx[i]];
            cum_aer_g[i] = cum;
        }

        // Invert benchmark HC: surge → AER
        std::vector<double> cum_aer_corr(nv);
        invert_hc_bench(
            &HC_bench[node * n_aer], tbl_aer, n_aer,
            surge_s.data(), nv, false,
            cum_aer_corr.data());

        // Track overlap mask BEFORE filling and monotonicity
        std::vector<bool> overlap(nv, false);
        int n_overlap = 0;
        for (int i = 0; i < nv; ++i) {
            if (!is_nan(cum_aer_corr[i])) {
                overlap[i] = true;
                n_overlap++;
            } else {
                cum_aer_corr[i] = cum_aer_g[i];  // fill with global
            }
        }
        if (n_overlap < 2) continue;

        // Enforce non-decreasing AER
        enforce_monotonicity_aer_nondecreasing(cum_aer_corr.data(), nv);

        // Compute log-AER delta ONLY at original overlap positions
        std::vector<double> overlap_aer_g, overlap_delta;
        overlap_aer_g.reserve(n_overlap);
        overlap_delta.reserve(n_overlap);
        for (int i = 0; i < nv; ++i) {
            if (!overlap[i]) continue;
            if (cum_aer_g[i] > 0.0 && cum_aer_corr[i] > 0.0) {
                double ld = std::log(cum_aer_g[i]) - std::log(cum_aer_corr[i]);
                if (is_finite(ld)) {
                    overlap_aer_g.push_back(cum_aer_g[i]);
                    overlap_delta.push_back(ld);
                }
            }
        }
        if (static_cast<int>(overlap_aer_g.size()) < 2) continue;

        // Map to 22 standard AERs via log-linear interpolation (extrapolate)
        interp_log_linear(
            overlap_aer_g.data(), overlap_delta.data(),
            static_cast<int>(overlap_aer_g.size()),
            tbl_aer, n_aer, true,
            &out_bias[node * n_aer]);
    }
}


// ─── Response-mode: compute bias table ───────────────────────────────────

/**
 * Compute response-mode bias table for all nodes.
 *
 * For each node:
 *   1. Sort storms descending by surge, compute cum_aer_s = cumsum(DSW)
 *   2. Interpolate benchmark surge at each storm's cum_aer position
 *   3. bias_raw = surge - bench_surge (at overlap positions)
 *   4. Map to intermediate grid (inter_grid), smooth, ramp endpoints
 *   5. Map from intermediate grid to 22 standard AERs
 *
 * inter_grid     : [n_inter] intermediate AER grid (631 or 22)
 * out_bias       : [m x n_aer] row-major
 */
inline void compute_bias_response(
    const double* Y_sub, int k, int m,
    const double* DSW_global,
    const double* HC_bench,
    const double* tbl_aer, int n_aer,
    double dry_thr,
    const double* inter_grid, int n_inter,
    double win_frac, double ramp_frac,
    double* out_bias  // [m x n_aer]
) {
    for (int i = 0; i < m * n_aer; ++i) out_bias[i] = 0.0;

    for (int node = 0; node < m; ++node) {
        // Collect valid storms
        std::vector<int> valid_idx;
        valid_idx.reserve(k);
        for (int j = 0; j < k; ++j) {
            double r = Y_sub[j * m + node];
            double d = DSW_global[j];
            if (!is_nan(r) && !is_nan(d) && r > dry_thr)
                valid_idx.push_back(j);
        }
        if (static_cast<int>(valid_idx.size()) < 2) continue;

        // Sort descending by surge
        std::sort(valid_idx.begin(), valid_idx.end(),
                  [&](int a, int b) {
                      return Y_sub[a * m + node] > Y_sub[b * m + node];
                  });

        int nv = static_cast<int>(valid_idx.size());
        std::vector<double> surge_s(nv), cum_aer_s(nv);
        double cum = 0.0;
        for (int i = 0; i < nv; ++i) {
            surge_s[i] = Y_sub[valid_idx[i] * m + node];
            cum += DSW_global[valid_idx[i]];
            cum_aer_s[i] = cum;
        }

        // Benchmark surge at each storm's cum_aer — no extrapolation
        // Collect valid benchmark entries
        std::vector<double> bench_aer_v, bench_surge_v;
        bench_aer_v.reserve(n_aer);
        bench_surge_v.reserve(n_aer);
        for (int a = 0; a < n_aer; ++a) {
            if (is_finite(HC_bench[node * n_aer + a])) {
                bench_aer_v.push_back(tbl_aer[a]);
                bench_surge_v.push_back(HC_bench[node * n_aer + a]);
            }
        }
        if (static_cast<int>(bench_aer_v.size()) < 2) continue;

        std::vector<double> bench_at_storm(nv);
        interp_log_linear(
            bench_aer_v.data(), bench_surge_v.data(),
            static_cast<int>(bench_aer_v.size()),
            cum_aer_s.data(), nv, false,
            bench_at_storm.data());

        // Compute bias at overlap positions
        std::vector<double> overlap_aer, overlap_bias;
        overlap_aer.reserve(nv);
        overlap_bias.reserve(nv);
        for (int i = 0; i < nv; ++i) {
            if (is_finite(bench_at_storm[i])) {
                overlap_aer.push_back(cum_aer_s[i]);
                overlap_bias.push_back(surge_s[i] - bench_at_storm[i]);
            }
        }
        if (static_cast<int>(overlap_bias.size()) < 2) continue;

        // Map bias to intermediate grid via log-linear interpolation (extrapolate)
        std::vector<double> bias_inter(n_inter);
        interp_log_linear(
            overlap_aer.data(), overlap_bias.data(),
            static_cast<int>(overlap_aer.size()),
            inter_grid, n_inter, true,
            bias_inter.data());

        // Find valid range for smoothing
        std::vector<int> valid_n;
        valid_n.reserve(n_inter);
        for (int i = 0; i < n_inter; ++i) {
            if (is_finite(bias_inter[i])) valid_n.push_back(i);
        }

        if (static_cast<int>(valid_n.size()) >= 3) {
            int nv2 = static_cast<int>(valid_n.size());
            std::vector<double> raw_subset(nv2), smooth_subset(nv2);
            for (int i = 0; i < nv2; ++i)
                raw_subset[i] = bias_inter[valid_n[i]];

            gaussian_smooth(raw_subset.data(), nv2, win_frac, smooth_subset.data());

            std::vector<double> ramped(nv2);
            ramp_endpoints(smooth_subset.data(), raw_subset.data(),
                           nv2, ramp_frac, ramped.data());

            for (int i = 0; i < nv2; ++i)
                bias_inter[valid_n[i]] = ramped[i];
        }

        // Map from intermediate grid to 22 standard AERs
        if (inter_grid != tbl_aer && n_inter != n_aer) {
            interp_log_linear(
                inter_grid, bias_inter.data(), n_inter,
                tbl_aer, n_aer, true,
                &out_bias[node * n_aer]);
        } else {
            for (int a = 0; a < n_aer; ++a)
                out_bias[node * n_aer + a] = bias_inter[a];
        }
    }
}

} // namespace qbm
