#pragma once
/**
 * dsw_core.hpp
 * ============
 * Header-only C++ implementation of the DSW (Discrete Storm Weight)
 * back-computation and JPM-OS hazard-curve reconstruction.
 *
 * Algorithm
 * ---------
 * Step 1 — Nodal DSW: at each node, sort k surges descending, interpolate
 *          benchmark HC in log-AER space, finite-difference → nodal DSWs.
 * Step 2 — Global DSW: weighted mean across active nodes per storm.
 * Step 3 — HC reconstruction: at each node, sort by descending surge,
 *          cumsum global DSWs, interpolate surge vs cumulative AER onto tbl_aer.
 * Step 4 — Residual metrics: bias, uncertainty, RMSE.
 *
 * All node iterations are independent → embarrassingly parallel.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "thread_pool.hpp"

namespace dsw {

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

inline bool is_nan(double x) { return std::isnan(x); }
inline bool is_finite(double x) { return std::isfinite(x); }

// ─── Log-linear interpolation ────────────────────────────────────────────────

/**
 * Piecewise-linear interpolation:  x_vals → y_vals  evaluated at query.
 * x_vals must be sorted (ascending or descending — caller decides).
 * Out-of-range queries return NaN.
 *
 * This replaces scipy.interp1d with bounds_error=False, fill_value=NaN.
 */
inline double interp_linear(
    const double* x_vals, const double* y_vals, int n,
    double query
) {
    if (n < 2) return NaN;

    bool ascending = x_vals[n - 1] > x_vals[0];

    if (ascending) {
        if (query < x_vals[0] || query > x_vals[n - 1]) return NaN;
        // Binary search for interval
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (x_vals[mid] <= query) lo = mid;
            else hi = mid;
        }
        double t = (query - x_vals[lo]) / (x_vals[hi] - x_vals[lo]);
        return y_vals[lo] + t * (y_vals[hi] - y_vals[lo]);
    } else {
        // Descending x_vals
        if (query > x_vals[0] || query < x_vals[n - 1]) return NaN;
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (x_vals[mid] >= query) lo = mid;
            else hi = mid;
        }
        double t = (query - x_vals[lo]) / (x_vals[hi] - x_vals[lo]);
        return y_vals[lo] + t * (y_vals[hi] - y_vals[lo]);
    }
}

/**
 * Map surge values → AER via log-linear interpolation of benchmark HC.
 *
 * hc_surge[N_AER] : benchmark surge at this node
 * hc_aer[N_AER]   : benchmark AER levels
 * query[q]        : surge values to look up
 * out[q]          : output AER values (NaN for out-of-range)
 *
 * Interpolation is in (surge, log(AER)) space, then exp().
 * Handles NaN entries and duplicate surge values.
 */
inline void surge_to_aer(
    const double* hc_surge, const double* hc_aer, int n_aer,
    const double* query, int q,
    double* out
) {
    // Collect valid (non-NaN) entries
    std::vector<double> s_valid, log_a_valid;
    s_valid.reserve(n_aer);
    log_a_valid.reserve(n_aer);
    for (int i = 0; i < n_aer; ++i) {
        if (!is_nan(hc_surge[i]) && !is_nan(hc_aer[i]) && hc_aer[i] > 0.0) {
            s_valid.push_back(hc_surge[i]);
            log_a_valid.push_back(std::log(hc_aer[i]));
        }
    }

    if (s_valid.size() < 2) {
        for (int i = 0; i < q; ++i) out[i] = NaN;
        return;
    }

    // Deduplicate by surge value (keep first occurrence)
    // Sort by surge ascending for consistent interpolation
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
            la_uniq.push_back(log_a_valid[idx]);
            prev_s = s_valid[idx];
        }
    }

    int n = static_cast<int>(s_uniq.size());
    if (n < 2) {
        for (int i = 0; i < q; ++i) out[i] = NaN;
        return;
    }

    // Interpolate each query point
    for (int i = 0; i < q; ++i) {
        double log_a = interp_linear(s_uniq.data(), la_uniq.data(), n, query[i]);
        out[i] = is_nan(log_a) ? NaN : std::exp(log_a);
    }
}

/**
 * JPM-OS integration at a single node: reconstruct the hazard curve.
 *
 * resp[k]    : storm surge responses
 * dsw[k]     : global DSW per storm
 * tbl_aer[N] : output AER grid
 * dry_thr    : dry threshold
 * out[N]     : output reconstructed surge at each AER level
 */
inline void jpm_integrate(
    const double* resp, const double* dsw, int k,
    const double* tbl_aer, int n_aer,
    double dry_thr,
    double* out
) {
    // Collect valid entries
    std::vector<int> valid_idx;
    valid_idx.reserve(k);
    for (int j = 0; j < k; ++j) {
        if (!is_nan(resp[j]) && !is_nan(dsw[j]) && resp[j] > dry_thr)
            valid_idx.push_back(j);
    }

    if (static_cast<int>(valid_idx.size()) < 2) {
        for (int i = 0; i < n_aer; ++i) out[i] = NaN;
        return;
    }

    // Sort valid entries by descending surge
    std::sort(valid_idx.begin(), valid_idx.end(),
              [&](int a, int b) { return resp[a] > resp[b]; });

    // Compute cumulative AER and log(cumAER)
    int nv = static_cast<int>(valid_idx.size());
    std::vector<double> surge(nv), log_cum_aer(nv);
    double cum = 0.0;
    int n_finite = 0;

    for (int i = 0; i < nv; ++i) {
        surge[i] = resp[valid_idx[i]];
        cum += dsw[valid_idx[i]];
        double lca = std::log(cum);
        if (is_finite(lca)) {
            surge[n_finite] = surge[i];
            log_cum_aer[n_finite] = lca;
            n_finite++;
        }
    }

    if (n_finite < 2) {
        for (int i = 0; i < n_aer; ++i) out[i] = NaN;
        return;
    }

    // Deduplicate by log_cum_aer (keep first = highest surge)
    std::vector<double> lca_uniq, s_uniq;
    lca_uniq.reserve(n_finite);
    s_uniq.reserve(n_finite);
    double prev = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_finite; ++i) {
        if (log_cum_aer[i] != prev) {
            lca_uniq.push_back(log_cum_aer[i]);
            s_uniq.push_back(surge[i]);
            prev = log_cum_aer[i];
        }
    }

    int nu = static_cast<int>(lca_uniq.size());
    if (nu < 2) {
        for (int i = 0; i < n_aer; ++i) out[i] = NaN;
        return;
    }

    // Interpolate: log(tbl_aer) → surge
    for (int i = 0; i < n_aer; ++i) {
        double log_q = std::log(tbl_aer[i]);
        out[i] = interp_linear(lca_uniq.data(), s_uniq.data(), nu, log_q);
    }
}


// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Compute global DSW per storm.
 *
 * Y_T      : [m x k] node-major  (Y_T[node*k + j] = storm j at node)
 * HC_bench : [m x N] row-major  (node-major)
 * tbl_aer  : [N]
 * node_w   : [m]  per-node weights (method 1: all 1s, method 3: variance)
 *            For method 2 pass nullptr.
 * method   : 1=equal, 2=surge-weighted, 3=variance-weighted
 *
 * Returns DSW_global via out_dsw[k].
 */
inline void compute_global_dsw(
    const double* Y_T, int k, int m,
    const double* HC_bench, const double* tbl_aer, int n_aer,
    double dry_thr, int min_wet_storms, int method,
    const double* node_w,  // [m] or nullptr for method 2
    double* out_dsw,       // [k] output
    int n_threads = 0
) {
    if (n_threads <= 0) n_threads = threading::default_threads();

    // Per-thread accumulators
    int nt = std::max(1, std::min(n_threads, m));
    std::vector<std::vector<double>> t_dsw_sum(nt, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> t_weight_sum(nt, std::vector<double>(k, 0.0));

    threading::parallel_for(m, nt, [&](int tid, int start, int end) {
        std::vector<int> sort_buf(k);

        for (int node = start; node < end; ++node) {
            // Node's k values are contiguous: Y_T[node*k .. node*k + k-1]
            const double* resp = &Y_T[node * k];

            // Count wet storms
            int wet_count = 0;
            for (int j = 0; j < k; ++j) {
                if (!is_nan(resp[j]) && resp[j] > dry_thr)
                    wet_count++;
            }
            if (wet_count < min_wet_storms)
                continue;

            std::iota(sort_buf.begin(), sort_buf.end(), 0);
            std::sort(sort_buf.begin(), sort_buf.end(),
                      [&](int a, int b) {
                          if (is_nan(resp[a])) return false;
                          if (is_nan(resp[b])) return true;
                          return resp[a] > resp[b];
                      });

            std::vector<int> inv_perm(k);
            for (int r = 0; r < k; ++r)
                inv_perm[sort_buf[r]] = r;

            std::vector<double> resp_sorted(k);
            for (int r = 0; r < k; ++r)
                resp_sorted[r] = resp[sort_buf[r]];

            std::vector<int> valid_pos;
            valid_pos.reserve(k);
            for (int r = 0; r < k; ++r) {
                if (!is_nan(resp_sorted[r]) && resp_sorted[r] > dry_thr)
                    valid_pos.push_back(r);
            }
            int n_valid = static_cast<int>(valid_pos.size());
            if (n_valid < 2) continue;

            std::vector<double> query(n_valid);
            for (int i = 0; i < n_valid; ++i)
                query[i] = resp_sorted[valid_pos[i]];

            std::vector<double> aer_q(n_valid);
            surge_to_aer(
                &HC_bench[node * n_aer], tbl_aer, n_aer,
                query.data(), n_valid,
                aer_q.data());

            bool all_nan = true;
            for (int i = 0; i < n_valid; ++i) {
                if (!is_nan(aer_q[i])) { all_nan = false; break; }
            }
            if (all_nan) continue;

            std::vector<double> dsw_valid(n_valid);
            dsw_valid[0] = is_nan(aer_q[0]) ? 0.0 : aer_q[0];
            for (int i = 1; i < n_valid; ++i) {
                if (is_nan(aer_q[i]) || is_nan(aer_q[i - 1]))
                    dsw_valid[i] = 0.0;
                else
                    dsw_valid[i] = aer_q[i] - aer_q[i - 1];
            }

            for (int i = 0; i < n_valid; ++i) {
                if (dsw_valid[i] < 0.0) dsw_valid[i] = 0.0;
            }

            std::vector<double> dsw_sorted(k, 0.0);
            for (int i = 0; i < n_valid; ++i)
                dsw_sorted[valid_pos[i]] = dsw_valid[i];

            for (int j = 0; j < k; ++j) {
                double dsw_orig = dsw_sorted[inv_perm[j]];
                bool active = dsw_orig > 0.0;

                if (method == 2) {
                    double s = resp[j];
                    double sw = (is_nan(s) || s < 0.0) ? 0.0 : s;
                    t_dsw_sum[tid][j] += dsw_orig * sw;
                    if (active) t_weight_sum[tid][j] += sw;
                } else {
                    double w = node_w[node];
                    t_dsw_sum[tid][j] += dsw_orig * w;
                    if (active) t_weight_sum[tid][j] += w;
                }
            }
        }
    });

    // Merge per-thread accumulators
    std::vector<double> dsw_sum(k, 0.0);
    std::vector<double> weight_sum(k, 0.0);
    for (int t = 0; t < nt; ++t) {
        for (int j = 0; j < k; ++j) {
            dsw_sum[j]    += t_dsw_sum[t][j];
            weight_sum[j] += t_weight_sum[t][j];
        }
    }

    // Final: weighted mean per storm
    for (int j = 0; j < k; ++j) {
        out_dsw[j] = (weight_sum[j] > 0.0)
            ? dsw_sum[j] / weight_sum[j]
            : NaN;
    }
}

/**
 * Reconstruct hazard curves at all nodes via JPM-OS integration.
 *
 * Y_T        : [m x k] node-major
 * DSW_global : [k]
 * tbl_aer    : [N]
 * out_hc     : [m x N] row-major output
 */
inline void reconstruct_hc(
    const double* Y_T, int k, int m,
    const double* DSW_global,
    const double* tbl_aer, int n_aer,
    double dry_thr,
    double* out_hc,  // [m x N]
    int n_threads = 0
) {
    if (n_threads <= 0) n_threads = threading::default_threads();

    threading::parallel_for(m, n_threads, [&](int tid, int start, int end) {
        for (int node = start; node < end; ++node) {
            // Node's k values are contiguous — pass directly
            jpm_integrate(
                &Y_T[node * k], DSW_global, k,
                tbl_aer, n_aer,
                dry_thr,
                &out_hc[node * n_aer]);
        }
    });
}

/**
 * Compute residual metrics between reconstructed and benchmark HCs.
 *
 * HC_recon : [m x N] row-major
 * HC_bench : [m x N] row-major
 *
 * Returns mean_bias, mean_uncertainty, mean_rmse via output pointers.
 */
inline void hc_residual_metrics(
    const double* HC_recon, const double* HC_bench,
    int m, int n_aer,
    double* mean_bias, double* mean_uncertainty, double* mean_rmse,
    int n_threads = 0
) {
    if (n_threads <= 0) n_threads = threading::default_threads();
    int nt = std::max(1, std::min(n_threads, m));

    std::vector<double> t_bias(nt, 0.0), t_unc(nt, 0.0), t_rmse(nt, 0.0);
    std::vector<int> t_count(nt, 0);

    threading::parallel_for(m, nt, [&](int tid, int start, int end) {
        for (int node = start; node < end; ++node) {
            double sum_r = 0.0, sum_r2 = 0.0;
            int cnt = 0;
            for (int a = 0; a < n_aer; ++a) {
                double r = HC_recon[node * n_aer + a] - HC_bench[node * n_aer + a];
                if (!is_nan(r)) {
                    sum_r += r;
                    sum_r2 += r * r;
                    cnt++;
                }
            }
            if (cnt == 0) continue;

            double node_bias = sum_r / cnt;
            double node_mse  = sum_r2 / cnt;
            double node_var  = node_mse - node_bias * node_bias;
            double node_unc  = (node_var > 0.0) ? std::sqrt(node_var) : 0.0;
            double node_rmse = std::sqrt(node_mse);

            t_bias[tid]  += node_bias;
            t_unc[tid]   += node_unc;
            t_rmse[tid]  += node_rmse;
            t_count[tid]++;
        }
    });

    double sum_bias = 0.0, sum_unc = 0.0, sum_rmse = 0.0;
    int n_valid_nodes = 0;
    for (int t = 0; t < nt; ++t) {
        sum_bias += t_bias[t];
        sum_unc  += t_unc[t];
        sum_rmse += t_rmse[t];
        n_valid_nodes += t_count[t];
    }

    if (n_valid_nodes > 0) {
        *mean_bias = sum_bias / n_valid_nodes;
        *mean_uncertainty = sum_unc / n_valid_nodes;
        *mean_rmse = sum_rmse / n_valid_nodes;
    } else {
        *mean_bias = NaN;
        *mean_uncertainty = NaN;
        *mean_rmse = NaN;
    }
}

/**
 * Full evaluate_hc_metrics: compute_global_dsw + reconstruct_hc + metrics.
 *
 * Convenience function that runs the complete pipeline.
 * node_w must be pre-computed by the caller (Python side).
 */
inline void evaluate_hc_metrics(
    const double* Y_T, int k, int m,
    const double* HC_bench,
    const double* tbl_aer, int n_aer,
    double dry_thr, int min_wet_storms, int method,
    const double* node_w,
    double* mean_bias, double* mean_uncertainty, double* mean_rmse,
    int n_threads = 0
) {
    std::vector<double> DSW_global(k);
    compute_global_dsw(Y_T, k, m, HC_bench, tbl_aer, n_aer,
                       dry_thr, min_wet_storms, method, node_w,
                       DSW_global.data(), n_threads);

    std::vector<double> HC_recon(static_cast<size_t>(m) * n_aer);
    reconstruct_hc(Y_T, k, m, DSW_global.data(),
                   tbl_aer, n_aer, dry_thr, HC_recon.data(), n_threads);

    hc_residual_metrics(HC_recon.data(), HC_bench, m, n_aer,
                        mean_bias, mean_uncertainty, mean_rmse, n_threads);
}

} // namespace dsw
