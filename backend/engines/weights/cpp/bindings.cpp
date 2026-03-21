#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dsw_core.hpp"
#include "qbm_core.hpp"
#include "thread_pool.hpp"

namespace py = pybind11;

// ─── Helper: transpose Y_sub [k x m] row-major → Y_T [m x k] node-major ─────
// Parallelized for large matrices.
static std::vector<double> transpose_to_node_major(
    const double* Y, int k, int m, int n_threads = 0
) {
    if (n_threads <= 0) n_threads = threading::default_threads();
    std::vector<double> Y_T(static_cast<size_t>(m) * k);
    threading::parallel_for(m, n_threads, [&](int /*tid*/, int start, int end) {
        for (int node = start; node < end; ++node) {
            for (int j = 0; j < k; ++j) {
                Y_T[node * k + j] = Y[j * m + node];
            }
        }
    });
    return Y_T;
}

// ─── Helper: compute node weights (mirrors Python _compute_node_weights) ─────

static py::array_t<double> compute_node_weights_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    int method,
    double dry_thr
) {
    py::buffer_info y_buf = Y_sub.request();
    if (y_buf.ndim != 2)
        throw std::runtime_error("Y_sub must be 2-D [k x m]");

    int k = static_cast<int>(y_buf.shape[0]);
    int m = static_cast<int>(y_buf.shape[1]);

    py::array_t<double> out(m);
    double* w = static_cast<double*>(out.request().ptr);

    if (method == 1) {
        for (int i = 0; i < m; ++i) w[i] = 1.0;
    } else if (method == 3) {
        // Transpose for cache-friendly column access
        auto Y_T = transpose_to_node_major(
            static_cast<const double*>(y_buf.ptr), k, m);

        threading::parallel_for(m, threading::default_threads(),
            [&](int /*tid*/, int start, int end) {
            for (int node = start; node < end; ++node) {
                const double* col = &Y_T[node * k];
                double sum = 0.0, sum2 = 0.0;
                for (int j = 0; j < k; ++j) {
                    double v = (dsw::is_nan(col[j]) || col[j] <= dry_thr)
                               ? 0.0 : col[j];
                    sum  += v;
                    sum2 += v * v;
                }
                double mean = sum / k;
                w[node] = sum2 / k - mean * mean;
            }
        });
    } else {
        throw std::runtime_error("compute_node_weights: method must be 1 or 3 (use method=2 with nullptr node_w)");
    }
    return out;
}

// ─── compute_global_dsw ──────────────────────────────────────────────────────

static py::array_t<double> compute_global_dsw_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> HC_bench,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr,
    int min_wet_storms,
    int method,
    py::object node_w_obj,  // None for method 2
    int n_threads
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();

    if (y_buf.ndim != 2) throw std::runtime_error("Y_sub must be 2-D");
    if (h_buf.ndim != 2) throw std::runtime_error("HC_bench must be 2-D");

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    if (h_buf.shape[0] != m || h_buf.shape[1] != n_aer)
        throw std::runtime_error("HC_bench shape mismatch with Y_sub/tbl_aer");

    const double* HC  = static_cast<const double*>(h_buf.ptr);
    const double* AER = static_cast<const double*>(a_buf.ptr);

    auto Y_T = transpose_to_node_major(
        static_cast<const double*>(y_buf.ptr), k, m, n_threads);

    const double* nw_ptr = nullptr;
    py::array_t<double> nw_arr;
    if (!node_w_obj.is_none()) {
        nw_arr = node_w_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        nw_ptr = static_cast<const double*>(nw_arr.request().ptr);
    }

    py::array_t<double> out(k);
    double* out_ptr = static_cast<double*>(out.request().ptr);

    dsw::compute_global_dsw(Y_T.data(), k, m, HC, AER, n_aer,
                            dry_thr, min_wet_storms, method,
                            nw_ptr, out_ptr, n_threads);
    return out;
}

// ─── reconstruct_hc ──────────────────────────────────────────────────────────

static py::array_t<double> reconstruct_hc_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> DSW_global,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr,
    int n_threads
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info d_buf = DSW_global.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    auto Y_T = transpose_to_node_major(
        static_cast<const double*>(y_buf.ptr), k, m, n_threads);

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    dsw::reconstruct_hc(
        Y_T.data(), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr, out_ptr, n_threads);

    return out;
}

// ─── evaluate_hc_metrics ─────────────────────────────────────────────────────

static py::dict evaluate_hc_metrics_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> HC_bench,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr,
    int min_wet_storms,
    int method,
    py::object node_w_obj,
    int n_threads
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    const double* HC  = static_cast<const double*>(h_buf.ptr);
    const double* AER = static_cast<const double*>(a_buf.ptr);

    auto Y_T = transpose_to_node_major(
        static_cast<const double*>(y_buf.ptr), k, m, n_threads);

    const double* nw_ptr = nullptr;
    py::array_t<double> nw_arr;
    if (!node_w_obj.is_none()) {
        nw_arr = node_w_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        nw_ptr = static_cast<const double*>(nw_arr.request().ptr);
    }

    double mean_bias, mean_unc, mean_rmse;
    dsw::evaluate_hc_metrics(Y_T.data(), k, m, HC, AER, n_aer,
                             dry_thr, min_wet_storms, method,
                             nw_ptr,
                             &mean_bias, &mean_unc, &mean_rmse,
                             n_threads);

    py::dict result;
    result["mean_bias"]        = mean_bias;
    result["mean_uncertainty"] = mean_unc;
    result["mean_rmse"]        = mean_rmse;
    return result;
}


// ─── QBM: compute_bias_aer ───────────────────────────────────────────────────

static py::array_t<double> compute_qbm_bias_aer_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> DSW_global,
    py::array_t<double, py::array::c_style | py::array::forcecast> HC_bench,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr,
    int n_threads
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info d_buf = DSW_global.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    auto Y_T = transpose_to_node_major(
        static_cast<const double*>(y_buf.ptr), k, m, n_threads);

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    qbm::compute_bias_aer(
        Y_T.data(), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr, out_ptr, n_threads);

    return out;
}

// ─── QBM: compute_bias_response ──────────────────────────────────────────────

static py::array_t<double> compute_qbm_bias_response_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> DSW_global,
    py::array_t<double, py::array::c_style | py::array::forcecast> HC_bench,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr,
    py::array_t<double, py::array::c_style | py::array::forcecast> inter_grid,
    double win_frac,
    double ramp_frac,
    int n_threads
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info d_buf = DSW_global.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();
    py::buffer_info g_buf = inter_grid.request();

    int k       = static_cast<int>(y_buf.shape[0]);
    int m       = static_cast<int>(y_buf.shape[1]);
    int n_aer   = static_cast<int>(a_buf.shape[0]);
    int n_inter = static_cast<int>(g_buf.shape[0]);

    auto Y_T = transpose_to_node_major(
        static_cast<const double*>(y_buf.ptr), k, m, n_threads);

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    qbm::compute_bias_response(
        Y_T.data(), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr,
        static_cast<const double*>(g_buf.ptr), n_inter,
        win_frac, ramp_frac,
        out_ptr, n_threads);

    return out;
}


// ─── Module definition ───────────────────────────────────────────────────────

PYBIND11_MODULE(_dsw_cpp, m) {
    m.doc() = "C++ accelerated DSW/QBM back-computation and HC reconstruction";

    m.def("compute_node_weights", &compute_node_weights_py,
          py::arg("Y_sub"), py::arg("method"), py::arg("dry_thr"),
          "Compute per-node aggregation weights (method 1 or 3).");

    m.def("default_threads", &threading::default_threads,
          "Return the default thread count (hardware_concurrency clamped to [1,256]).");

    m.def("compute_global_dsw", &compute_global_dsw_py,
          py::arg("Y_sub"), py::arg("HC_bench"), py::arg("tbl_aer"),
          py::arg("dry_thr"), py::arg("min_wet_storms"), py::arg("method"),
          py::arg("node_w") = py::none(),
          py::arg("n_threads") = 0,
          "Back-compute global DSW per storm [k].");

    m.def("reconstruct_hc", &reconstruct_hc_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("tbl_aer"),
          py::arg("dry_thr"),
          py::arg("n_threads") = 0,
          "Reconstruct hazard curves [m x N_AER] via JPM integration.");

    m.def("evaluate_hc_metrics", &evaluate_hc_metrics_py,
          py::arg("Y_sub"), py::arg("HC_bench"), py::arg("tbl_aer"),
          py::arg("dry_thr"), py::arg("min_wet_storms"), py::arg("method"),
          py::arg("node_w") = py::none(),
          py::arg("n_threads") = 0,
          "Full DSW pipeline → {mean_bias, mean_uncertainty, mean_rmse}.");

    m.def("compute_qbm_bias_aer", &compute_qbm_bias_aer_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("HC_bench"),
          py::arg("tbl_aer"), py::arg("dry_thr"),
          py::arg("n_threads") = 0,
          "Compute AER-mode QBM bias table [m x N_AER].");

    m.def("compute_qbm_bias_response", &compute_qbm_bias_response_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("HC_bench"),
          py::arg("tbl_aer"), py::arg("dry_thr"),
          py::arg("inter_grid"), py::arg("win_frac"), py::arg("ramp_frac"),
          py::arg("n_threads") = 0,
          "Compute response-mode QBM bias table [m x N_AER].");
}
