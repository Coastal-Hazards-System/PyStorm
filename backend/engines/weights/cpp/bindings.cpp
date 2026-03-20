#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dsw_core.hpp"
#include "qbm_core.hpp"

namespace py = pybind11;

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
    const double* Y = static_cast<const double*>(y_buf.ptr);

    py::array_t<double> out(m);
    double* w = static_cast<double*>(out.request().ptr);

    if (method == 1) {
        for (int i = 0; i < m; ++i) w[i] = 1.0;
    } else if (method == 3) {
        // Variance of cleaned column (NaN/dry → 0)
        for (int node = 0; node < m; ++node) {
            double sum = 0.0, sum2 = 0.0;
            for (int j = 0; j < k; ++j) {
                double val = Y[j * m + node];
                double v = (dsw::is_nan(val) || val <= dry_thr) ? 0.0 : val;
                sum  += v;
                sum2 += v * v;
            }
            double mean = sum / k;
            w[node] = sum2 / k - mean * mean;
        }
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
    py::object node_w_obj  // None for method 2
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

    const double* Y   = static_cast<const double*>(y_buf.ptr);
    const double* HC  = static_cast<const double*>(h_buf.ptr);
    const double* AER = static_cast<const double*>(a_buf.ptr);

    const double* nw_ptr = nullptr;
    py::array_t<double> nw_arr;
    if (!node_w_obj.is_none()) {
        nw_arr = node_w_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        nw_ptr = static_cast<const double*>(nw_arr.request().ptr);
    }

    py::array_t<double> out(k);
    double* out_ptr = static_cast<double*>(out.request().ptr);

    dsw::compute_global_dsw(Y, k, m, HC, AER, n_aer,
                            dry_thr, min_wet_storms, method,
                            nw_ptr, out_ptr);
    return out;
}

// ─── reconstruct_hc ──────────────────────────────────────────────────────────

static py::array_t<double> reconstruct_hc_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> Y_sub,
    py::array_t<double, py::array::c_style | py::array::forcecast> DSW_global,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbl_aer,
    double dry_thr
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info d_buf = DSW_global.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    dsw::reconstruct_hc(
        static_cast<const double*>(y_buf.ptr), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr, out_ptr);

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
    py::object node_w_obj
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    const double* Y   = static_cast<const double*>(y_buf.ptr);
    const double* HC  = static_cast<const double*>(h_buf.ptr);
    const double* AER = static_cast<const double*>(a_buf.ptr);

    const double* nw_ptr = nullptr;
    py::array_t<double> nw_arr;
    if (!node_w_obj.is_none()) {
        nw_arr = node_w_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        nw_ptr = static_cast<const double*>(nw_arr.request().ptr);
    }

    double mean_bias, mean_unc, mean_rmse;
    dsw::evaluate_hc_metrics(Y, k, m, HC, AER, n_aer,
                             dry_thr, min_wet_storms, method,
                             nw_ptr,
                             &mean_bias, &mean_unc, &mean_rmse);

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
    double dry_thr
) {
    py::buffer_info y_buf = Y_sub.request();
    py::buffer_info d_buf = DSW_global.request();
    py::buffer_info h_buf = HC_bench.request();
    py::buffer_info a_buf = tbl_aer.request();

    int k     = static_cast<int>(y_buf.shape[0]);
    int m     = static_cast<int>(y_buf.shape[1]);
    int n_aer = static_cast<int>(a_buf.shape[0]);

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    qbm::compute_bias_aer(
        static_cast<const double*>(y_buf.ptr), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr, out_ptr);

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
    double ramp_frac
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

    py::array_t<double> out({m, n_aer});
    double* out_ptr = static_cast<double*>(out.request().ptr);

    qbm::compute_bias_response(
        static_cast<const double*>(y_buf.ptr), k, m,
        static_cast<const double*>(d_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(a_buf.ptr), n_aer,
        dry_thr,
        static_cast<const double*>(g_buf.ptr), n_inter,
        win_frac, ramp_frac,
        out_ptr);

    return out;
}


// ─── Module definition ───────────────────────────────────────────────────────

PYBIND11_MODULE(_dsw_cpp, m) {
    m.doc() = "C++ accelerated DSW/QBM back-computation and HC reconstruction";

    m.def("compute_node_weights", &compute_node_weights_py,
          py::arg("Y_sub"), py::arg("method"), py::arg("dry_thr"),
          "Compute per-node aggregation weights (method 1 or 3).");

    m.def("compute_global_dsw", &compute_global_dsw_py,
          py::arg("Y_sub"), py::arg("HC_bench"), py::arg("tbl_aer"),
          py::arg("dry_thr"), py::arg("min_wet_storms"), py::arg("method"),
          py::arg("node_w") = py::none(),
          "Back-compute global DSW per storm [k].");

    m.def("reconstruct_hc", &reconstruct_hc_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("tbl_aer"),
          py::arg("dry_thr"),
          "Reconstruct hazard curves [m x N_AER] via JPM-OS integration.");

    m.def("evaluate_hc_metrics", &evaluate_hc_metrics_py,
          py::arg("Y_sub"), py::arg("HC_bench"), py::arg("tbl_aer"),
          py::arg("dry_thr"), py::arg("min_wet_storms"), py::arg("method"),
          py::arg("node_w") = py::none(),
          "Full DSW pipeline → {mean_bias, mean_uncertainty, mean_rmse}.");

    m.def("compute_qbm_bias_aer", &compute_qbm_bias_aer_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("HC_bench"),
          py::arg("tbl_aer"), py::arg("dry_thr"),
          "Compute AER-mode QBM bias table [m x N_AER].");

    m.def("compute_qbm_bias_response", &compute_qbm_bias_response_py,
          py::arg("Y_sub"), py::arg("DSW_global"), py::arg("HC_bench"),
          py::arg("tbl_aer"), py::arg("dry_thr"),
          py::arg("inter_grid"), py::arg("win_frac"), py::arg("ramp_frac"),
          "Compute response-mode QBM bias table [m x N_AER].");
}
