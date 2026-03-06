#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kmedoids_core.hpp"

namespace py = pybind11;

/**
 * Python-callable wrapper around kmedoids::pam().
 *
 * Parameters
 * ----------
 * D      : (n, n) float64 distance matrix (row-major, contiguous).
 * k      : number of medoids.
 * seed   : RNG seed for BUILD initialisation.
 * forced : int32 array of indices that must appear in the result (may be empty).
 *
 * Returns
 * -------
 * medoids : (k,) int32 array of selected row indices (sorted).
 */
py::array_t<int32_t> kmedoids_pam(
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    int k,
    uint64_t seed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> forced
) {
    py::buffer_info D_buf = D.request();
    if (D_buf.ndim != 2 || D_buf.shape[0] != D_buf.shape[1])
        throw std::runtime_error("D must be a square 2-D array");

    int n = static_cast<int>(D_buf.shape[0]);
    const double* D_ptr = static_cast<const double*>(D_buf.ptr);

    // Convert forced array.
    py::buffer_info f_buf = forced.request();
    std::vector<int> forced_vec;
    if (f_buf.size > 0) {
        const int32_t* f_ptr = static_cast<const int32_t*>(f_buf.ptr);
        forced_vec.assign(f_ptr, f_ptr + f_buf.size);
    }

    std::vector<int> result = kmedoids::pam(D_ptr, n, k, seed, forced_vec);

    // Copy to numpy.
    py::array_t<int32_t> out(static_cast<py::ssize_t>(result.size()));
    auto out_buf = out.request();
    int32_t* out_ptr = static_cast<int32_t*>(out_buf.ptr);
    for (std::size_t i = 0; i < result.size(); ++i)
        out_ptr[i] = static_cast<int32_t>(result[i]);

    return out;
}

PYBIND11_MODULE(_kmedoids_cpp, m) {
    m.doc() = "C++ accelerated k-medoids PAM with FastPAM1 optimisation";
    m.def("kmedoids_pam", &kmedoids_pam,
          py::arg("D"), py::arg("k"), py::arg("seed"), py::arg("forced"),
          "Run PAM k-medoids on precomputed distance matrix D.\n\n"
          "Parameters\n"
          "----------\n"
          "D      : (n, n) float64 distance matrix\n"
          "k      : number of medoids\n"
          "seed   : RNG seed\n"
          "forced : int32 array of forced medoid indices (may be empty)\n\n"
          "Returns\n"
          "-------\n"
          "medoids : (k,) int32 sorted array of selected indices");
}
