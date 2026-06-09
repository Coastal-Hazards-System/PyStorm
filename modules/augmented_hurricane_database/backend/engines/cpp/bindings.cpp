// bindings.cpp — pybind11 bindings for the _gpm GP-metamodel kernel engine.
//
// Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>
//
// Exposes a single hot-loop accelerator:
//   corr(A, B, theta, p) -> R   (power-exponential correlation, OpenMP)
// The GP algebra (Cholesky, GLS, likelihood gradient) stays in NumPy/SciPy,
// which dispatch the O(n^3) work to LAPACK; only this O(m*n*d) kernel benefits
// from a hand-written parallel C++ loop.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "gpm_kernel.hpp"

namespace py = pybind11;

static py::array_t<double> corr(py::array_t<double, py::array::c_style | py::array::forcecast> A,
                                py::array_t<double, py::array::c_style | py::array::forcecast> B,
                                py::array_t<double, py::array::c_style | py::array::forcecast> theta,
                                double p) {
    auto bufA = A.request(), bufB = B.request(), bufT = theta.request();
    if (bufA.ndim != 2 || bufB.ndim != 2)
        throw std::runtime_error("A and B must be 2-D");
    const std::size_t m = static_cast<std::size_t>(bufA.shape[0]);
    const std::size_t d = static_cast<std::size_t>(bufA.shape[1]);
    const std::size_t n = static_cast<std::size_t>(bufB.shape[0]);
    if (static_cast<std::size_t>(bufB.shape[1]) != d)
        throw std::runtime_error("A and B must share the feature dimension");
    if (static_cast<std::size_t>(bufT.shape[0]) != d)
        throw std::runtime_error("theta length must equal the feature dimension");

    auto R = py::array_t<double>({m, n});
    // Resolve all buffer pointers BEFORE releasing the GIL — buffer_info /
    // request() are Python C-API calls and must not run without the GIL.
    double* Rptr = static_cast<double*>(R.request().ptr);
    const double* Aptr = static_cast<const double*>(bufA.ptr);
    const double* Bptr = static_cast<const double*>(bufB.ptr);
    const double* Tptr = static_cast<const double*>(bufT.ptr);
    {
        py::gil_scoped_release release;   // pure C++ compute, no Python access
        gpm::corr(Aptr, Bptr, Tptr, p, m, n, d, Rptr);
    }
    return R;
}

PYBIND11_MODULE(_gpm, mod) {
    mod.doc() = "GP-metamodel C++ kernel (power-exponential correlation, OpenMP)";
    mod.def("corr", &corr, py::arg("A"), py::arg("B"), py::arg("theta"), py::arg("p"),
            "Power-exponential correlation matrix R_ij = exp(-sum_k theta_k |A_ik-B_jk|^p).");
}
