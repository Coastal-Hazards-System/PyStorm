# `_jdm` C++ engine - Weibull bootstrap kernel

Accelerates the JDM compute hot spot: the per-CRL jitter bootstrap of the Dp
Weibull marginal. The kernel resamples a peak sample with replacement, jitters each
replicate by the local order-statistic spacing (rejecting any replicate below the
truncation floor), and fits a two-parameter Weibull by maximum likelihood to each.

| File | Role |
|------|------|
| `JDMBootstrap.hpp` | Header-only kernel (`jdm::weibull_bootstrap`, `jdm::weibull_mle`). |
| `jdm_bindings.cpp` | pybind11 conduit publishing `_jdm.weibull_bootstrap`; releases the GIL. |
| `build.py` | Standalone builder (setuptools, then CMake, then a direct compiler call). |
| `CMakeLists.txt` | CMake build, installs `_jdm` into `backend/python/joint_distribution_model/`. |

## Build

```bash
python backend/engines/cpp/build.py          # build + install in-place
python backend/engines/cpp/build.py clean     # remove artefacts
```

Needs `pybind11` (`pip install pybind11`) and a C++17 toolchain (MSVC or MinGW on
Windows, gcc/clang elsewhere). The engine is optional: when `_jdm` is not built the
module falls back to an equivalent pure-NumPy bootstrap (vectorized Weibull MLE) in
`marginals.fit_weibull_boot`, so every workflow runs with or without a compiler.

## Contract

```
weibull_bootstrap(sample[n], n_boot, th, seed) -> params[n_boot, 2]   # (scale A, shape k)
```

Arrays in, arrays out; deterministic given `seed`. Because the kernel releases the
GIL, the orchestrator parallelizes the per-CRL fits with threads when `_jdm` is
present (avoiding the process-spawn and pickling overhead), and with processes
otherwise.
