#!/usr/bin/env python
"""PyStorm environment doctor.

Reports whether this interpreter can run every module and build the C++
kernels, BEFORE a run, so silent pure-Python fallbacks are never a surprise.

    python check_env.py

Exit code is non-zero if a required runtime package is missing. A missing C++
toolchain is only a warning: modules still run on the pure-Python fallback.
Output is plain ASCII so it is safe on any console.
"""
from __future__ import annotations

import importlib
import shutil
import sys

# (import name, pip name, why) -- the union of every module's runtime needs.
CORE = [
    ("numpy", "numpy", "arrays / linear algebra (all modules)"),
    ("scipy", "scipy", "stats and optimization (AHD, PST, RSS, JDM, CSH)"),
    ("pandas", "pandas", "tabular I/O (all modules)"),
    ("matplotlib", "matplotlib", "figures (all modules)"),
    ("pydantic", "pydantic", "config validation (most modules)"),
    ("requests", "requests", "data download: HURDAT2 / NOAA / Natural Earth"),
    ("pyproj", "pyproj", "best-track geodesy (AHD)"),
    ("h5py", "h5py", "tc_data.h5 bundle I/O (RSS)"),
    ("yaml", "PyYAML", "preprocess.yaml config (RSS)"),
    ("sklearn", "scikit-learn", "k-medoids selection (RSS)"),
    ("shapefile", "pyshp", "Natural Earth basemap maps (SCA)"),
]
BUILD = [
    ("pybind11", "pybind11", "build-time for the C++ kernels"),
    ("pytest", "pytest", "test suite"),
]

OK, MISSING = "[ ok ]", "[MISS]"


def _check(group):
    missing = []
    for mod, pip_name, why in group:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {OK} {pip_name:<16} {ver:<10} {why}")
        except Exception:
            missing.append(pip_name)
            print(f"  {MISSING} {pip_name:<16} {'--':<10} {why}")
    return missing


def main() -> int:
    print("PyStorm environment check")
    print("=" * 60)

    v = sys.version_info
    py_ok = (v.major, v.minor) >= (3, 10)
    print(f"Python {v.major}.{v.minor}.{v.micro}  "
          f"{'[ ok ]' if py_ok else '[MISS] need >= 3.10'}")
    print(f"  {sys.executable}")

    print("\nCore runtime packages:")
    miss_core = _check(CORE)

    print("\nBuild / test packages:")
    miss_build = _check(BUILD)

    print("\nC++ toolchain (optional; enables the fast kernels):")
    compilers = [c for c in ("cl", "gcc", "g++", "clang", "clang++") if shutil.which(c)]
    cmake = shutil.which("cmake")
    if compilers:
        print(f"  {OK} compiler        {', '.join(compilers)}")
    else:
        print(f"  [warn] no C++ compiler on PATH (MSVC / gcc / clang)")
    print(f"  {OK if cmake else '[warn]'} cmake           "
          f"{'found' if cmake else 'not found (build.py can fall back to setuptools)'}")

    print("\n" + "=" * 60)
    problems = []
    if not py_ok:
        problems.append("Python < 3.10")
    if miss_core:
        problems.append("missing core packages: " + ", ".join(miss_core))
    if miss_build:
        print("note: missing build/test packages (" + ", ".join(miss_build) +
              ") -- only needed to compile kernels or run tests.")
    if not compilers:
        print("note: no compiler found -- modules run on the pure-Python fallback.")

    if problems:
        print("RESULT: NOT READY -- " + "; ".join(problems))
        print("Fix with:  pip install -r requirements.txt")
        return 1
    print("RESULT: ready to run all modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
