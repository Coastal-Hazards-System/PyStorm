#!/usr/bin/env python
"""build.py — standalone build helper for the _gpm pybind11 extension.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Builds the C++ power-exponential correlation kernel (``gpm_kernel.hpp`` +
``bindings.cpp``) and installs the extension into the GP-metamodel package
(``backend/python/augmented_hurricane_database/gp_metamodel/``). The kernel is
the dominant cost of GP prediction and of the per-evaluation R build; the GP
algebra stays in NumPy/SciPy (LAPACK). OpenMP is enabled when available.

Usage
-----
    python build.py          # build in-place
    python build.py clean    # remove build artefacts

Requires: pybind11 (pip install pybind11)
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_build"
PYTHON_PKG_DIR = (HERE / ".." / ".." / "python" / "augmented_hurricane_database"
                  / "gp_metamodel").resolve()

MODULE_NAME = "_gpm"
SOURCE_FILE = HERE / "bindings.cpp"


def _install_artifact(src: Path) -> None:
    PYTHON_PKG_DIR.mkdir(parents=True, exist_ok=True)
    dest = PYTHON_PKG_DIR / src.name
    shutil.copy2(src, dest)
    print(f"[build.py] Installed {dest}")


def _build_setuptools() -> None:
    import pybind11
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    ext = Extension(
        MODULE_NAME,
        sources=[str(SOURCE_FILE)],
        include_dirs=[pybind11.get_include(), str(HERE)],
        language="c++",
    )
    # OpenMP-parallel + fast-math: the kernel is the hot loop of prediction and
    # the per-evaluation R build. The GP releases the GIL during the C++ compute,
    # so threads scale freely (measured ~10-34x over the NumPy broadcast).
    if sys.platform == "win32":
        ext.extra_compile_args = ["/std:c++17", "/O2", "/EHsc", "/openmp", "/fp:fast"]
    elif sys.platform == "darwin":
        # Apple clang lacks OpenMP out of the box; build serial (kernel still runs).
        ext.extra_compile_args = ["-std=c++17", "-O3", "-ffast-math"]
    else:
        ext.extra_compile_args = ["-std=c++17", "-O3", "-ffast-math",
                                  "-funroll-loops", "-fopenmp"]
        ext.extra_link_args = ["-fopenmp"]

    dist = Distribution({"ext_modules": [ext]})
    dist.parse_config_files()
    cmd = build_ext(dist)
    cmd.inplace = False
    cmd.build_temp = str(BUILD_DIR / "temp")
    cmd.build_lib = str(BUILD_DIR / "lib")
    cmd.ensure_finalized()
    cmd.run()

    lib_dir = Path(cmd.build_lib)
    for produced in lib_dir.rglob(f"{MODULE_NAME}*"):
        _install_artifact(produced)
        return
    print("[build.py] WARNING: compiled module not found in build output")


def _build_direct() -> None:
    import pybind11

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("No C++ compiler found on PATH")
    import importlib.machinery
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    py_includes = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True).strip()
    BUILD_DIR.mkdir(exist_ok=True)
    output = BUILD_DIR / f"{MODULE_NAME}{ext_suffix}"
    obj = BUILD_DIR / "bindings.o"
    omp = [] if sys.platform == "darwin" else ["-fopenmp"]
    subprocess.check_call([cxx, "-c", "-O3", "-ffast-math", "-funroll-loops",
                           "-std=c++17", "-fPIC", *omp,
                           f"-I{pybind11.get_include()}", f"-I{py_includes}",
                           f"-I{HERE}", str(SOURCE_FILE), "-o", str(obj)])

    link = [cxx, "-shared", *omp]
    py_dll = None
    if sys.platform == "win32":
        # mingw cannot use the MSVC import lib (pythonXY.lib); link the CPython
        # DLL directly. "-static" pulls libgcc/libstdc++/libgomp/winpthread into
        # the .pyd so it has no external mingw-runtime DLL dependency at import.
        # (The GIL is released during compute, so the static OpenMP runtime does
        # not conflict — the earlier crash was a GIL-misuse bug, now fixed.)
        import os
        ver = f"{sys.version_info.major}{sys.version_info.minor}"
        py_dll = os.path.join(sys.base_prefix, f"python{ver}.dll")
        link += ["-static"]
    else:
        link += ["-static-libgcc", "-static-libstdc++"]
    link += ["-o", str(output), str(obj)]
    if py_dll:
        link.append(py_dll)
    subprocess.check_call(link)
    _install_artifact(output)


def build() -> None:
    try:
        print("[build.py] Trying setuptools build ...")
        _build_setuptools()
        return
    except Exception as e:                                       # noqa: BLE001
        print(f"[build.py] setuptools build failed: {e}")
    print("[build.py] Trying direct compiler build ...")
    _build_direct()


def clean() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"[build.py] Removed {BUILD_DIR}")
    for f in PYTHON_PKG_DIR.glob(f"{MODULE_NAME}*"):
        if f.suffix in (".pyd", ".so", ".dylib") or ".cpython" in f.name:
            f.unlink()
            print(f"[build.py] Removed {f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        build()
