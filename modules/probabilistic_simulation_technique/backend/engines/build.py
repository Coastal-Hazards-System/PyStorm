#!/usr/bin/env python
"""build.py — standalone build helper for the _pst pybind11 extension.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Builds the C++ kernel defined in ``PSTBootstrap.hpp`` + ``pst_bindings.cpp``
and installs the resulting extension into the expanded orchestration package
(``backend/python/probabilistic_simulation_technique/``) per CyHAN v2.0
§16.2 / §16.5.

Usage
-----
    python build.py          # build in-place
    python build.py clean    # remove build artefacts

Requires: pybind11 (pip install pybind11)

Build strategies (tried in order)
---------------------------------
  1. setuptools Extension (MSVC on Windows, gcc/clang elsewhere)
  2. CMake (if available)
  3. Direct g++/clang++ invocation (uses lld on Windows when present)
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE      = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_build"
PYTHON_PKG_DIR = HERE / ".." / "python" / "probabilistic_simulation_technique"
PYTHON_PKG_DIR = PYTHON_PKG_DIR.resolve()

MODULE_NAME = "_pst"
SOURCE_FILE = HERE / "pst_bindings.cpp"


def _get_ext_suffix() -> str:
    import importlib.machinery
    return importlib.machinery.EXTENSION_SUFFIXES[0]


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

    if sys.platform == "win32":
        ext.extra_compile_args = ["/std:c++17", "/O2", "/EHsc"]
    else:
        ext.extra_compile_args = ["-std=c++17", "-O3"]

    dist = Distribution({"ext_modules": [ext]})
    dist.parse_config_files()

    cmd = build_ext(dist)
    cmd.inplace = False
    cmd.build_temp = str(BUILD_DIR / "temp")
    cmd.build_lib  = str(BUILD_DIR / "lib")
    cmd.ensure_finalized()
    cmd.run()

    lib_dir = Path(cmd.build_lib)
    for produced in lib_dir.rglob(f"{MODULE_NAME}*"):
        _install_artifact(produced)
        return

    print("[build.py] WARNING: compiled module not found in build output")


def _build_cmake() -> None:
    BUILD_DIR.mkdir(exist_ok=True)

    cmake_args = ["cmake", str(HERE), f"-DPython3_EXECUTABLE={sys.executable}"]
    if shutil.which("ninja"):
        cmake_args += ["-G", "Ninja"]
    elif sys.platform == "win32" and not shutil.which("cl"):
        if shutil.which("g++") or shutil.which("gcc"):
            cmake_args += ["-G", "MinGW Makefiles"]

    print(f"[build.py] Configuring in {BUILD_DIR} ...")
    subprocess.check_call(cmake_args, cwd=BUILD_DIR)

    print("[build.py] Building ...")
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"], cwd=BUILD_DIR)

    # The CMakeLists install() target writes directly into PYTHON_PKG_DIR;
    # invoke the install step so users only need a single command.
    print("[build.py] Installing ...")
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", "Release"], cwd=BUILD_DIR)


def _build_direct() -> None:
    import pybind11

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("No C++ compiler found on PATH")

    includes    = pybind11.get_include()
    py_includes = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True).strip()

    ext_suffix = _get_ext_suffix()
    output = BUILD_DIR / f"{MODULE_NAME}{ext_suffix}"
    BUILD_DIR.mkdir(exist_ok=True)
    obj_file = BUILD_DIR / "pst_bindings.o"

    compile_cmd = [
        cxx,
        "-c", "-O3", "-Wall", "-std=c++17", "-fPIC",
        f"-I{includes}",
        f"-I{py_includes}",
        f"-I{HERE}",
        str(SOURCE_FILE),
        "-o", str(obj_file),
    ]
    print(f"[build.py] Compiling with {cxx} ...")
    subprocess.check_call(compile_cmd)

    link_cmd = [cxx, "-shared", "-static-libgcc", "-static-libstdc++"]
    if sys.platform == "win32":
        if shutil.which("ld.lld"):
            link_cmd += ["-fuse-ld=lld"]
        link_cmd += ["-Wl,-Bstatic", "-lwinpthread", "-Wl,-Bdynamic"]

    link_cmd += ["-o", str(output), str(obj_file)]

    if sys.platform == "win32":
        import sysconfig
        py_lib_dir = sysconfig.get_config_var("installed_base") + "/libs"
        ver = f"{sys.version_info.major}{sys.version_info.minor}"
        link_cmd += [f"-L{py_lib_dir}", f"-lpython{ver}"]

    print(f"[build.py] Linking ...")
    subprocess.check_call(link_cmd)

    _install_artifact(output)


def build() -> None:
    try:
        print("[build.py] Trying setuptools build ...")
        _build_setuptools()
        return
    except Exception as e:
        print(f"[build.py] setuptools build failed: {e}")

    if shutil.which("cmake"):
        try:
            print("[build.py] Trying CMake build ...")
            _build_cmake()
            return
        except Exception as e:
            print(f"[build.py] CMake build failed: {e}")

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
