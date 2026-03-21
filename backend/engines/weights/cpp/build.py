#!/usr/bin/env python
"""
Standalone build script for _dsw_cpp extension module.

Usage:
    python build.py          # build in-place
    python build.py clean    # remove build artefacts

Requires: pybind11 (pip install pybind11)

Build strategies (tried in order):
  1. setuptools Extension (MSVC on Windows, gcc/clang elsewhere)
  2. CMake (if available)
  3. Direct g++/clang++ invocation (uses lld on Windows to avoid MinGW ld bugs)

Developed by: Norberto C. Nadal-Caraballo, PhD
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_build"
MODULE_NAME = "_dsw_cpp"


def _get_ext_suffix():
    """Return the platform extension suffix, e.g. '.cp312-win_amd64.pyd'."""
    import importlib.machinery
    return importlib.machinery.EXTENSION_SUFFIXES[0]


def _build_setuptools():
    """Build using setuptools (most robust with MSVC on Windows)."""
    import pybind11
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    ext = Extension(
        MODULE_NAME,
        sources=[str(HERE / "bindings.cpp")],
        include_dirs=[
            pybind11.get_include(),
            str(HERE),
        ],
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
    cmd.build_lib = str(BUILD_DIR / "lib")
    cmd.ensure_finalized()
    cmd.run()

    lib_dir = Path(cmd.build_lib)
    for f in lib_dir.rglob(f"{MODULE_NAME}*"):
        dest = HERE / f.name
        shutil.copy2(f, dest)
        print(f"[build.py] Installed {dest}")
        return

    print("[build.py] WARNING: compiled module not found in build output")


def _build_cmake():
    """Build using CMake."""
    BUILD_DIR.mkdir(exist_ok=True)

    cmake_args = [
        "cmake",
        str(HERE),
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

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

    for ext in (".pyd", ".so", ".dylib"):
        for path in BUILD_DIR.rglob(f"{MODULE_NAME}*{ext}"):
            dest = HERE / path.name
            shutil.copy2(path, dest)
            print(f"[build.py] Installed {dest}")
            return

    for path in (BUILD_DIR / "Release").glob(f"{MODULE_NAME}*"):
        dest = HERE / path.name
        shutil.copy2(path, dest)
        print(f"[build.py] Installed {dest}")
        return

    print("[build.py] WARNING: compiled module not found in build output")


def _build_direct():
    """Build using direct compiler invocation (two-step: compile + link)."""
    import pybind11

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("No C++ compiler found on PATH")

    includes = pybind11.get_include()
    py_includes = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True).strip()

    ext_suffix = _get_ext_suffix()
    output = HERE / f"{MODULE_NAME}{ext_suffix}"
    BUILD_DIR.mkdir(exist_ok=True)
    obj_file = BUILD_DIR / "bindings.o"

    # Step 1: compile to object file
    compile_cmd = [
        cxx,
        "-c", "-O3", "-Wall", "-std=c++17", "-fPIC",
        f"-I{includes}",
        f"-I{py_includes}",
        f"-I{HERE}",
        str(HERE / "bindings.cpp"),
        "-o", str(obj_file),
    ]
    print(f"[build.py] Compiling with {cxx} ...")
    subprocess.check_call(compile_cmd)

    # Step 2: link
    link_cmd = [
        cxx,
        "-shared",
        "-static-libgcc", "-static-libstdc++",
    ]

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
    print(f"[build.py] Built {output}")


def build():
    # Strategy 1: setuptools (handles MSVC on Windows automatically)
    try:
        print("[build.py] Trying setuptools build ...")
        _build_setuptools()
        return
    except Exception as e:
        print(f"[build.py] setuptools build failed: {e}")

    # Strategy 2: CMake
    if shutil.which("cmake"):
        try:
            print("[build.py] Trying CMake build ...")
            _build_cmake()
            return
        except Exception as e:
            print(f"[build.py] CMake build failed: {e}")

    # Strategy 3: direct compiler (uses lld on Windows)
    print("[build.py] Trying direct compiler build ...")
    _build_direct()


def clean():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"[build.py] Removed {BUILD_DIR}")
    for f in HERE.glob(f"{MODULE_NAME}*"):
        if f.suffix in (".pyd", ".so", ".dylib") or ".cpython" in f.name:
            f.unlink()
            print(f"[build.py] Removed {f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        build()
