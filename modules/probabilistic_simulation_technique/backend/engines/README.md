# probabilistic_simulation_technique - C++ engine

Header-only truncated-noise bootstrap kernel for the PST inner loop, plus the
`_pst` pybind11 binding.

| Artifact                | Role                                                            |
|-------------------------|-----------------------------------------------------------------|
| `PSTBootstrap.hpp`      | `pst::bootstrap()` - descending POT + spacing perturbation     |
| `pst_bindings.cpp`      | pybind11 module → `_pst.bootstrap_truncated()`                 |
| `CMakeLists.txt`        | CMake target; installs into `../python/probabilistic_simulation_technique/` |
| `build.py`              | Standalone build helper (setuptools → CMake → direct compiler) |

## Build

```bash
python build.py            # in-place, installs into the Python package
python build.py clean      # remove _build/ and the installed .pyd / .so
```

Requires `pybind11` (`pip install pybind11`) and a C++17 toolchain (MSVC on
Windows, gcc/clang elsewhere).

## CyHAN v2.2 install destination

Per CyHAN v2.2 §16.2 / §16.5, the compiled `_pst.cp<py>-<plat>.{pyd,so}`
extension lives inside the expanded orchestration package, so callers write:

```python
from probabilistic_simulation_technique import _pst
```

Both `CMakeLists.txt` (via `install(TARGETS … LIBRARY DESTINATION …)`) and
`build.py` (via direct copy) honour that destination.
