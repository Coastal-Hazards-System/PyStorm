# peaks_over_threshold — C++ engine

Header-only kernel for the POT iterative threshold-search loop, plus the
`_pot` pybind11 binding.

| Artifact                 | Role                                                            |
|--------------------------|-----------------------------------------------------------------|
| `POTThresholdSearch.hpp` | `pot::find_threshold_for_target()` — percentile bisection + segmentation |
| `pot_bindings.cpp`       | pybind11 module → `_pot.find_threshold_for_target()`           |
| `CMakeLists.txt`         | CMake target; installs into `../../python/peaks_over_threshold/` |
| `build.py`               | Standalone build helper (setuptools → CMake → direct compiler) |

## Build

```bash
python build.py            # in-place, installs into the Python package
python build.py clean      # remove _build/ and the installed .pyd / .so
```

Requires `pybind11` (`pip install pybind11`) and a C++17 toolchain.

Callers import as:

```python
from peaks_over_threshold import _pot
```
