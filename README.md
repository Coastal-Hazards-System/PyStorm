# PyStorm

### *A Modular Python + C++ Framework for Probabilistic Modeling of Coastal Storm Hazards*

---

## Overview

**PyStorm** is an open-source framework for probabilistic modeling and analysis,
designed for engineers, scientists, and researchers working on coastal hazard
quantification, stochastic storm simulation, life-cycle analysis, and extreme
value analysis. It pairs **Python** for orchestration, configuration, I/O, and
post-processing with **C++** engines for the inner numerical kernels that
dominate runtime — keeping workflows script-driven and reproducible while
leaving the heavy lifting to compiled code.

The architecture follows **CyHAN Standard v1.1** — a module-first decomposition
in which each capability is a self-contained vertical with its own C++ engine
and Python orchestration. See
[`docs/CyHAN-Standard-v1.1.md`](docs/CyHAN-Standard-v1.1.md). Source comments
and docstrings (Python and C++) follow the project-wide
[Comment & Docstring Standard](docs/PyStorm-Comment-Standard-v0.2.md).

---

## Architecture at a glance

Every module is a vertical with the same internal shape:

| Layer                  | Language | Role                                                                                            |
|------------------------|----------|-------------------------------------------------------------------------------------------------|
| `scripts/`             | Python   | Thin launchers — parse command-line arguments / read config, then call into the orchestration package. |
| `backend/python/<mod>` | Python   | Orchestration: data flow, I/O, configuration, diagnostics, plotting.                            |
| `backend/engines/cpp/` | C++      | Compute kernels (e.g. `kmedoids_core.hpp`) exposed to Python via `pybind11`.                    |
| `tests/`               | Python   | Smoke + integration tests covering both the pure-Python path and the C++ binding.               |

Python orchestration always provides a **pure-Python fallback** so that workflows
remain runnable when the compiled extension is unavailable. The C++ engine, when
built, is loaded transparently — no API change at the Python call site.

---

## Layout (CyHAN v1.1)

```
PyStorm/
│
├── modules/
│   └── storm_selection/                Representative TC Subset selection
│       ├── README.md                   Module reference (methods, data flow, API)
│       ├── backend/
│       │   ├── engines/cpp/            C++ k-medoids engine (pybind11)
│       │   │   ├── kmedoids_core.hpp   Header-only PAM with FastPAM1 refinement
│       │   │   ├── bindings.cpp        Python binding
│       │   │   ├── CMakeLists.txt      CMake build (alt: build.py)
│       │   │   └── build.py            Standalone build helper
│       │   └── python/storm_selection/ Orchestration package
│       ├── scripts/                    Launchers (Python)
│       │   ├── preprocess.py           Raw inputs → tc_data.h5
│       │   ├── run_storm_selection.py  RTCS selection (fixed | optimal)
│       │   └── dsw.py                  Post-selection DSW + HC reconstruction
│       ├── tests/                      Smoke + round-trip tests
│       └── data/                       Raw inputs & processed outputs (gitignored)
│
├── docs/
│   ├── CyHAN-Standard-v1.1.md          Architecture standard
│   └── PyStorm-Comment-Standard-v0.2.md   Comment & docstring conventions
│
└── archive/                            Pre-refactor snapshot
```

Root-level `backend/`, `frontend/`, and shared `api/` directories are anticipated
future work (CyHAN v1.1 §16.8) and intentionally absent in this revision.

---

## Quickstart — storm_selection

```bash
cd modules/storm_selection

# Build the C++ k-medoids engine (pure-Python fallback is used if you skip this)
python backend/engines/cpp/build.py

# Ingest raw inputs → tc_data.h5
python scripts/preprocess.py

# Run the RTCS selection (fixed-k default; --mode optimal for growth loop)
python scripts/run_storm_selection.py

# (Optional) Post-selection DSW + hazard-curve reconstruction
python scripts/dsw.py
```

See [`modules/storm_selection/README.md`](modules/storm_selection/README.md) for
module-specific details.

---

## Building the C++ engines

The compiled extensions are not required for correctness, but they accelerate
the inner loops by ~1–2 orders of magnitude on large problems. Build per
module:

```bash
python modules/storm_selection/backend/engines/cpp/build.py
```

`build.py` tries `setuptools`, then CMake, then a direct compiler invocation.
Requires `pybind11` (`pip install pybind11`) and a C++17 toolchain
(Microsoft Visual C++ — MSVC — on Windows, gcc/clang elsewhere).

---

## Acronyms

| Acronym | Expansion                                              |
|---------|--------------------------------------------------------|
| API     | Application Programming Interface                      |
| CLI     | Command-Line Interface                                 |
| CyHAN   | C++/Python Hybrid Architecture Network                 |
| DSW     | Discrete Storm Weight                                  |
| HC      | Hazard Curve                                           |
| MSVC    | Microsoft Visual C++ (compiler)                        |
| PAM     | Partitioning Around Medoids (k-medoids algorithm)      |
| RTCS    | Reduced Tropical Cyclone Suite                         |
| TC      | Tropical Cyclone                                       |
