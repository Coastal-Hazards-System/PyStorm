# PyStorm

### *A Modular Python/C++ Framework for Probabilistic Modeling of Coastal Storm Hazards*

---

## Overview

**PyStorm** is an open-source framework for probabilistic modeling and analysis,
designed for engineers, scientists, and researchers working on coastal hazard
quantification, stochastic storm simulation, life-cycle analysis, and extreme
value analysis. It pairs **Python** for orchestration, configuration, I/O, and
post-processing with **C++** engines for the inner numerical kernels that
dominate runtime — keeping workflows script-driven and reproducible while
leaving the heavy lifting to compiled code.

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
│   ├── reduced_storm_suite/                       RTCS selection (CyHAN v2.0)
│   │   ├── run_reduced_storm_suite.py             Launcher at module root (§5.3)
│   │   ├── README.md                              Module reference (methods, data flow, API)
│   │   ├── pyproject.toml                         pip-installable orchestrator package
│   │   ├── ENGINE_MANIFEST.toml                   Structured module manifest
│   │   ├── backend/
│   │   │   ├── engines/cpp/                       C++ k-medoids engine (pybind11 → _rss)
│   │   │   │   ├── kmedoids_core.hpp              Header-only PAM with FastPAM1 refinement
│   │   │   │   ├── bindings.cpp                   Python binding
│   │   │   │   ├── CMakeLists.txt                 CMake build (alt: build.py)
│   │   │   │   └── build.py                       Standalone build helper
│   │   │   └── python/
│   │   │       ├── main_reduced_storm_suite.py    Orchestrator entry (§5.3)
│   │   │       └── reduced_storm_suite/           Expanded orchestration package
│   │   ├── scripts/                               Ancillary tools (§16.10)
│   │   │   ├── preprocess.py                      Raw inputs → tc_data.h5
│   │   │   └── dsw.py                             Post-selection DSW + HC reconstruction
│   │   ├── tests/                                 Smoke + round-trip tests
│   │   └── data/                                  inputs/{raw,processed}/ & outputs/ (gitignored)
│   │
│   ├── probabilistic_simulation_technique/   PST hazard curves w/ bootstrap (CyHAN v2.0)
│   │   ├── run_probabilistic_simulation_technique.py   Launcher at module root (§5.3)
│   │   ├── README.md                   Module reference (methods, workflow, outputs)
│   │   ├── pyproject.toml              pip-installable orchestrator package
│   │   ├── ENGINE_MANIFEST.toml        Structured module manifest
│   │   ├── backend/
│   │   │   ├── engines/                C++ truncated-noise bootstrap kernel
│   │   │   │   ├── PSTBootstrap.hpp    Header-only kernel
│   │   │   │   ├── pst_bindings.cpp    pybind11 → _pst
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── build.py
│   │   │   └── python/
│   │   │       ├── main_probabilistic_simulation_technique.py    Orchestrator entry (§5.3)
│   │   │       └── probabilistic_simulation_technique/           Expanded orchestration package
│   │   ├── tests/                      Smoke tests
│   │   └── data/                       inputs/{raw,processed}/ & outputs/ (gitignored)
│   │
│   └── peaks_over_threshold/                 POT event extraction (CyHAN v2.0)
│       ├── run_peaks_over_threshold.py       Launcher at module root (§5.3)
│       ├── README.md                         Module reference (methods, workflow, outputs)
│       ├── pyproject.toml                    pip-installable orchestrator package
│       ├── ENGINE_MANIFEST.toml              Structured module manifest
│       ├── backend/
│       │   ├── engines/cpp/                  C++ iterative threshold-search kernel
│       │   │   ├── POTThresholdSearch.hpp    Header-only kernel
│       │   │   ├── pot_bindings.cpp          pybind11 → _pot
│       │   │   ├── CMakeLists.txt
│       │   │   └── build.py
│       │   └── python/
│       │       ├── main_peaks_over_threshold.py    Orchestrator entry (§5.3)
│       │       └── peaks_over_threshold/           Expanded orchestration package
│       ├── tests/                            Smoke tests
│       └── data/                             inputs/{raw,processed}/ & outputs/ (gitignored)
│
├── docs/
│   ├── CyHAN-Standard-v2.0.md             Architecture standard
│   └── PyStorm-Comment-Standard-v0.2.md   Comment & docstring conventions
│
└── archive/                               Pre-refactor snapshot
```

Root-level `backend/api/` and `frontend/` integration tiers are permitted by
CyHAN v2.0 (§6, §16.9) but are not yet present in this repo. All three
modules — `reduced_storm_suite`, `probabilistic_simulation_technique`, and
`peaks_over_threshold` — are CyHAN v2.0-compliant.

---

## Quickstart — reduced_storm_suite

```bash
cd modules/reduced_storm_suite

# Build the C++ k-medoids engine (pure-Python fallback is used if you skip this)
python backend/engines/cpp/build.py

# Ingest raw inputs → tc_data.h5
python scripts/preprocess.py

# Run the RTCS selection (fixed-k default; --mode optimal for growth loop)
python scripts/run_reduced_storm_suite.py

# (Optional) Post-selection DSW + hazard-curve reconstruction
python scripts/dsw.py
```

See [`modules/reduced_storm_suite/README.md`](modules/reduced_storm_suite/README.md) for
module-specific details.

---

## Building the C++ engines

The compiled extensions are not required for correctness, but they accelerate
the inner loops by ~1–2 orders of magnitude on large problems. Build per
module:

```bash
python modules/reduced_storm_suite/backend/engines/cpp/build.py
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

---

The architecture follows **CyHAN Standard v1.1** — a module-first decomposition
in which each capability is a self-contained vertical with its own C++ engine
and Python orchestration. See
[`docs/CyHAN-Standard-v1.1.md`](docs/CyHAN-Standard-v1.1.md). Source comments
and docstrings (Python and C++) follow the project-wide
[Comment & Docstring Standard](docs/PyStorm-Comment-Standard-v0.2.md).
