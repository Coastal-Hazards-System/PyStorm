# PyStorm

*A modular Python and C++ framework for probabilistic modeling of coastal storm hazards.*

---

## Overview

PyStorm supports coastal hazard quantification, stochastic storm simulation, and
extreme-value analysis. Python handles orchestration, configuration, I/O, and
plotting; C++ engines handle the inner numerical kernels that dominate runtime.
Workflows stay script-driven and reproducible while the compiled kernels carry
the heavy compute.

Every compiled engine has a pure-Python fallback, so each workflow runs whether
or not the extension is built. When a kernel is built it loads transparently,
with no change at the Python call site.

## Modules

PyStorm has three modules. POT and PST chain together; RSS is independent.

| Module | Purpose |
|--------|---------|
| [`peaks_over_threshold`](modules/peaks_over_threshold/README.md) (POT) | Extract independent storm peaks from a continuous water-level or NTR time series. |
| [`probabilistic_simulation_technique`](modules/probabilistic_simulation_technique/README.md) (PST) | Turn a POT peak sample into a hazard curve (response magnitude versus AER) with a confidence band. |
| [`reduced_storm_suite`](modules/reduced_storm_suite/README.md) (RSS) | Select a small, representative Reduced Tropical Cyclone Suite that reproduces the full synthetic suite's hazard. |

POT writes per-station peak files that PST reads directly.

## Architecture

Every module is a self-contained vertical with the same internal shape:

| Path | Language | Role |
|------|----------|------|
| `run_<module>.py` | Python | Launcher at the module root. User options only; calls the orchestrator. |
| `backend/python/` | Python | Orchestration: data flow, I/O, configuration, diagnostics, plotting. |
| `backend/engines/` | C++ | Compute kernels exposed to Python through pybind11. |
| `scripts/` | Python | Ancillary tools (preprocessing, post-processing, testbeds). |
| `tests/` | Python | Smoke and integration tests covering the Python path and the C++ binding. |

## Quickstart

Run a module from its directory. The C++ kernel builds automatically on the
first run; if no compiler is available the pure-Python fallback runs instead.

```bash
# POT: extract peaks from a water-level / NTR series
cd modules/peaks_over_threshold
python run_peaks_over_threshold.py

# PST: hazard curves from the POT peaks
cd modules/probabilistic_simulation_technique
python run_probabilistic_simulation_technique.py

# RSS: reduced storm suite selection
cd modules/reduced_storm_suite
python run_reduced_storm_suite.py
```

Each launcher has a USER OPTIONS block at the top and `--help` for command-line
overrides. Per-module data lives under `modules/<module>/data/` (`inputs/` and
`outputs/`, both gitignored).

## Building the C++ engines

The compiled kernels are not required for correctness; they accelerate the inner
loops by roughly one to two orders of magnitude on large problems. Each launcher
builds its kernel on first run, or build it manually:

```bash
python modules/peaks_over_threshold/backend/engines/cpp/build.py            # _pot
python modules/probabilistic_simulation_technique/backend/engines/build.py  # _pst
python modules/reduced_storm_suite/backend/engines/cpp/build.py             # _rss
```

`build.py` tries setuptools, then CMake, then a direct compiler call. It needs
pybind11 (`pip install pybind11`) and a C++17 toolchain (MSVC or MinGW on
Windows, gcc or clang elsewhere).

## Repository layout

```
PyStorm/
│
├── modules/
│   ├── peaks_over_threshold/                      POT event extraction
│   │   ├── run_peaks_over_threshold.py            Launcher at module root
│   │   ├── README.md                              Module reference (methods, workflow, outputs)
│   │   ├── pyproject.toml                         Installable orchestrator package
│   │   ├── ENGINE_MANIFEST.toml                   Structured module manifest
│   │   ├── backend/
│   │   │   ├── engines/cpp/                       C++ iterative threshold-search kernel
│   │   │   │   ├── POTThresholdSearch.hpp         Header-only kernel
│   │   │   │   ├── pot_bindings.cpp               pybind11 → _pot
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── build.py
│   │   │   └── python/
│   │   │       ├── main_peaks_over_threshold.py   Orchestrator entry
│   │   │       └── peaks_over_threshold/          Expanded orchestration package
│   │   ├── tests/                                 Smoke + preprocessing tests
│   │   └── data/                                  inputs/{raw,processed}/ & outputs/ (gitignored)
│   │
│   ├── probabilistic_simulation_technique/        PST hazard curves with bootstrap band
│   │   ├── run_probabilistic_simulation_technique.py   Launcher at module root
│   │   ├── README.md                             Module reference (methods, workflow, outputs)
│   │   ├── pyproject.toml                         Installable orchestrator package
│   │   ├── ENGINE_MANIFEST.toml                   Structured module manifest
│   │   ├── backend/
│   │   │   ├── engines/                           C++ smoothed-bootstrap kernel
│   │   │   │   ├── PSTBootstrap.hpp               Header-only kernel
│   │   │   │   ├── pst_bindings.cpp               pybind11 → _pst
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── build.py
│   │   │   └── python/
│   │   │       ├── main_probabilistic_simulation_technique.py   Orchestrator entry
│   │   │       └── probabilistic_simulation_technique/          Expanded orchestration package
│   │   ├── scripts/                               Ancillary tools
│   │   │   └── method_testbed.py                  Four-method selection comparison
│   │   ├── tests/                                 Smoke tests
│   │   └── data/                                  inputs/{raw,processed}/ & outputs/ (gitignored)
│   │
│   └── reduced_storm_suite/                       RTCS selection
│       ├── run_reduced_storm_suite.py            Launcher at module root
│       ├── README.md                             Module reference (methods, data flow, API)
│       ├── pyproject.toml                         Installable orchestrator package
│       ├── ENGINE_MANIFEST.toml                   Structured module manifest
│       ├── backend/
│       │   ├── engines/cpp/                       C++ k-medoids engine (pybind11 → _rss)
│       │   │   ├── kmedoids_core.hpp              Header-only PAM with FastPAM1 refinement
│       │   │   ├── bindings.cpp                   Python binding
│       │   │   ├── CMakeLists.txt
│       │   │   └── build.py
│       │   └── python/
│       │       ├── main_reduced_storm_suite.py   Orchestrator entry
│       │       └── reduced_storm_suite/          Expanded orchestration package
│       ├── scripts/                               Ancillary tools
│       │   ├── preprocess.py                      Raw inputs → tc_data.h5
│       │   └── dsw.py                             Post-selection DSW + HC reconstruction
│       ├── tests/                                 Smoke + round-trip tests
│       └── data/                                  inputs/{raw,processed}/ & outputs/ (gitignored)
│
├── docs/
│   ├── CyHAN-Standard-v2.0.md                     Architecture standard
│   └── PyStorm-Comment-Standard-v0.2.md           Comment and docstring conventions
│
└── archive/                                       Pre-refactor snapshot
```

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| AER | Annual Exceedance Rate |
| API | Application Programming Interface |
| CLI | Command-Line Interface |
| CyHAN | C++/Python Hybrid Architecture Network |
| DSW | Discrete Storm Weight |
| GPD | Generalized Pareto Distribution |
| HC | Hazard Curve |
| MRI | Mean Return Interval (MRI = 1 / AER) |
| MSVC | Microsoft Visual C++ (compiler) |
| NTR | Non-Tidal Residual |
| PAM | Partitioning Around Medoids (k-medoids) |
| POT | Peaks Over Threshold |
| PST | Probabilistic Simulation Technique |
| RTCS | Reduced Tropical Cyclone Suite |
| TC | Tropical Cyclone |

## Standards

The architecture follows CyHAN Standard v2.0: each capability is a self-contained
module with its own C++ engine and Python orchestration. See
[docs/CyHAN-Standard-v2.0.md](docs/CyHAN-Standard-v2.0.md). Source comments and
docstrings follow the
[Comment and Docstring Standard](docs/PyStorm-Comment-Standard-v0.2.md).
