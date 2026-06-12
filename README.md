# PyStorm

*A modular Python and C++ framework for probabilistic modeling of coastal storm hazards.*

---

## Overview

The PyStorm framework supports coastal hazards quantification, probabilistic
modeling, stochastic storm simulation, and extreme-value analysis. Python handles
orchestration, configuration, I/O, and plotting; C++ engines carry the inner
numerical kernels that dominate runtime. Workflows stay script-driven and
reproducible.

Following CyHAN v2.1, a C++ engine is shipped where a module has
performance-critical computation, and is omitted where it does not. Compute-heavy
modules (POT, PST, RTCS) ship a C++ kernel with a pure-Python fallback so each
workflow runs whether or not the extension is built; pure-Python modules (TCA,
SSH) ship no engine. AHD ships an optional C++ accelerator with a NumPy fallback.

## Modules

PyStorm has six modules. Two short chains, two independent:

| Module | Purpose |
|--------|---------|
| [`augmented_hurricane_database`](modules/augmented_hurricane_database/README.md) (AHD) | Build an augmented HURDAT2 best-track: parse NHC HURDAT2, derive storm motion, optionally backfill Rmax from EBTRK and impute missing Cp/Rmax with a Gaussian-process metamodel. The best-track foundation for downstream TC analyses. |
| [`tc_climatological_analysis`](modules/tc_climatological_analysis/README.md) | Per-CRL tropical-cyclone storm recurrence rates (SRR/DSRR) from the augmented HURDAT2, annual and monthly, via the Gaussian Kernel Function. |
| [`peaks_over_threshold`](modules/peaks_over_threshold/README.md) (POT) | Extract independent storm peaks from a continuous water-level or NTR time series. |
| [`probabilistic_simulation_technique`](modules/probabilistic_simulation_technique/README.md) (PST) | Turn a POT peak sample into a hazard curve (response magnitude versus AER) with a confidence band. |
| [`reduced_tc_suite`](modules/reduced_tc_suite/README.md) (RTCS) | Select a small, representative Reduced Tropical Cyclone Suite that reproduces the full synthetic suite's hazard. |
| [`storm_surge_hydrograph`](modules/storm_surge_hydrograph/README.md) (SSH) | Reduce an ensemble of synthetic-TC surge series to one dimensionless surge shape per save point, scalable by a peak elevation and an equivalent width. |

Data flow: **AHD writes the augmented best-track that TCA reads; POT writes
per-station peak files that PST reads.** RTCS and SSH are independent.

## Architecture

Every module is a self-contained vertical (CyHAN v2.1 §5) with the same core
shape: a user-facing launcher at the module root that calls a non-user-facing
orchestrator under `backend/python/`.

| Path | Language | Role |
|------|----------|------|
| `run_<module>.py` | Python | Launcher at the module root. User options only; calls the orchestrator. |
| `backend/python/main_<module>.py` | Python | Orchestrator entry: the Python Orchestration role (data flow, I/O, config, diagnostics, plotting). |
| `backend/engines/` | C++ | Compute kernel exposed through pybind11 (only where a module has heavy compute). |
| `tests/` | Python | Smoke and integration tests covering the Python path and, where present, the C++ binding. |

What each module ships varies with its workload:

| Module | C++ engine | Ancillary scripts |
|--------|------------|-------------------|
| AHD | optional `_gpm` accelerator (NumPy fallback) | — |
| TCA | none (pure Python) | `analysis/` (kernel sensitivity) |
| POT | `_pot` kernel | — |
| PST | `_pst` kernel | `scripts/` (method testbed) |
| RTCS | `_rtcs` kernel | `scripts/` (preprocess, DSW) |
| SSH | none (pure Python) | `analysis/` (shape/timescale studies) |

## Quickstart

Run a module from its directory. Where a module has a C++ kernel it builds
automatically on the first run; if no compiler is available the pure-Python
fallback runs instead.

```bash
# AHD: build the augmented HURDAT2 best-track
cd modules/augmented_hurricane_database
python run_augmented_hurricane_database.py

# TCA: per-CRL storm recurrence rates from the augmented best-track
cd modules/tc_climatological_analysis
python run_tc_climatological_analysis.py

# POT: extract peaks from a water-level / NTR series
cd modules/peaks_over_threshold
python run_peaks_over_threshold.py

# PST: hazard curves from the POT peaks
cd modules/probabilistic_simulation_technique
python run_probabilistic_simulation_technique.py

# RTCS: reduced TC suite selection
cd modules/reduced_tc_suite
python run_reduced_tc_suite.py

# SSH: unit storm-surge hydrographs
cd modules/storm_surge_hydrograph
python run_storm_surge_hydrograph.py
```

Each launcher has a USER OPTIONS block at the top and `--help` for command-line
overrides. Per-module data lives under `modules/<module>/data/` (`inputs/` and
`outputs/`, both gitignored).

## Building the C++ engines

The compiled kernels are not required for correctness; they accelerate the inner
loops by roughly one to two orders of magnitude on large problems. Each kernel
builds on first run, or build it manually:

```bash
python modules/peaks_over_threshold/backend/engines/cpp/build.py             # _pot
python modules/probabilistic_simulation_technique/backend/engines/build.py   # _pst
python modules/reduced_tc_suite/backend/engines/cpp/build.py                 # _rtcs
python modules/augmented_hurricane_database/backend/engines/cpp/build.py     # _gpm (optional)
```

`build.py` tries setuptools, then CMake, then a direct compiler call. It needs
pybind11 (`pip install pybind11`) and a C++17 toolchain (MSVC or MinGW on
Windows, gcc or clang elsewhere). TCA and SSH have no engine to build.

## Repository layout

```
PyStorm/
│
├── modules/                                  six self-contained capability verticals
│   ├── augmented_hurricane_database/         AHD  augmented HURDAT2 best-track
│   ├── tc_climatological_analysis/           per-CRL SRR/DSRR (consumes AHD)
│   ├── peaks_over_threshold/                 POT  storm-peak extraction
│   ├── probabilistic_simulation_technique/   PST  hazard curves (consumes POT)
│   ├── reduced_tc_suite/                     RTCS representative suite selection
│   └── storm_surge_hydrograph/               SSH  unit storm-surge hydrographs
│
│   Each module:
│     run_<module>.py            launcher (user options)
│     README.md                  module reference (methods, workflow, outputs)
│     ENGINE_MANIFEST.toml       structured module manifest
│     pyproject.toml             installable orchestrator package
│     backend/
│       engines/                 C++ kernel + pybind11 binding (compute-heavy modules)
│       python/
│         main_<module>.py       orchestrator entry
│         <module>/              expanded orchestration package
│     tests/                     smoke + integration tests
│     data/                      inputs/{raw,processed}/ & outputs/ (gitignored)
│
├── docs/
│   ├── CyHAN-Standard-v2.1.md                architecture standard
│   └── CyHAN-Comment-Standard-v0.3.md        comment and docstring conventions
│
├── backend/   (planned, CyHAN §6.1)          shared API surface above the modules
├── common/    (CyHAN §5.2 / §16.10)          shared library: pystorm_common
│                                             (palette, style_ax, save_figure) - used by all modules
└── archive/                                  pre-refactor snapshot
```

The root-level integration tier (`backend/api/`, `frontend/`) and the shared
`common/` library are permitted by CyHAN v2.1 but are not required for any module
to build or run; each module remains independently operable through its launcher.
See [Shared common library](#shared-common-library).

## Shared common library

`common/python/pystorm_common/` is the shared library that holds cross-module
presentation helpers, so there is one source of truth instead of per-module
copies. All six modules write their figures through it.

- **Contents:** the Wave Maker design palette, the `style_ax` axes-styling helper,
  and the `save_figure` writer (which fixes the PyStorm figure DPI standard at
  `DEFAULT_DPI = 150` and creates parent dirs). The palette and `style_ax` were
  previously duplicated in POT and PST (the palette byte-for-byte, `_style_ax`
  already drifting); every module's figure write now goes through `save_figure` at
  the 150 DPI standard. AHD and SSH still pass their fast PNG settings
  (`compress_level`, no tight bbox) through `save_figure`. The one exception is
  TCA's per-CRL map renderer (a blit/PIL fast-path for ~1000+ maps that does not
  use `savefig`), which stays at 110 dpi by design.
- **Scope (presentation and pure utilities only):** it **must not** hold module
  domain logic, numerical kernels, or orchestration - that would couple modules
  through their compute, not just their look.
- **How modules find it:** each launcher adds `common/python/` to `sys.path`
  alongside `backend/python/` (tests do the same via `conftest.py`); it is also
  pip-installable (`pip install -e common/`). CyHAN §5.2 permits a module to depend
  on a shared common library and §16.10 lists it as an optional extension.
- **Self-containment caveat:** importing from `common/` makes it part of a module's
  dependency set, so vendoring a module standalone means vendoring `common/` too.
  Modules still run in isolation through their launcher; `common/` is an
  integration-tier dependency, not a sibling-module dependency (CyHAN §5.2 forbids
  the latter).

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| AER | Annual Exceedance Rate |
| AHD | Augmented Hurricane Database |
| API | Application Programming Interface |
| CLI | Command-Line Interface |
| CRL | Coastal Reference Location |
| CyHAN | C++/Python Hybrid Architecture Network |
| DSRR | Directional Storm Recurrence Rate |
| DSW | Discrete Storm Weight |
| EBTRK | Extended Best Track |
| GKF | Gaussian Kernel Function |
| GP | Gaussian Process |
| GPD | Generalized Pareto Distribution |
| HC | Hazard Curve |
| HURDAT2 | HURricane DATabase, 2nd generation (NHC best-track) |
| MRI | Mean Return Interval (MRI = 1 / AER) |
| MSVC | Microsoft Visual C++ (compiler) |
| NTR | Non-Tidal Residual |
| PAM | Partitioning Around Medoids (k-medoids) |
| POT | Peaks Over Threshold |
| PST | Probabilistic Simulation Technique |
| RTCS | Reduced Tropical Cyclone Suite |
| SRR | Storm Recurrence Rate |
| SSH | Storm Surge Hydrograph |
| TC | Tropical Cyclone |

## Standards

The architecture follows CyHAN Standard v2.1: each capability is a self-contained
module with its own Python orchestration and, where it has performance-critical
computation, a C++ engine. See
[docs/CyHAN-Standard-v2.1.md](docs/CyHAN-Standard-v2.1.md). Source comments and
docstrings follow the
[Comment and Docstring Standard](docs/CyHAN-Comment-Standard-v0.3.md).
