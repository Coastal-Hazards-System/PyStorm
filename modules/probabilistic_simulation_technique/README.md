# probabilistic_simulation_technique

**Probabilistic Simulation Technique (PST) for Coastal Hazard Curves**

PyStorm, Module: PST (CyHAN v2.0 §5)

---

## Introduction

The **Probabilistic Simulation Technique (PST)** quantifies aleatory and
epistemic uncertainty in extreme-value hazard curves derived from a
Peaks-Over-Threshold (POT) sample. Given a univariate record of peaks (e.g.
storm-surge maxima), PST:

1. Selects the Generalized Pareto Distribution (GPD) location μ by minimizing a
   Quantile Delta Optimization (QDO) weighted-mean-square error (WMSE) over a
   candidate band, and emits a diagnostics plot to visually assess the choice.
2. Bootstraps the descending-sorted exceedances using truncated Gaussian (or
   Uniform) perturbations.
3. Fits a GPD to each realization and evaluates its inverse-CDF on a dense
   plotting grid of Annual Exceedance Rates (AERs).
4. Splices the GPD upper tail onto the empirical-Weibull lower tail.
5. Interpolates the merged curve onto a standard 22-AER reporting grid and
   writes both the ensemble and the hazard-curve tables.

The inner Monte Carlo loop — the truncated-noise bootstrap matrix — is
implemented in C++ (`backend/engines/PSTBootstrap.hpp`) and exposed through
the `_pst` pybind11 extension. A pure-Python fallback (`scipy.stats`) is used
when the extension is not built.

---

## 1. Module Layout (CyHAN v2.0 §16.1)

```
probabilistic_simulation_technique/
├── run_probabilistic_simulation_technique.py   ← launcher (user-facing, §5.3)
├── README.md
├── pyproject.toml
├── requirements.txt
├── ENGINE_MANIFEST.toml
├── backend/
│   ├── engines/
│   │   ├── PSTBootstrap.hpp                    header-only truncated-noise bootstrap
│   │   ├── pst_bindings.cpp                    pybind11 → _pst
│   │   ├── CMakeLists.txt
│   │   ├── build.py                            standalone build helper
│   │   └── README.md
│   └── python/
│       ├── main_probabilistic_simulation_technique.py   ← orchestrator entry (§5.3)
│       └── probabilistic_simulation_technique/          ← expanded package (§5.3)
│           ├── __init__.py
│           ├── config.py            pydantic PSTConfig / BootstrapConfig
│           ├── orchestrator.py      PSTOrchestrator workflow runner
│           ├── solver.py            thin _pst binding wrapper
│           ├── sampling/
│           │   ├── bootstrap.py     BootstrapGenerator (C++ or fallback)
│           │   └── gpd_threshold.py QDO-WMSE threshold search
│           ├── hazard/
│           │   └── curve.py         ensemble fit + tail splice + table interp
│           ├── postproc/
│           │   └── plots.py         HazardCurvePlotter
│           └── io/
│               └── pot_csv.py       POT reader + result writers
├── tests/
│   └── test_smoke.py
├── data/                                       § 16.7
│   ├── inputs/
│   │   ├── raw/                                unmodified source inputs
│   │   └── processed/                          POT CSVs (operator default target)
│   └── outputs/                                PST ensembles, HC tables, plots/
├── research/                                   ad-hoc validation probes
└── docs/                                       extended notes
```

The two **mandatory entry artifacts** per CyHAN v2.0 §5.3:

| Artifact     | Location                                                                              | Role               |
|--------------|---------------------------------------------------------------------------------------|--------------------|
| Launcher     | `run_probabilistic_simulation_technique.py`                                            | user-facing entry  |
| Orchestrator | `backend/python/main_probabilistic_simulation_technique.py`                            | non-user-facing    |

The launcher imports `run` from the orchestrator entry; orchestration logic
lives in the expanded `backend/python/probabilistic_simulation_technique/`
package per §5.3 ("Begin as a single file and expand into a
`backend/python/<name>/` package as complexity warrants, preserving its import
entry point").

---

## 2. Methods

### 2.1 GPD Location μ Selection (QDO-WMSE)

The POT sample is already extracted above the POT threshold `u`, with rate
`λ_u = λ = n / record_length_years`. PST then re-optimizes a **separate** GPD
location parameter `μ ≥ u` for the distribution fit — `u` defines the sample
(and `λ_u`); `μ` defines where the GPD tail begins.

Let `values_pot` be the sample sorted descending and
`weibull_aer[i] = (i + 1) / (n + 1) · λ_u` the empirical Weibull
plotting-position AERs. For each candidate location `μ` in the
percentile band `[μ_min, μ_max]` (default `20–80%` of the value range):

1. Take the exceedances `pot > μ` and the associated `aer`.
2. Fit a GPD with `floc = μ` to the exceedances.
3. Predict at the empirical positions and compute
   `WMSE = Σ wᵢ (potᵢ − predᵢ)² / Σ wᵢ`, with `wᵢ = 1/aerᵢ` for `aerᵢ < 1`.

The lowest-μ candidate whose WMSE is within 5% (relative) of the in-band
minimum WMSE is selected — preferring data-rich (lower-μ) fits only when the
WMSE is genuinely near-minimal. Anchoring on the minimum (rather than a range
normalized by the degenerate high-μ spikes) keeps the tolerance tight.
Implementation: `sampling/gpd_threshold.py`.

### 2.2 Truncated-Noise Bootstrap

Given the descending-sorted exceedances `pot[0..n_pot-1]` above the threshold
and the descending spacing `delta[i] = pot[i+1] - pot[i]` (with
`delta[last] = 0`), each of `n_sims` realizations is constructed by:

1. Draw `n_pot` indices `idx ~ U{0, …, n_pot-1}`.
2. Draw `n_pot` truncated noise variates `z` from the configured distribution
   (Gaussian via rejection on `N(0,1)` or Uniform on `[lo, hi]`).
3. Compute `perturbed[i] = pot[idx[i]] + delta[idx[i]] · z[i]`.
4. Sort the column descending.

The C++ kernel in `PSTBootstrap.hpp` is the default backend (preferred for
`num_simulations >> 10²`). The pure-Python implementation in
`sampling/bootstrap.py` is algorithmically identical but slower; it activates
automatically when `_pst` is unavailable.

### 2.3 GPD Ensemble Fit and Hazard-Curve Assembly

For each bootstrap column the GPD is refit (`floc = μ`) and its shape `c` is
clipped to the Luceño-style band `[c_lo, c_hi]` (defaults `[-0.5, +0.33]`)
before the ICDF is evaluated on the plot AER grid restricted to
`aer < λ_μ = (# exceedances) / record_length_years`. The realization stack is
collapsed to a best-estimate mean and the 10/90% percentile bounds.

The empirical bulk (`pot ≤ μ`) at its Weibull AERs is concatenated below the
GPD tail; bulk uncertainty is taken as zero per the v1 convention. The merged
curve is then log-interpolated onto the 22-AER reporting grid
(`make_aer_grids()` in `hazard/curve.py`).

---

## 3. Workflow

**Input selection (launcher `INPUT_MODE`, resolved in `main`):**

- `"path"` — one POT CSV at an explicit `INPUT_CSV`.
- `"station"` — batch over the **peaks_over_threshold** module's outputs for
  `STATION_IDS` (one or many), chosen by `PST_TARGETS` (`"dwl"`, `"ntr"`, or
  `"both"`), resolving to `peaks_over_threshold/data/outputs/<station>/<target>_<station>_pot.csv`.
  PST runs once per (station × target); e.g. 3 stations × `"both"` = 6 output sets.

```
┌──────────────────────────────────────────────────────────────────────┐
│  POT CSV  (path mode: INPUT_CSV │ station mode: <target>_<station>_pot)│
│           column = STORM_COLUMN ("value")                             │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  [1]  read_pot_csv  →  values, λ = n / record_length_years            │
├──────────────────────────────────────────────────────────────────────┤
│  [2]  Sort descending; Weibull AERs                                   │
├──────────────────────────────────────────────────────────────────────┤
│  [3]  select_gpd_threshold_qdo  →  μ                                  │
├──────────────────────────────────────────────────────────────────────┤
│  [4]  Split exceedances / bulk; λ_μ = |exceed| / record_length_years  │
├──────────────────────────────────────────────────────────────────────┤
│  [5]  BootstrapGenerator (_pst or Python fallback)                    │
│       → boot_matrix [n_pot × n_sims]                                  │
├──────────────────────────────────────────────────────────────────────┤
│  [6]  fit_gpd_ensemble → BE, CB10, CB90 on plot AER grid              │
├──────────────────────────────────────────────────────────────────────┤
│  [7]  assemble_hazard_curve + interpolate_to_table                    │
├──────────────────────────────────────────────────────────────────────┤
│  [8]  write_pst_outputs + HazardCurvePlotter                          │
│       → data/outputs/<base>_pst*.csv                                  │
│       → data/outputs/plots/<base>_pst_hc.png                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Outputs

| File                       | Contents                                                          |
|----------------------------|-------------------------------------------------------------------|
| `<base>_pst.csv`           | Bootstrap GPD ensemble, plot AER grid columns                     |
| `<base>_pst_hc_be_tbl.csv` | Best estimate on the 22-AER reporting grid                        |
| `<base>_pst_hc_cb_tbl.csv` | 10/90% confidence bounds on the 22-AER grid                       |
| `<base>_pst_hc_be_plt.csv` | BE on the dense plotting grid (merged GPD + empirical)            |
| `<base>_pst_hc_cb_plt.csv` | CB10/CB90 on the dense plotting grid                              |
| `<base>_pst_hc.png`        | Hazard-curve plot (empirical < μ / > μ, GPD mean + CL band, μ cross) |
| `<base>_qdo_threshold.png` | QDO μ-selection diagnostics: WMSE(μ), GPD shape ξ(μ), # exceedances(μ) |

---

## 5. Quickstart

```bash
cd modules/probabilistic_simulation_technique

# (Optional) Build the C++ kernel. Pure-Python fallback works without this.
python backend/engines/build.py

# Make the package importable
pip install -e .

# Edit USER OPTIONS in run_probabilistic_simulation_technique.py, then run:
python run_probabilistic_simulation_technique.py
```

All settings live in the launcher's USER OPTIONS block. For the default
`INPUT_MODE = "station"`, set `STATION_IDS` (one or many) and `PST_TARGETS`
("dwl" / "ntr" / "both"); for `INPUT_MODE = "path"`, set `INPUT_CSV`.

Smoke tests:

```bash
pytest tests/
```

---

## 6. CyHAN v2.0 Compliance

| Requirement                                                       | Status                                                         |
|-------------------------------------------------------------------|----------------------------------------------------------------|
| §1   API → Orchestrator → Engine; one-way dependency              | ✓ engine is header-only; orchestrator owns side effects        |
| §4.1 Binding is a conduit, not authority                          | ✓ `_pst` exposes one function; orchestration lives in Python   |
| §4.2 Orchestration in Python, non-user-facing                     | ✓ `main_<name>.py` + expanded package                          |
| §5.1 Module ships engine + orchestrator + launcher                | ✓                                                              |
| §5.2 Self-contained; no sibling-module imports                    | ✓                                                              |
| §5.3 Launcher `run_<name>.py` at module root, user-facing         | ✓                                                              |
| §5.3 Orchestrator `main_<name>.py` at `backend/python/`           | ✓                                                              |
| §5.3 Launcher contains no orchestration logic                     | ✓ delegates to `main_<name>.run`                               |
| §5.4 `snake_case` module identifier end-to-end                    | ✓                                                              |
| §16.1 / §16.2 Recommended folder layout + layer mapping           | ✓                                                              |
| §16.7 Data convention (`inputs/raw/`, `inputs/processed/`, `outputs/`) | ✓                                                          |

---

## 7. Acronyms

| Acronym  | Expansion                                                |
|----------|----------------------------------------------------------|
| AER      | Annual Exceedance Rate                              |
| BE       | Best Estimate                                            |
| CB       | Confidence Bound                                         |
| CDF      | Cumulative Distribution Function                         |
| CyHAN    | C++/Python Hybrid Architecture Network                   |
| GPD      | Generalized Pareto Distribution                          |
| HC       | Hazard Curve                                             |
| ICDF     | Inverse CDF (quantile function)                          |
| POC      | Point Of Contact                                         |
| POT      | Peaks Over Threshold                                     |
| PST      | Probabilistic Simulation Technique                       |
| QDO      | Quantile Delta Optimization                                    |
| RNG      | Random Number Generator                                  |
| WMSE     | Weighted Mean Square Error                               |
| WPP      | Weibull Plotting Position                                |
