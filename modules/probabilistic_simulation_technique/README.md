# probabilistic_simulation_technique

**Probabilistic Simulation Technique (PST) for Coastal Hazard Curves**

PyStorm, Module: PST (CyHAN v2.0 §5)

---

## Introduction

The **Probabilistic Simulation Technique (PST)** quantifies aleatory and
epistemic uncertainty in extreme-value hazard curves derived from a
Peaks-Over-Threshold (POT) sample. Given a univariate record of peaks (e.g.
storm-surge maxima), PST:

1. Selects a Generalized Pareto Distribution (GPD) threshold by minimizing a
   Quantile-Delta-Method (QDM) weighted-mean-square error (WMSE) over a
   candidate band.
2. Bootstraps the descending-sorted exceedances using truncated Gaussian (or
   Uniform) perturbations.
3. Fits a GPD to each realization and evaluates its inverse-CDF on a dense
   plotting grid of Annual Exceedance Frequencies (AEFs).
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
│           │   └── gpd_threshold.py QDM-WMSE threshold search
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

### 2.1 GPD Threshold Selection (QDM-WMSE)

Let `values_pot` be the sample sorted descending and
`weibull_aef[i] = (i + 1) / (n + 1) · λ` the empirical Weibull
plotting-position AEFs (scaled by the population intensity
`λ = n / record_length_years`). For each candidate threshold `θ` in the
percentile band `[θ_min, θ_max]` (default `20–80%` of the value range):

1. Take the exceedances `pot > θ` and the associated `aef`.
2. Fit a GPD with `floc = θ` to the exceedances.
3. Predict at the empirical positions and compute
   `WMSE = Σ wᵢ (potᵢ − predᵢ)² / Σ wᵢ`, with `wᵢ = 1/aefᵢ` for `aefᵢ < 1`.

The lowest-θ candidate whose normalized WMSE is within 5% of the minimum is
selected, preferring data-rich fits when the WMSE surface is flat.
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

For each bootstrap column the GPD is refit (`floc = θ`) and its shape `c` is
clipped to the Luceño-style band `[c_lo, c_hi]` (defaults `[-0.5, +0.33]`)
before the ICDF is evaluated on the plot AEF grid restricted to
`aef < λ_θ = (# exceedances) / record_length_years`. The realization stack is
collapsed to a best-estimate mean and the 10/90% percentile bounds.

The empirical bulk (`pot ≤ θ`) at its Weibull AEFs is concatenated below the
GPD tail; bulk uncertainty is taken as zero per the v1 convention. The merged
curve is then log-interpolated onto the 22-AER reporting grid
(`make_aef_grids()` in `hazard/curve.py`).

---

## 3. Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│  data/inputs/processed/<base>_POT.csv         (column = STORM_COLUMN) │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  [1]  read_pot_csv  →  values, λ = n / record_length_years            │
├──────────────────────────────────────────────────────────────────────┤
│  [2]  Sort descending; Weibull AEFs                                   │
├──────────────────────────────────────────────────────────────────────┤
│  [3]  select_gpd_threshold_qdm  →  θ                                  │
├──────────────────────────────────────────────────────────────────────┤
│  [4]  Split exceedances / bulk; λ_θ = |exceed| / record_length_years  │
├──────────────────────────────────────────────────────────────────────┤
│  [5]  BootstrapGenerator (_pst or Python fallback)                    │
│       → boot_matrix [n_pot × n_sims]                                  │
├──────────────────────────────────────────────────────────────────────┤
│  [6]  fit_gpd_ensemble → BE, CB10, CB90 on plot AEF grid              │
├──────────────────────────────────────────────────────────────────────┤
│  [7]  assemble_hazard_curve + interpolate_to_table                    │
├──────────────────────────────────────────────────────────────────────┤
│  [8]  write_pst_outputs + HazardCurvePlotter                          │
│       → data/outputs/<base>_PST*.csv                                  │
│       → data/outputs/plots/<base>_PST_HC.png                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Outputs

| File                       | Contents                                                          |
|----------------------------|-------------------------------------------------------------------|
| `<base>_PST.csv`           | Bootstrap GPD ensemble, plot AEF grid columns                     |
| `<base>_PST_HC_BE_tbl.csv` | Best estimate on the 22-AER reporting grid                        |
| `<base>_PST_HC_CB_tbl.csv` | 10/90% confidence bounds on the 22-AER grid                       |
| `<base>_PST_HC_BE_plt.csv` | BE on the dense plotting grid (merged GPD + empirical)            |
| `<base>_PST_HC_CB_plt.csv` | CB10/CB90 on the dense plotting grid                              |
| `<base>_PST_HC.png`        | Hazard-curve plot (empirical scatter + GPD curve + CB band)       |

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

# Or override on the CLI for ad-hoc runs:
python run_probabilistic_simulation_technique.py \
    --input data/inputs/processed/storm_surge_8518750_1920_2025_POT.csv \
    --record-length 106 \
    --num-simulations 1000 \
    --seed 628
```

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
| AEF      | Annual Exceedance Frequency                              |
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
| QDM      | Quantile Delta Method                                    |
| RNG      | Random Number Generator                                  |
| WMSE     | Weighted Mean Square Error                               |
| WPP      | Weibull Plotting Position                                |
