# probabilistic_simulation_technique

*Probabilistic Simulation Technique (PST) for coastal hazard curves with bootstrap confidence bands.*

PyStorm module

---

## Overview

PST turns a Peaks-Over-Threshold (POT) peak sample into a hazard curve: the
response magnitude as a function of its Annual Exceedance Rate (AER), with a
confidence band. The upper tail is a fitted Generalized Pareto Distribution
(GPD); the frequent range is carried empirically; the two are spliced into one
continuous curve.

The Monte Carlo loop that produces the confidence band runs in C++
(`backend/engines/PSTBootstrap.hpp`, exposed as `_pst`). A pure-Python fallback
runs when the kernel is not built. The launcher builds the kernel on first run.

## Method

Notation: μ = GPD location, ξ = shape, σ = scale.

### 1. Population rate

With `n_pot` peaks over a record of `record_length` years, the base rate is
λ_u = n_pot / record_length (events per year). `record_length` is taken from
`RECORD_LENGTH_YEARS`, or auto-derived as n_pot / `EVENTS_PER_YEAR`. The POT
module trims each output to exactly `EVENTS_PER_YEAR × effective_duration` peaks,
so the auto value recovers the effective duration. Keep `EVENTS_PER_YEAR` equal
to the POT target rate.

### 2. Empirical AER

Sort the peaks descending (rank i = 1 is the largest). Each peak gets an
unconditional AER_i = (i / (n_pot + 1)) × λ_u (Weibull plotting position).

### 3. GPD location μ

PST re-optimizes the threshold the GPD is fitted above (the location μ), distinct
from the POT extraction threshold `u`. Candidate μ values are scanned across a
band defined by empirical percentiles of the POT values (`BAND_FLOOR_PCT` to
`BAND_CEILING_PCT`, default 50 to 95). For each candidate the exceedances are fit
by a GPD (location fixed at μ, ξ clipped to the admissible band, σ refit when ξ is
clipped); the conditional rate above μ is λ_μ = n_exc / record_length; magnitudes
are predicted at the empirical AERs through the GPD inverse CDF using λ_μ (the
same convention as the hazard curve); and the fit is scored by a
frequency-weighted mean-square error WMSE = Σ wᵢ (xᵢ − pᵢ)² / Σ wᵢ, where pᵢ is
the GPD-predicted magnitude at AER_i and wᵢ = 1 / AER_i.

μ is chosen by `GPD_SELECTION`, one of four methods:

- **`wmse`** (default). The WMSE-tolerance set is every in-band candidate with at
  least `MIN_EXCEEDANCES` exceedances whose WMSE is within `WMSE_TOLERANCE` of the
  climb from the best fit (the floor) up to a robust ceiling: the highest in-band
  WMSE that is not a Tukey outlier (≤ Q3 + 1.5·IQR). So
  ceiling = best + `WMSE_TOLERANCE` × (robust_max − best). The upper anchor is as
  high as the in-band fits honestly reach but immune to a single freak-high WMSE,
  and `WMSE_TOLERANCE` keeps one meaning (a fraction of the floor→robust-max
  spread). The pool is the full in-band set, so a genuinely bounded short tail (ξ
  at the lower clip, for example a hurricane-dominated tail) stays in the set;
  `GPD_TIEBREAK` picks μ within it. When the pick is ξ-pinned at the clip the run
  prints a warning pointing to `stability`.
- **`stability`**. Among eligible candidates (in band, above the floor, ξ not
  pinned at the lower clip) the stability plateau is those within `STABILITY_TOL`
  of the minimum robust ξ-dispersion. That dispersion is the scaled MAD of ξ over
  a window of ±`STABILITY_WINDOW` neighbouring candidates: MAD is the median
  absolute deviation, median(|ξ − median(ξ)|), and "scaled" multiplies it by
  1.4826 so it reads on the same scale as a standard deviation for normal data
  while staying immune to a single anomalous fit (which would inflate the ordinary
  standard deviation). A near-zero value marks the flat-ξ threshold-stability
  shelf; `GPD_TIEBREAK` picks within it. Avoids the sparse-tail trap with no
  per-station tuning.
- **`mrl`**. Automated mean-residual-life (Langousis et al. 2016, WRR, eqs 4 to
  6): the lowest in-band threshold at which the mean-excess curve becomes linear
  (a local minimum of a weighted-least-squares fit). Non-parametric, since it fits
  a line to the mean-excess curve, not the GPD.
- **`gof`**. Choulakian-Stephens failure-to-reject: the lowest in-band threshold
  at which the GPD fit is not rejected by an EDF goodness-of-fit test
  (`GOF_STATISTIC` = Anderson-Darling A² or Cramer-von Mises W²) at significance
  `GOF_SIGNIFICANCE`, using the asymptotic critical values.

The GPD fit estimator is set by `GPD_FIT_METHOD`: `mle` (default) or `mom` (method
of moments, closed form, more robust for small or quantized samples). It applies
to both the selection and the bootstrap ensemble. Each selection method writes its
own diagnostics plot.

### 4. Tail and bulk split

Peaks above μ feed the GPD tail; peaks at or below μ are kept as empirical points.
The exceedance rate at μ is λ_μ = n_exc / record_length.

### 5. Smoothed bootstrap (confidence band)

A single GPD fit has no uncertainty band. The band comes from re-fitting the GPD
to `NUM_SIMULATIONS` smoothed-bootstrap resamples of the exceedances above μ. With
the exceedances descending-sorted (x₁ ≥ x₂ ≥ ... ≥ xₙ), each realization:

1. resamples n_exc values with replacement;
2. perturbs each by additive noise from a smoothing kernel whose bandwidth is the
   local order-statistic spacing to the next (i+1, adjacent smaller) value,
   sᵢ = xᵢ − xᵢ₊₁: x′ᵢ = xᵢ − sᵢ z, with z drawn from `BOOTSTRAP_DISTRIBUTION`
   ("gaussian" truncated normal or "uniform") on `BOOTSTRAP_TRUNCATION`;
3. sorts the realization descending.

A GPD is fit to each realization. The best estimate is the across-realization
mean; the confidence band is the 10th and 90th percentiles. `RANDOM_SEED` makes
the ensemble reproducible; more realizations give smoother bounds.

### 6. Hazard-curve assembly

The GPD tail (AER < λ_μ) is spliced onto the empirical bulk (AER ≥ λ_μ) and
interpolated in log-AER onto a fixed 22-point reporting grid spanning AER from 10
down to 1e-6 per year. (Mean return interval MRI = 1 / AER; PST reports in AER
throughout.)

## Inputs

`INPUT_MODE` selects the source:

- `station` (default): batch over the peaks_over_threshold outputs for
  `STATION_IDS`, choosing series via `PST_TARGETS` ("dwl", "ntr", or "both"),
  resolving to
  `peaks_over_threshold/data/outputs/<station>/<target>_<station>_pot.csv`. PST
  runs once per station and target.
- `path`: one POT CSV at `INPUT_CSV`. The peak-magnitude column is `STORM_COLUMN`
  (default "value").

CLI batch in path mode: pass one or more POT CSV paths as positional arguments.

## Outputs

Data files go to `data/outputs/<station>/` (path mode: `data/outputs/`); plots go
to the shared `data/outputs/plots/`.

| File | Contents |
|------|----------|
| `<base>_pst.csv` | bootstrap GPD ensemble on the dense AER grid |
| `<base>_pst_hc_be_tbl.csv` | best estimate on the 22-AER reporting grid |
| `<base>_pst_hc_cb_tbl.csv` | 10th and 90th confidence bounds on the 22-AER grid |
| `<base>_pst_hc_be_plt.csv` | best estimate on the dense plotting grid |
| `<base>_pst_hc_cb_plt.csv` | confidence band on the dense plotting grid |
| `<base>_pst_hc.png` | hazard-curve plot (empirical points, GPD mean and band, μ cross) |
| `<base>_qdo_threshold.png` | selection diagnostics (panels depend on `GPD_SELECTION`) |

## Quickstart

```bash
cd modules/probabilistic_simulation_technique

# Edit the USER OPTIONS block, then run (the C++ kernel builds on first run):
python run_probabilistic_simulation_technique.py

# CLI batch over explicit POT CSV paths (path mode):
python run_probabilistic_simulation_technique.py path/to/ntr_8518750_pot.csv

# Tests:
pytest tests/
```

For the default `INPUT_MODE = "station"`, set `STATION_IDS` and `PST_TARGETS`. For
`INPUT_MODE = "path"`, set `INPUT_CSV`.

## Programmatic API

Like every PyStorm module, PST exposes one entry point in
`backend/python/api_probabilistic_simulation_technique.py`:

```python
run(config) -> PSTResult | dict[str, PSTResult]
```

The launcher (`run_probabilistic_simulation_technique.py`) only assembles
`config` (input selection, parameters) and calls `run`. To drive PST from your
own code:

```python
import sys
sys.path.insert(0, "modules/probabilistic_simulation_technique/backend/python")
from api_probabilistic_simulation_technique import run

result = run(config)   # config: a dict (or a PSTConfig)
```

It returns a single **`PSTResult`** for one input, or a `{target: PSTResult}`
mapping when several inputs run (station mode). Each `PSTResult` carries the
`gpd_threshold`, the rates `lambda_val` / `lambda_mu`, the best-estimate and
confidence-bound hazard-curve tables (`hc_table_be`, `hc_table_cb10`,
`hc_table_cb90`), the AER grid (`aer_table`), the bootstrap `ensemble`, and
`used_cpp_kernel`.

## Method testbed

`scripts/method_testbed.py` runs all four selection methods through the full
pipeline on each station and stacks the hazard curves, one subplot per station,
with a per-method metrics box. Output goes to `data/outputs/plots/testbed/`, one
figure per series and fit, so results never overwrite one another.

```bash
python scripts/method_testbed.py                 # both series, both fits (default)
python scripts/method_testbed.py --series dwl --fit mom
```

## Layout

```text
probabilistic_simulation_technique/
├── run_probabilistic_simulation_technique.py   Launcher (user options only)
├── pyproject.toml                              Installable orchestrator package
├── ENGINE_MANIFEST.toml                        Structured module manifest
├── backend/
│   ├── engines/                                C++ smoothed-bootstrap kernel (_pst)
│   │   ├── PSTBootstrap.hpp                     Header-only kernel
│   │   ├── pst_bindings.cpp                     pybind11 → _pst
│   │   ├── CMakeLists.txt
│   │   └── build.py
│   └── python/
│       ├── api_probabilistic_simulation_technique.py   Orchestrator entry (input resolution)
│       └── probabilistic_simulation_technique/          Orchestration package
│           ├── config.py                       PSTConfig + BootstrapConfig (pydantic)
│           ├── orchestrator.py                 PSTOrchestrator
│           ├── gpd_fit.py                       Shared GPD fit (clip, σ-refit, mle/mom)
│           ├── sampling/bootstrap.py           Smoothed bootstrap (C++ or fallback)
│           ├── sampling/gpd_threshold.py       GPD-location selection (four methods)
│           ├── hazard/curve.py                 Ensemble fit, tail splice, table interp
│           ├── postproc/plots.py               Hazard-curve + selection diagnostics
│           └── io/pot_csv.py                   POT reader + result writers
├── scripts/method_testbed.py                   Four-method comparison testbed
├── tests/                                      Smoke tests
└── data/                                       inputs/{raw,processed}/ & outputs/ (gitignored)
```

Per CyHAN v2.2, the launcher holds user options only and calls the orchestrator
(`backend/python/api_probabilistic_simulation_technique.py`), which resolves the
inputs and runs PST on each. The module is self-contained.

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| AD | Anderson-Darling (EDF statistic) |
| AER | Annual Exceedance Rate |
| BE | Best Estimate |
| CB | Confidence Bound |
| CvM | Cramer-von Mises (EDF statistic) |
| GoF | Goodness of Fit |
| GPD | Generalized Pareto Distribution |
| MAD | Median Absolute Deviation |
| MLE | Maximum Likelihood Estimation |
| MoM | Method of Moments |
| MRI | Mean Return Interval (MRI = 1 / AER) |
| MRL | Mean Residual Life |
| POT | Peaks Over Threshold |
| PST | Probabilistic Simulation Technique |
| QDO | Quantile Delta Optimization |
| WMSE | Weighted Mean Square Error |
