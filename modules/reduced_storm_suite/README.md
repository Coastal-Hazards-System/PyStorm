# reduced_storm_suite

*Reduced Storm Suite (RSS) selection for probabilistic coastal hazard analysis.*

PyStorm module

> **Storm type (TC / ETC).** RSS runs in two storm-type modes, set by `STORM_TYPE`
> (`--storm-type`): **`tc`** (tropical cyclones, the implemented selection) and
> **`etc`** (extratropical cyclones, a *placeholder* that raises `NotImplementedError`
> for now; the same k-medoids / DSW / HC selection would run on an ETC
> synthetic-storm set). The rest of this README documents the `tc` mode.

---

## Introduction

Probabilistic coastal hazard analyses based on the Joint Probability Method
(JPM) require a representative subset of synthetic tropical cyclones (TCs) to
be selected from a much larger candidate suite. The cost of hydrodynamic
simulation makes brute-force enumeration impossible, so the selected subset
must simultaneously (a) span the multi-dimensional TC parameter space, (b)
span the hydrodynamic response space at the coastal nodes of interest, and
(c) reproduce benchmark hazard curves (HCs) within a stated tolerance after
discrete storm weights are back-computed.

The `reduced_storm_suite` module implements an end-to-end, reproducible pipeline
that addresses these three objectives jointly. Inputs are a TC parameter
matrix `X`, a peak surge response matrix `Y`, and an optional benchmark
hazard-curve cube `HC`. Outputs are a small set of representative storm
indices, a discrete storm weight per selected storm, a reconstructed hazard
curve at every node, and a post-correction bias surface obtained via Quantile
Bias Mapping (QBM). Two operating modes are exposed: a **fixed-k** mode that
runs the full diagnostic suite at a user-specified subset size, and an
**optimal-k** mode that grows the subset iteratively until coverage,
discrepancy, and HC-RMSE stopping criteria are met.

This document describes the methods, the data flow, and the workflow stages
in sufficient detail to be reproduced from the source.

---

## 1. Problem Statement

Let

- `X ∈ ℝ^{n × p}`: `n` candidate storms described by `p` TC parameters (e.g.
  central pressure deficit, radius of maximum wind, translation speed,
  heading, landfall location).
- `Y ∈ ℝ^{n × m}`: peak surge at `m` ADCIRC (Advanced Circulation Model) mesh
  nodes, one row per storm.
- `HC ∈ ℝ^{m × N_AER}`: benchmark hazard curves at each node, on a fixed
  table of Annual Exceedance Rates (AER).

The goal is to choose `k ≪ n` storm indices `S ⊂ {1, …, n}` such that

1. `X[S, :]` is space-filling in standardized parameter space,
2. `Y[S, :]` covers the principal modes of the response,
3. the JPM hazard curve reconstructed from `Y[S, :]` and the back-computed
   Discrete Storm Weights (DSW) reproduces `HC` to within a target tolerance.

The exact selection problem is combinatorial and has no efficient
solution at relevant scales. The `reduced_storm_suite` module attacks it
with a sequence of well-defined, computationally tractable surrogates.

---

## 2. Methods

### 2.1 Dimensionality Reduction of the Response Space (PCA / POD)

The response matrix `Y` has `m` columns, where `m` is the number of mesh
nodes (often `10^4` to `10^6`). Direct clustering in this dimension is
wasteful because the surge response is highly spatially correlated. We apply
Principal Component Analysis (equivalent here to Proper Orthogonal
Decomposition) and retain enough components to explain a user-set fraction
`τ` (default `τ = 0.95`) of the total variance:

`Y_r = PCA_τ(Y)   with   Y_r ∈ ℝ^{n × r},   r ≪ m`

The PCA is fit on the full set of `n` candidate storms so the basis encodes
the population-level response variability, not the subset's. Implementation:
`scikit-learn` full-SVD (Singular Value Decomposition) solver in `sampling/pca.py`.

### 2.2 The Joint Feature Matrix

Both objectives (parameter coverage and response coverage) must enter the
clustering simultaneously. We standardize `X` and `Y_r` independently and
concatenate them with user-controlled weights:

```
Z = [ α · z(X) | β · z(Y_r) ] ∈ ℝ^{n × (p + r)}
```

where `z(·)` is column-wise zero-mean, unit-variance standardization
(`StandardScaler`), and `(α, β)` are scalar weights controlling the relative
emphasis on parameters vs. response. The default `(α, β) = (10, 0.1)` biases
selection toward parameter coverage; alternative working points
`(α, β) ≈ (0.5, 1.0)` reverse the emphasis.

Implementation: `sampling/joint_matrix.py`.

### 2.3 Subset Selection: k-medoids (PAM - Partitioning Around Medoids)

Given the joint matrix `Z`, we seek a subset `S` of size `k` that minimises
the total within-cluster dissimilarity:

`S* = argmin_{|S|=k}  Σ_{i=1}^n  min_{j ∈ S}  ‖Z_i − Z_j‖_2`

This is the classic k-medoids objective, solved by Partitioning Around
Medoids (PAM) with maximin **BUILD** (initialization) and an exhaustive
**SWAP** (refinement) phase. Three back-ends dispatch in order:

1. A C++ kernel (`backend/engines/cpp/`, binding installed as
   `reduced_storm_suite._rss`) exposed through pybind11. Honours arbitrary
   forced medoids.
2. `sklearn-extra` `KMedoids(method="pam")` when no forced medoids are
   requested.
3. A pure-Python BUILD + SWAP fallback (`_greedy_kmedoids`).

A second, faster alternative is **greedy maximin (farthest-point) sampling**
(pure BUILD, no SWAP), used for the optional Sub-RSS stage when an
extreme-coverage subset of an already-selected subset is desired.

Implementation: `sampling/kmedoids.py`.

#### 2.3.1 Forced (pre-selected) storms

A subset of storms (for example, a previously simulated historical event
suite or an existing JPM deployment) can be locked into the selection by
passing them as `forced_indices`. PAM treats them as immovable medoids
during SWAP. This guarantees backward compatibility with existing
simulation suites while letting the algorithm choose the remaining
`k − |forced|` representatives.

### 2.4 Space-Filling Quality Metrics

Three diagnostics are computed on every candidate subset:

| Metric          | Definition                                                                                            | Direction |
|-----------------|-------------------------------------------------------------------------------------------------------|-----------|
| **Coverage**    | Fraction of k-means clusters fit on the full `Y_r` that are represented (assigned to) by `Y_r[S]`     | maximize  |
| **Discrepancy** | Centered L2 discrepancy of `z(X)[S]` mapped to `[0,1]^p`, measuring deviation from a uniform fill      | minimize  |
| **Maximin**     | Minimum pairwise Euclidean distance within `z(X)[S]`                                                  | maximize  |

Coverage and discrepancy quantify how well the subset fills the response and
parameter spaces, respectively. Maximin reports the worst-case crowding
among selected storms. Implementation: `sampling/metrics.py`.

### 2.5 Discrete Storm Weight (DSW) Back-Computation

For each selected storm `j`, the JPM framework requires a probability weight
`w_j` (the Discrete Storm Weight). We back-compute these weights from the
benchmark HC such that the reconstructed hazard curve is consistent with the
benchmark at the coastline.

**Nodal DSW**, at each node `i`:

1. Sort the `k` selected surges in descending order.
2. Interpolate the benchmark HC at node `i` in log-AER space to obtain the
   AER associated with each sorted surge.
3. Finite-difference the AER sequence to obtain a nodal DSW per sorted
   storm. Clip negative values (caused by non-monotone benchmark segments)
   to zero and emit a warning.
4. Map back to the original storm order.

**Active node filter**: a node is included in the global average only if at
least `min_wet_storms` (default 2) selected surges exceed the dry threshold
and the HC interpolation returns at least one finite AER.

**Global DSW**, one weight per storm, aggregated across active nodes:

`w_j = Σ_i  W_i · dsw_{ij}  /  Σ_i  W_i · 𝟙[storm j active at node i]`

Three aggregation modes are supported via `dsw_method`:

| Method | Per-node weight `W_i`                | Notes                                            |
|--------|--------------------------------------|--------------------------------------------------|
| 1      | `1`                                  | Equal-weight average (classic JPM).              |
| 2      | per-storm-per-node surge             | Surge-weighted, emphasizes high responses.       |
| 3      | `Var_j(Y_{ji})` (default)            | Variance-weighted, emphasizes informative nodes. |

Implementation: `weights/dsw.py`.

### 2.6 Hazard Curve Reconstruction (JPM)

Given selected surges and global DSWs, the hazard curve at each node is
reconstructed by JPM integration:

1. At node `i`, sort the `k` storms by descending surge.
2. Form the cumulative AER `Λ_l = Σ_{j ≤ l} w_{(j)}` (largest-first).
3. Interpolate `(log Λ_l, surge_{(l)})` onto the standard `tbl_aer` grid.

This yields a reconstructed hazard curve `HC_recon ∈ ℝ^{m × N_AER}` that
can be compared cell-by-cell with the benchmark.

**Residual metrics**: nodal bias, uncertainty (std), and Root-Mean-Square
Error (RMSE); the mean across nodes is reported as `mean_bias`,
`mean_uncertainty`, `mean_rmse`.
Per-AER-level biases (`bias_aer10`, `bias_aer100`, `bias_aer1000` - the AER
hazard levels at MRI = 10, 100, 1000 yr, i.e. AER = 1/N) are also reported to
highlight tail performance.

### 2.7 Quantile Bias Mapping (QBM) Post-Correction

DSW reconstruction leaves a residual bias that is typically smooth in
log-AER. Quantile Bias Mapping is a post-hoc, node-wise correction applied
on top of the reconstructed HC:

- **`qbm_mode = "aer"`** *(default)*: horizontal correction. At each node,
  surge values are left untouched; the cumulative AER at each storm's
  position is remapped through the inverse benchmark HC so the corrected
  `(AER, surge)` pairs land on the benchmark curve. Physical model outputs
  are preserved.

- **`qbm_mode = "response"`** *(legacy)*: vertical correction. Surge values
  are shifted to match the benchmark at the storm's cumulative AER
  position. Alters the model output; retained for back-compatibility.

Both modes use a shared smoothing infrastructure:

1. Raw bias is computed at each storm's cumulative-AER position.
2. Bias is interpolated onto an intermediate AER grid (dense 631-point
   grid, default, or the standard 22-point `tbl_aer`).
3. A Gaussian kernel (window fraction `win_frac`, default 0.10) is applied
   in index space.
4. A C¹ smoothstep ramp (`ramp_frac`, default 0.03) tapers the smoothed
   curve back to the raw bias at the endpoints.
5. Monotonicity is enforced on the corrected output.

Output is a per-node, per-AER bias table written to `qbm_bias.h5`.
Implementation: `weights/qbm.py`.

### 2.8 Geographic Bounding-Box Filter (Optional)

For regional studies, only a sub-region of the mesh and a sub-set of the TC
suite are relevant. The bbox stage performs three operations:

1. **Node filter**: drop `Y` columns whose `(lat, lon)` fall outside the
   user box.
2. **Geographic medoid**: find the surviving mesh node whose summed
   haversine distance to the other surviving nodes is minimum. This medoid
   serves as a region-representative point.
3. **Storm filter**: drop storms whose track (loaded from per-storm ITCS
   TROP files) does not approach within `max_track_dist_km` of the medoid.

Forced (pre-selected) storms that fall outside the radius are automatically
re-added so they remain available downstream. Implementation:
`geo/bbox_filter.py`.

---

## 3. Workflow

The module exposes a single canonical launcher
(`run_reduced_storm_suite.py` at the module root, CyHAN §5.3). It dispatches
to one of two workflows based on `--mode`:

```
  --mode fixed    →  workflows/rss_selection.run_rss_selection      (§3.2)
  --mode optimal  →  workflows/growth_evaluation.run_growth_evaluation (§3.3)
```

The two workflows share the upstream data-prep stages - bounding-box
filter, data load, PCA, joint-matrix build - and the downstream HC
verification and QBM post-correction. They diverge at the **selection
phase** and at the outputs that selection produces. Diagrams for each
mode are given below; step numbers are aligned across modes where the
step is the same.

### 3.1 Shared upstream stages

Both workflows begin with the same data-prep pipeline through step [2]:

```
┌─────────────────────────────────────────────────────────────────────┐
│  preprocess.py  →  data/inputs/processed/tc_data.h5                 │
│  (raw X / Y / HC / track files  →  validated HDF5 store)            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [0]  Bounding-box filter            (optional, geo/bbox_filter.py) │
│       └─ node filter • medoid • track distance filter               │
├─────────────────────────────────────────────────────────────────────┤
│  [1]  Load X, Y, HC from HDF5                                       │
│       └─ apply bbox storm/node indices • apply node stride          │
├─────────────────────────────────────────────────────────────────────┤
│  [2]  PCA on Y                                       (sampling/pca) │
│       └─ retain τ = 95% variance  →  Y_r                            │
└─────────────────────────────────────────────────────────────────────┘
```

The paths diverge at step [3]:

- **Fixed-k** runs an optional α/β sweep at step [3], then builds the joint
  matrix at step [4] with the optimized (or default) α, β.
- **Optimal-k** has no α/β sweep and goes straight to step [4] with
  `alpha_default`, `beta_default`.

The `build_joint_matrix` call itself is the same in both modes - the only
difference is the α, β values it receives.

### 3.2 Fixed-k workflow

Runs the full pipeline once at a user-specified subset size
`k = |forced| + k_additional`. Suitable for downstream consumers that
need a known subset size (e.g. a fixed hydrodynamic simulation budget).
All diagnostic stages (α/β sweep, Sub-RSS, HC verification, QBM) are
available; HC-dependent stages are skipped automatically when `HC_bench`
is absent.

```
                              [0]-[2]  shared upstream  (§3.1)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [3]  (Optional)  α/β sweep                  (HC-driven optimizer)  │
│       └─ for each (α, β) ∈ grid:                                    │
│              build Z, run PAM, evaluate HC metrics,                 │
│              keep argmin |bias| + rmse                              │
├─────────────────────────────────────────────────────────────────────┤
│  [4]  Build joint matrix Z = [α·z(X) | β·z(Y_r)]   (sampling/joint) │
│       └─ uses optimized α, β from [3] (or defaults if [3] skipped)  │
├─────────────────────────────────────────────────────────────────────┤
│  [5]  Select k medoids via PAM - single call    (sampling/kmedoids) │
│       └─ k = |forced| + k_additional  • honours forced indices      │
├─────────────────────────────────────────────────────────────────────┤
│  [6]  Space-filling metrics      (sampling/metrics - one row)       │
│       └─ coverage • discrepancy • maximin                           │
├─────────────────────────────────────────────────────────────────────┤
│  [7]  Save  selected_storms.csv  +  selection_metrics.csv           │
├─────────────────────────────────────────────────────────────────────┤
│  [8]  Diagnostic plots          (postproc/plots)                    │
│       └─ Y-space PCA scatter (initial + final)                      │
│       └─ X-parameter SPLOM   (initial + final)                      │
├─────────────────────────────────────────────────────────────────────┤
│  [9]  (Optional)  Sub-RSS                                          │
│       └─ within         → PAM on Z[indices]                         │
│       └─ within_maximin → greedy farthest-point on Z[indices]       │
│       └─ additional     → PAM on Z with (forced + k_sub) forced     │
├─────────────────────────────────────────────────────────────────────┤
│ [10]  HC verification           (weights/dsw, if HC_bench present)  │
│       └─ compute_global_dsw  →  reconstruct_hc_global_dsw           │
│       └─ overlay reconstructed vs benchmark HC at 9 sample nodes    │
├─────────────────────────────────────────────────────────────────────┤
│ [11]  QBM post-correction       (weights/qbm, if HC_bench present)  │
│       └─ compute_qbm_bias   →  qbm_bias.h5                          │
│       └─ overlay corrected vs benchmark HC at 9 sample nodes        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Optimal-k workflow

The user inputs a minimum subset size `k_min`, a maximum subset size
`k_max`, a step `k_step`, and a global RMSE tolerance
(`rmse_threshold`, units: m). The workflow sweeps the full range,
evaluating PAM + DSW + HC + RMSE at every `k`, then picks the smallest
`k` whose reconstructed hazard curve meets the tolerance.

```
                              [0]-[2]  shared upstream  (§3.1)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [4]  Build joint matrix Z = [α·z(X) | β·z(Y_r)]   (sampling/joint) │
│       └─ uses alpha_default, beta_default  (no α/β sweep in this    │
│          mode - step [3] is skipped)                                │
├─────────────────────────────────────────────────────────────────────┤
│  [5]  Growth sweep - for k in [k_min, k_max] step k_step:           │
│           PAM(Z, k, forced)              (sampling/kmedoids, C++)   │
│           SF metrics  (coverage, discrepancy, maximin)              │
│           DSW back-compute               (weights/dsw)              │
│           HC reconstruction via JPM      (weights/dsw)              │
│           record (k, coverage, discrepancy, RMSE, bias, bias_aerN)  │
│       └─ runs to completion - no early stopping                     │
├─────────────────────────────────────────────────────────────────────┤
│  [6]  Pick k_selected:                                              │
│           smallest k with global RMSE ≤ rmse_threshold              │
│           else argmin global RMSE  (warning emitted)                │
├─────────────────────────────────────────────────────────────────────┤
│  [7]  Re-select RSS at k_selected   (sampling/kmedoids)            │
├─────────────────────────────────────────────────────────────────────┤
│  [8]  Save  selected_storms.csv  +  growth_history.csv              │
│       Per-node RMSE at k_selected  →  node_rmse.csv                 │
├─────────────────────────────────────────────────────────────────────┤
│  [9]  Plot  global RMSE vs k   →   rmse_vs_k.png                    │
│       └─ tolerance horizontal line • selected-k vertical line       │
├─────────────────────────────────────────────────────────────────────┤
│ [10]  Diagnostic plots at k_selected     (postproc/plots)           │
│       └─ Y-space PCA scatter • X-parameter SPLOM   (single set)     │
├─────────────────────────────────────────────────────────────────────┤
│ [11]  HC verification at k_selected     (weights/dsw - REQUIRED)    │
│       └─ overlay reconstructed vs benchmark HC at 9 sample nodes    │
├─────────────────────────────────────────────────────────────────────┤
│ [12]  QBM post-correction               (weights/qbm)               │
│       └─ compute_qbm_bias   →  qbm_bias.h5                          │
│       └─ overlay corrected vs benchmark HC at 9 sample nodes        │
└─────────────────────────────────────────────────────────────────────┘
```

Pseudo-code for the inner loop:

```
# Sweep
for k in range(k_min, k_max + 1, k_step):
    indices     ← PAM(Z, k, forced)                  (sampling/kmedoids)
    sf_metrics  ← coverage, discrepancy, maximin     (sampling/metrics)
    nodal_DSW   ← back-compute from HC_bench         (weights/dsw)
    DSW_global  ← aggregate across active nodes      (weights/dsw)
    HC_recon    ← JPM(Y[indices], DSW_global)        (weights/dsw)
    node_RMSE   ← sqrt(mean((HC_recon − HC_bench)^2, axis=AER))
    global_RMSE ← nanmean(node_RMSE)
    record(k, **sf_metrics, global_RMSE, mean_bias, bias_aerN)

# Selection
if any k has global_RMSE ≤ rmse_threshold:
    k_selected ← smallest such k
else:
    k_selected ← argmin global_RMSE      (warning emitted)
```

Key properties:

- The sweep is **not** stopped early. The full RMSE-vs-k curve is always
  produced so the user can see the shape of the trade-off.
- `coverage` and `discrepancy` are recorded in `growth_history.csv` for
  inspection, but are **not** used as stopping criteria.
- HC reconstruction is **required** (no `HC_bench` ⇒ hard error) because
  the RMSE tolerance is the operative selection criterion.
- If `forced` storms are configured and `k_min < |forced|`, `k_min` is
  raised to `|forced|` automatically.
- The α/β sweep (§3.2 step [3]) and Sub-RSS (§3.2 step [9]) are
  **not** available in optimal-k mode.

### 3.4 What's different at a glance

| Stage                          | Fixed-k                     | Optimal-k                                    |
|--------------------------------|-----------------------------|----------------------------------------------|
| α/β sweep (step [3])           | Optional                    | Skipped (uses `alpha_default`, `beta_default`) |
| Selection (step [5])           | Single `PAM(Z, k)` call     | Sweep over `k` of `PAM + SF + DSW + HC + RMSE` |
| SF metrics                     | One row                     | One row per `k` → `growth_history.csv`       |
| Sub-RSS                       | Available                   | Not invoked                                  |
| Extra outputs vs the other mode| -                           | `growth_history.csv`, `node_rmse.csv`, `rmse_vs_k.png` |
| Diagnostic plots               | Initial + final (PCA, SPLOM)| At `k_selected` only                         |
| HC verification (step [10]/[11])| Conditional on `HC_bench`  | Required (`HC_bench` mandatory)              |
| QBM post-correction            | Conditional on `HC_bench`   | Required                                     |

---

## 4. Outputs

Outputs are written to `data/outputs/<dataset>/<scope>/<mode>/`, keyed by
dataset, scope, and mode so that runs never overwrite each other.

| File                       | Mode  | Contents                                                          |
|----------------------------|-------|-------------------------------------------------------------------|
| `selected_storms.csv`      | both  | Original-index, storm_id, TC parameters of the RSS               |
| `selected_storms_sub.csv`  | fixed | Sub-RSS (optional)                                               |
| `selection_metrics.csv`    | fixed | coverage / discrepancy / maximin at the chosen k                  |
| `growth_history.csv`       | opt.  | One row per `k` step: coverage, discrepancy, global RMSE, bias    |
| `node_rmse.csv`            | opt.  | Per-node RMSE and bias at `k_selected`                            |
| `alpha_beta_sweep.csv`     | fixed | Per-(α,β) HC metrics from the optimizer sweep                     |
| `qbm_bias.h5`              | both  | Post-correction bias table `[m × N_AER]`                          |
| `tc_splom_*.png`           | both  | TC parameter scatter-plot matrix                                  |
| `pca_yspace_*.png`         | both  | Response-space PCA scatter                                        |
| `hc_comparison*.png`       | both  | Reconstructed vs benchmark HC at 9 sample nodes                   |
| `hc_qbm*.png`              | both  | QBM-corrected vs benchmark HC at 9 sample nodes                   |
| `rmse_vs_k.png`            | opt.  | Global RMSE vs k, with tolerance and selected-k marker            |
| `bbox_map.png`             | both  | Mesh + bbox + storm tracks (only if bbox active)                  |

---

## 5. Data Contract

The pipeline reads a single HDF5 store (`tc_data.h5`) produced by the
ingestion stage (`scripts/preprocess.py` → `workflows/ingest.py`). The
store schema is:

| Path                         | Shape              | Dtype     | Notes                              |
|------------------------------|--------------------|-----------|------------------------------------|
| `/X`                         | `[n × p]`          | `float64` | TC parameter matrix                |
| `/X.attrs/param_names`       | `[p]`              | `str`     | Column labels                      |
| `/X.attrs/storm_ids`         | `[n]`              | `str`     | Optional storm identifiers         |
| `/Y`                         | `[n × m]`          | `float32` | Peak surge at every mesh node      |
| `/Y.attrs/node_ids`          | `[m]`              | `str`     | Mesh node identifiers              |
| `/HC`                        | `[m × N_AER]`      | `float64` | Benchmark hazard curve cube        |
| `/HC.attrs/aer_levels`       | `[N_AER]`          | `float64` | AER table (default 22-level grid)  |

The `/HC` group is optional. The fixed-k workflow degrades gracefully
without it (skips α/β sweep and verification plots), but the optimal-k
workflow requires it.

---

## 6. Implementation Notes

### 6.1 Engine isolation (CyHAN §4.1, §4.2, §16.4)

- The C++ k-medoids kernel lives in `backend/engines/cpp/` and exposes only
  `kmedoids_pam(D, k, seed, forced)`. It carries no I/O, no configuration,
  and no orchestration logic.
- All workflow logic is Python and resides in
  `backend/python/reduced_storm_suite/`. The binding is imported as a conduit
  by `sampling/kmedoids.py`; the Python package owns the dispatch chain
  (C++ → sklearn-extra → pure-Python fallback).

### 6.2 Reproducibility

A single `random_seed` (default 42) seeds:

- the k-medoids BUILD initialization,
- k-means fitting in the coverage metric,
- pseudo-random sampling for diagnostic node selection.

All other operations are deterministic.

### 6.3 Numerical safeguards

- Non-monotone segments of the benchmark HC can produce negative nodal DSW
  values; these are clipped to zero and a `RuntimeWarning` is emitted
  reporting the total count.
- Dry nodes (insufficient wet storms among the selected subset) are
  excluded from the global DSW average to avoid biasing the weights.
- QBM enforces monotonicity on the corrected output (`aer` mode:
  non-decreasing cumulative AER; `response` mode: non-increasing surge
  with increasing AER).

---

## 7. Quickstart

```bash
cd modules/reduced_storm_suite

# Edit USER OPTIONS in run_reduced_storm_suite.py, then run. The C++ k-medoids
# kernel builds on first run (pure-Python fallback otherwise), and tc_data.h5
# is built from the raw inputs automatically if it is missing.
python run_reduced_storm_suite.py                # uses the MODE constant
python run_reduced_storm_suite.py --mode fixed
python run_reduced_storm_suite.py --mode optimal

# Optional ancillary tools:
python scripts/preprocess.py     # rebuild tc_data.h5 from raw inputs
python scripts/dsw.py            # standalone post-selection DSW + HC reconstruction
```

Outputs land under `data/outputs/<dataset>/<scope>/<mode>/`.

---

## 8. Module Layout (CyHAN v2.2 §16.1)

```
reduced_storm_suite/
├── run_reduced_storm_suite.py            Launcher (user options only)
├── pyproject.toml                        Installable orchestrator package
├── ENGINE_MANIFEST.toml                  Structured module manifest
├── backend/
│   ├── engines/cpp/                      C++ k-medoids kernel (_rss)
│   │   ├── kmedoids_core.hpp             Header-only PAM with FastPAM1 refinement
│   │   ├── bindings.cpp                  pybind11 → _rss
│   │   ├── CMakeLists.txt
│   │   └── build.py
│   └── python/
│       ├── api_reduced_storm_suite.py   Orchestrator entry (input + path wiring)
│       └── reduced_storm_suite/          Orchestration package
│           ├── config/                   defaults, YAML loader, templates
│           ├── io/                       HDF5 store, multi-format readers
│           ├── geo/                      bbox filter, basemap rendering
│           ├── sampling/                 PCA, joint matrix, k-medoids, space-filling metrics
│           ├── weights/                  DSW back-comp, QBM bias correction
│           ├── postproc/                 diagnostic plots
│           └── workflows/                ingest, rss_selection, growth_evaluation
├── scripts/                              Ancillary tools
│   ├── preprocess.py                     Raw inputs → tc_data.h5
│   └── dsw.py                            Post-selection DSW + HC reconstruction
├── tests/                               Smoke + round-trip tests
└── data/                                inputs/{raw,processed}/ & outputs/<dataset>/<scope>/<mode>/ (gitignored)
```

The two mandatory entry artifacts per CyHAN v2.2 §5.3:

| Artifact     | Location                                       | Role               |
|--------------|------------------------------------------------|--------------------|
| Launcher     | `run_reduced_storm_suite.py`                   | user-facing entry  |
| Orchestrator | `backend/python/api_reduced_storm_suite.py`   | non-user-facing    |

---

## 9. References

### Joint Probability Method and Coastal Hazards System (Nadal-Caraballo et al.)

- Nadal-Caraballo, N. C., and Melby, J. A. (2014). *North Atlantic Coast
  Comprehensive Study Phase I: Statistical Analysis of Historical Extreme
  Water Levels with Sea Level Change*. ERDC/CHL TR-14-7. U.S. Army Engineer
  Research and Development Center, Vicksburg, MS.
- Nadal-Caraballo, N. C., Melby, J. A., Gonzalez, V. M., and Cox, A. T.
  (2015). *Coastal Storm Hazards from Virginia to Maine*. ERDC/CHL TR-15-5.
  U.S. Army Engineer Research and Development Center, Vicksburg, MS.
- Nadal-Caraballo, N. C., Gonzalez, V. M., Melby, J. A., and Cialone, M. A.
  (2016). Wave and water level statistics for coastal flood and risk
  studies. *Proceedings of the 35th International Conference on Coastal
  Engineering*, Antalya, Turkey.
- Nadal-Caraballo, N. C., Campbell, M. O., Gonzalez, V. M., Torres, M. J.,
  Melby, J. A., and Taflanidis, A. A. (2020). Coastal Hazards System: A
  probabilistic coastal hazard analysis framework. *Journal of Coastal
  Research*, Special Issue No. 95, 1211 to 1216.
- Nadal-Caraballo, N. C., Gonzalez, V. M., Chouinard, L. M., Cialone, M. A.,
  Melby, J. A., and Cox, A. T. (2022). Joint probability analysis of
  tropical cyclones for coastal hazard assessment: Methodology and
  applications. *Coastal Engineering*, 178, 104129.
- Nadal-Caraballo, N. C., Campbell, M. O., Gonzalez, V. M., Torres, M. J.,
  Massey, T. C., and Taflanidis, A. A. (2023). *StormSim Coastal Hazards
  Rapid Prediction System*. U.S. Army Engineer Research and Development
  Center, Coastal and Hydraulics Laboratory.

### Algorithmic background

- Resio, D. T., Irish, J. L., and Cialone, M. A. (2009). A surge response
  function approach to coastal hazard assessment. Part 1: Basic concepts.
  *Natural Hazards*, 51(1), 163 to 182.
- Toro, G. R., Resio, D. T., Divoky, D., Niedoroda, A. W., and Reed, C.
  (2010). Efficient joint-probability methods for hurricane surge frequency
  analysis. *Ocean Engineering*, 37(1), 125 to 134.
- Kaufman, L., and Rousseeuw, P. J. (1990). *Finding Groups in Data: An
  Introduction to Cluster Analysis*. Wiley (PAM algorithm).
- Joe, S., and Kuo, F. Y. (2008). Constructing Sobol' sequences with better
  two-dimensional projections. *SIAM Journal on Scientific Computing*,
  30(5), 2635 to 2654 (centered L2 discrepancy).
- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer Series
  in Statistics.

---

## 10. Acronyms

| Acronym  | Expansion                                                               |
|----------|-------------------------------------------------------------------------|
| ADCIRC   | Advanced Circulation Model (hydrodynamic surge solver)                  |
| AER      | Annual Exceedance Rate                                                  |
| BE       | Best-Estimate (hazard curve)                                            |
| CHS      | Coastal Hazards System                                                  |
| CHS-LA   | Coastal Hazards System - Louisiana study                                |
| CLI      | Command-Line Interface                                                  |
| CyHAN    | C++/Python Hybrid Architecture Network                                  |
| DSW      | Discrete Storm Weight                                                   |
| HC       | Hazard Curve                                                            |
| HTTP     | Hypertext Transfer Protocol                                             |
| ITCS     | Initial Tropical Cyclone Suite                                          |
| JPM      | Joint Probability Method                                                |
| PAM      | Partitioning Around Medoids (k-medoids algorithm)                       |
| PCA      | Principal Component Analysis                                            |
| POD      | Proper Orthogonal Decomposition                                         |
| QBM      | Quantile Bias Mapping                                                   |
| RMSE     | Root-Mean-Square Error                                                  |
| RSS     | Reduced Storm Suite                                          |
| SF       | Space-Filling (subset-quality metrics: coverage, discrepancy, maximin)  |
| SPLOM    | Scatter-Plot Matrix (pairs plot)                                        |
| SVD      | Singular Value Decomposition                                            |
| TC       | Tropical Cyclone                                                        |
| TROP     | TC track file format ("tropical") used by ITCS storm files              |
| UI       | User Interface                                                          |
| YAML     | YAML Ain't Markup Language (human-readable config format)               |
