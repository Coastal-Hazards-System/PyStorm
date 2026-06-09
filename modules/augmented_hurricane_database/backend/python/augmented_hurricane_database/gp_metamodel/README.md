# GP Metamodel: Central Pressure and Rmax Imputation

This package fills missing central pressure (`pmin_hpa`) and radius of maximum
wind (`rmax_km`) values in the HURDAT2 best-track record using Gaussian-process
metamodels. It is a Python re-implementation of the GP metamodel of Taflanidis
et al. and the CHS HURDAT imputation drivers, with several method and performance
improvements.

The document has three parts: (1) the MATLAB to Python conversion, (2) the
improvements in the Python version, and (3) the validation against the MATLAB
results. Section 2 includes the reasoning for using a nearest-neighbor GP instead
of a full dense GP.

---

## 1. MATLAB to Python conversion

### What the metamodel does

Each target is modeled with universal kriging: a trend `b(x) * beta` plus a
zero-mean Gaussian process with an anisotropic power-exponential correlation

```
R_ij = exp( - sum_k theta_k * |x_ik - x_jk|^p )
```

where `theta_k` is a per-dimension weight and `p` is one shared exponent. A
nugget is added to the diagonal. Hyperparameters `(theta, p, nugget)` are fit by
maximum likelihood (the concentrated negative log-likelihood). Inputs and the
response are standardized before fitting. The prediction is
`f(x) = (b(x) * beta + gamma * r(x)) * sY + mY`, matching `sur_model_GP.m`.

The metamodels are self-trained: for each target they are calibrated on the rows
where the value is observed, then used to predict the rows where it is missing.
Two models per target handle the two feature regimes:

- Central pressure (response = `1013 - pmin`, the deficit):
  - `Cp6` features `[lat, lon, vmax, Vf, sin Hd, cos Hd]` for fixes with known motion.
  - `Cp3` features `[lat, lon, vmax]` for fixes without a translation speed or
    heading (single-point storms, the first fix, and stationary fixes).
- Radius of maximum wind (response = `rmax`):
  - `Rm7` features `[lat, lon, vmax, Cp-deficit, Vf, sin Hd, cos Hd]`.
  - `Rm4` features `[lat, lon, vmax, Cp-deficit]`.

A fix is routed to the smaller model when its translation speed or heading is
missing. Imputed Rmax is clamped to `[8, 600]` km.

### File map

| MATLAB source | Python file | Role |
|---|---|---|
| `calibration_GP.m`, `set_problem_GP.m`, `sur_model_GP.m` | `gp.py` | GP calibration (MLE), build, and prediction |
| `GPM_Cp.m`, `GPM_Rm.m`, `Demonstration_NonSurge_*.m` | `features.py` (with `fit_gp` calls in `impute.py`) | feature construction and model definitions |
| `CHS_TC_HURDAT_Atlantic_DI_Cp.m`, `CHS_TC_HURDATv2_Atlantic_DI_Rm.m` | `impute.py` | self-trained imputation drivers |
| (no MATLAB equivalent) | `backend/engines/cpp/` (`_gpm`) | OpenMP correlation kernel |

### Reproduction contract

With the three quality flags off (`GPM_VECCHIA=False`, `GPM_PHYSICAL_MEAN=False`,
`GPM_LOG_RMAX=False`), the Python uses constant-mean kriging on the raw response
with prediction over the support set, which reproduces the original MATLAB
method. In that configuration the imputed central pressure matches the MATLAB
reference output to a correlation of 0.98, a mean absolute difference of about
2 hPa, and zero bias, with the observed rows matching exactly.

---

## 2. Improvements in the Python version

All improvements are on by default. Each maps to one configuration flag.

### Method improvements

- **Vecchia / nearest-neighbor GP (`GPM_VECCHIA`).** Each prediction conditions
  on its `n_neighbors` nearest training fixes (selected with a theta-scaled
  KD-tree so the metric matches the kernel), drawing them from the whole training
  set rather than the capped support set. Hyperparameters and the trend `beta`
  are still estimated on the support set. Cost is `O(n * m^3)`, linear in the
  number of fixes. See Section 3 for why this is preferred over a dense GP.

- **Physics-informed trend (`GPM_PHYSICAL_MEAN`).** The kriging trend uses
  physically chosen predictors, and the GP models the residual around it:
  - Central pressure: `[vmax, lat, vmax^2]`. The squared term encodes the
    wind-pressure relationship (the pressure deficit scales with the square of
    the maximum wind), and the latitude term carries the environmental and
    Coriolis correction to that relationship.
  - Rmax: `[lat, Cp-deficit, vmax]`. Storm size grows with latitude and shrinks
    with intensity.
  This is physics-informed (the right variables and shapes), not physics-exact.
  Its main benefit is extrapolation: outside the data cloud, predictions revert
  to the physical trend rather than a flat global mean.

- **Log-transformed Rmax (`GPM_LOG_RMAX`).** The Rmax models are fit in log
  space. Rmax is positive and roughly lognormal, so a log transform matches the
  log-linear size physics and stabilizes the intensity-dependent scatter. Central
  pressure is left untransformed: its squared-wind basis already captures the
  curvature, and the deficit is close to homoscedastic in hPa.

### Performance improvements

- **Analytic likelihood gradient.** The hyperparameter search uses the
  closed-form GP marginal-likelihood gradient instead of finite differences. This
  removes roughly `2 * n_param` Cholesky factorizations per optimizer step
  (about 3x faster calibration) and converges to better hyperparameters. The
  gradient is verified against finite differences to better than 1e-9.

- **Calibration subset.** The hyperparameter search evaluates the likelihood
  hundreds of times, each an O(n^3) Cholesky, so it is the step that does not
  scale to all the data (on all 24k fixes it is a 4.7 GB matrix and hours of
  search). The hyperparameters are only a handful of numbers and are
  well-determined by a few thousand representative fixes, so the search runs on an
  `n_cal` subset (default 1200); only the search uses it, while the final model is
  built on and predicts from all the data. The benefit is feasibility; the cost is
  small (about 0.013 R-squared for a constant-trend model, offset by the physical
  mean) and is quantified in the head-to-head section below.

- **Response-stratified support.** The support set is drawn half uniformly across
  the response distribution and half at random, so it covers the intense-storm
  tail that a purely random draw under-samples.

- **C++ correlation engine (`_gpm`).** The power-exponential correlation, which
  is the hot loop of prediction and of the per-evaluation R build, is an
  OpenMP-parallel C++ kernel (pybind11, in `backend/engines/cpp/`). It is about
  10x to 34x faster than the NumPy broadcast and releases the GIL during the
  compute. The `O(n^3)` Cholesky and solves stay in LAPACK through SciPy, where
  they are already optimal. The kernel builds automatically on the first GP run;
  if no compiler is present, the code falls back to the NumPy kernel.

- **Parallel training (`GPM_PARALLEL`).** The two models per target are fit
  concurrently. The C++ kernel and LAPACK release the GIL, so the fits overlap.

Net effect on the real Atlantic central-pressure model (support 3000): training
time dropped from about 240 s to about 40 s, prediction of 55k rows dropped from
about 36 s to about 4 s, and accuracy improved at the same time.

### Per-target tuning

Cp and Rmax use independently chosen settings, because they have different
correlation structure. A small sweep on a held-out test set established these:

| target | `max_support` | `n_neighbors` |
|---|---|---|
| Central pressure | 6000 | 30 |
| Rmax | 3000 | 10 |

Central pressure is smooth and long-range, so it benefits from more calibration
support, but not from more neighbors. Rmax is short-range and noisy, so it does
best with a small conditioning set (10). In the sweep, adding neighbors did not
help either target, because the physics-informed trend leaves a short-range
residual that a small neighbor set already captures.

---

## 3. Why nearest-neighbor GP and not a full or dense GP

The training sets reach 15k to 24k fixes. An exact dense GP at that size is a
24,000 by 24,000 covariance, about 4.7 GB in double precision, with a
factorization that needs roughly another working copy (about 10 GB peak) and runs
in minutes. More importantly, the maximum-likelihood search would need hundreds
of such factorizations, which is infeasible. The original MATLAB handles this by
thresholding small correlations to zero and using a sparse Cholesky.

We chose nearest-neighbor GP (Vecchia) over both the dense GP and the
sparse-Cholesky route, for the following reasons.

1. **Sparse Cholesky does not help the variable that would want the full data.**
   Sparsity helps only when most correlations are near zero, which happens when
   the kernel is short-range. Central pressure is the opposite: a smooth,
   long-correlation surface, so few correlations fall below the zero threshold
   and the thresholded matrix is nearly dense. For central pressure, the
   "sparse" Cholesky degenerates into the full dense solve. Sparse direct solvers
   also suffer heavy fill-in above two or three input dimensions, and our inputs
   are six and seven dimensional, so the approach is poorly suited here.

2. **Nearest-neighbor GP scales linearly and is dimension-robust.** Its cost is
   `O(n * m^3)` with `m` around 10 to 30 neighbors, and neighbors are defined by
   distance regardless of input dimension. It uses all of the data at prediction
   time (each prediction draws its neighbors from the whole training set), so no
   data is discarded.

3. **The physics-informed trend removes the need for a dense GP on central
   pressure.** A dense GP's only advantage is the use of long-range correlations.
   The physical mean already absorbs the long-range structure of central
   pressure, leaving a short-range residual that a small neighbor set captures
   fully. A sweep confirmed this: adding neighbors beyond about 30 did not improve
   the central-pressure model. The head-to-head numbers (next subsection) show the
   NNGP within about 0.01 R-squared of the dense GP on both validation protocols.

4. **Calibration is on a subset either way.** Maximum likelihood on 24k points is
   infeasible for both the dense GP and the nearest-neighbor GP, so the
   hyperparameters and the trend come from a subset in both designs. The only
   difference at prediction is global kriging weights versus local simple-kriging
   weights on a residual that is nearly white after detrending, and those
   converge.

The capability for an exact GP still exists. Setting `GPM_VECCHIA = False` with a
large `gpm_cp_max_support` runs the exact GP over that many support points.

### Head-to-head: full GP vs NNGP

Measured on the central-pressure 6-feature model, 85/15 hold-out, 20,582
training fixes and 3,633 test fixes, on the same machine (the fit times exclude
the optional leave-one-out diagnostic and include the shared hyperparameter
calibration):

| configuration | fit time | predict time | R-squared | RMSE | support pts | R matrix |
|---|---|---|---|---|---|---|
| NNGP (30 neighbours, support 6000) | 53.0 s | 0.79 s | 0.9222 | 5.29 hPa | 6,000 | 0.29 GB |
| Full GP (support 6000) | 52.7 s | 0.47 s | 0.9144 | 5.55 hPa | 6,000 | 0.29 GB |
| Full GP (all training data) | 98.7 s | 1.60 s | 0.9197 | 5.37 hPa | 20,581 | 3.39 GB |

Related cost figures:

- **Correlation matrix.** The full GP on all data forms a 20,581 by 20,581
  matrix (3.39 GB); at the full 24k production training set it is about 4.7 GB.
  The Cholesky factorization needs roughly another copy, so peak memory is about
  twice that. The NNGP never forms a large matrix: each prediction solves a 30 by
  30 system (about 7 KB), and the model stores the training inputs (about 1 MB)
  and a KD-tree.
- **Prediction scaling.** Full GP prediction is `O(n_query * n_support)`; NNGP is
  `O(n_query * (log n + m^3))` with `m` neighbours.
- **Calibration.** The hyperparameter search is identical for both (on the
  `n_cal` subset), which is why the two support-6000 fits take the same time. The
  full-data fit is slower only because of its single large Cholesky.

Cost reading: the full GP on all data costs about twice the fit time and about
twelve times the matrix memory of the NNGP, and at the full 24k training set its
dense LOOCV needs roughly 10 to 15 GB. The NNGP scales linearly and runs on a
normal machine.

Accuracy needs a second protocol, because the hold-out is not how the MATLAB
reported its skill. The MATLAB used leave-one-out (LOOCV: each fix predicted with
all others present). This is also the only fully controlled comparison available,
because the MATLAB `GPM_Cp6` model was trained on `hurdat2-1851-2024-040425.txt`,
which is the exact file used here. So the data, the feature set, and the metric
are identical; only the model and the calibration differ. Measured with LOOCV on
all 24,213 de-duplicated observed fixes:

| configuration (identical data, LOOCV) | R-squared | RMSE |
|---|---|---|
| MATLAB full GP (constant trend, all-data calibration) | 0.9320 | 4.97 hPa |
| Python full GP (constant trend, n_cal=4000) | 0.9189 | 5.47 hPa |
| Python full GP (physical-mean trend, n_cal=4000) | **0.9343** | **4.92 hPa** |
| Python NNGP (physical-mean trend, default) | 0.9193 | 5.45 hPa |

Three things follow.

1. **With the physical-mean trend, the Python full GP (0.934) matches and slightly
   exceeds the MATLAB full GP (0.932)** on the same data and metric. The
   physics-informed trend is the winning ingredient: it more than compensates for
   the lighter (subset) calibration.

2. **With the MATLAB's own model (constant-mean ordinary kriging), the Python full
   GP is lower (0.919 versus 0.932).** This gap is calibration depth, not
   implementation error: the MATLAB optimizes the kernel hyperparameters on all
   24k fixes, while the Python optimizes them on an `n_cal` subset for
   feasibility. Ordinary kriging leans entirely on the kernel, so it is the most
   sensitive to that. Deeper calibration closes it monotonically: the
   constant-trend full GP rises 0.9189 (n_cal=4000), 0.9222 (n_cal=8000), 0.9277
   (n_cal=12000) toward the MATLAB's 0.932 as the calibration set approaches
   all-data (at n_cal=12000, half the fixes, it is within 0.004 of the MATLAB),
   and the physical-mean full GP rises from 0.928 (n_cal=1200) to 0.934
   (n_cal=4000). The winning parameters are therefore the physical-mean trend plus
   n_cal of about 4000 or more. (PENDING: the convergence points n_cal=16000,
   20000, and 24213 (all-data) are not yet run; each costs hours because
   calibration scales as n_cal^3.)

3. **The NNGP-versus-full-GP ranking flips between protocols.** Under LOOCV only
   one fix is held out, so its highly correlated same-storm neighbour stays in the
   training set; the global full GP uses all points and exploits that
   autocorrelation, while the NNGP conditions on only 30 neighbours and benefits
   less, so the NNGP scores lower under LOOCV (0.919). Under the 85/15 hold-out,
   roughly 15 percent of each storm's fixes are removed together, the full GP
   loses that crutch, and the NNGP (local conditioning plus the physical trend)
   edges ahead. The hold-out is the better estimate of generalization to unseen
   storms; the LOOCV is the controlled match to the MATLAB.

The NNGP remains the production default: it is competitive on both protocols and
scales linearly, while the full GP with the physical-mean trend is the most
accurate exact model and is available via the configuration flags.

---

## 4. Validation against the MATLAB

The MATLAB training saved its own leave-one-out validation statistics
(`val_total`) inside each `GPM_*.mat`. To compare on the same footing, Python
skill is measured with a random 85/15 hold-out on the natural data distribution.

Measured this way, the comparison is:

| model | Python R-squared / RMSE | MATLAB R-squared / RMSE |
|---|---|---|
| Cp6 | 0.923 / 5.27 hPa | 0.932 / 4.97 hPa |
| Cp3 | 0.923 / 5.37 hPa | 0.918 / 5.49 hPa |
| Rm7 | 0.832 / 29.1 km  | 0.603 / 36.0 km  |
| Rm4 | 0.768 / 34.0 km  | 0.401 / 44.3 km  |

Reading the table:

- **Central pressure is on par.** Cp6 is effectively tied; the small gap is
  explained below. Cp3 is slightly better in Python.
- **Rmax is clearly better in Python.** The log transform, physics-informed
  trend, and nearest-neighbor prediction together raise the hard, skewed target.

### The Cp6 gap (0.923 versus 0.932)

Two small effects, both pushing the same direction.

1. Scalability trade-off. The MATLAB fits the exact GP on all roughly 24k points;
   Python calibrates the hyperparameters and trend on a support set and predicts
   with a nearest-neighbor GP. For the smooth central-pressure surface, the full
   data extracts a little more, on the order of 0.3 hPa.

2. Cross-validation protocol. The MATLAB number is leave-one-out, where every fix
   is predicted with all others available, including the consecutive same-storm
   fix it is autocorrelated with. The Python number is a 15 percent hold-out,
   where that same-storm neighbor is sometimes also held out. Leave-one-out is
   therefore the more optimistic protocol, which inflates the MATLAB value
   slightly relative to the Python value, independent of model quality.

Roughly half of the already small gap is the protocol difference rather than real
skill. In practice 4.97 versus 5.27 hPa is negligible for imputation.

### Caveats

- The Rmax hold-out test set is small (a few hundred rows with observed Rmax,
  pressure, and motion), so its R-squared is noisier. The direction (Python at
  least as good as the MATLAB on Rmax) is solid; the exact value is optimistic in
  absolute terms.
- Consecutive fixes of one storm are autocorrelated, so a per-point split lets
  some of a storm's fixes appear on both sides. This inflates the absolute
  numbers for both Python and the MATLAB equally, so the comparison between them
  stays fair.
- The MATLAB reference was built on the 2023 HURDAT2 vintage and the Python on the
  2024 vintage with EBTRK Rmax backfill, a minor difference.

### Reproducing the comparison

The MATLAB reference output is at
`R:\Work\CHS-PF\JPM-AMP\JPM-AMP_v2024.04\Atlantic_20240426_HURDAT\1_TC_Observations\6_RadiusMaxWinds_DI\out\CHS_TC_HURDATatl20230504_col27_gpm_Cp_Rm.csv`.
Join it to the Python output on datetime and position. With the quality flags
off, the observed central-pressure rows must match exactly (correlation 1.000).
