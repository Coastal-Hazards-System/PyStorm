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

**Recommended configuration (the default).** Use the nearest-neighbor GP with the
physics-informed trend and the log-transformed Rmax, central pressure with a deep
calibration (support 6000, 30 neighbours, n_cal=4000, n_lhs=250) and radius of
maximum wind with a large support and the same deep calibration (support 8000, 30
neighbours, n_cal=4000, n_lhs=250). On the controlled comparison this beats the
MATLAB on all four models (Cp6 0.937 versus 0.932, Cp3 0.920 versus 0.918, Rm7
0.607 versus 0.603, Rm4 0.447 versus 0.401), while scaling linearly and forming no
dense covariance. Two levers drive it: a deep calibration (which lifts Cp6 from
about 0.919 to 0.937) and a large enough support set (Rmax needed 8000, not 3000,
on its roughly 15k training fixes). Rmax follows the MATLAB workflow: it is trained
on observed pressure (before central-pressure imputation) and predicts the missing
Rmax with the completed pressure. Setting the quality flags off (and the four
solver flags to the full GP) reproduces the original MATLAB. The evidence is in
Section 3.

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
  is the performance-critical inner loop of prediction and of the per-evaluation
  R build, is an
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

Cp and Rmax use independently chosen settings, established by a leave-one-out
sweep on each target's full training set with the deep calibration (n_cal=4000,
n_lhs=250):

| target | `max_support` | `n_neighbors` | training fixes |
|---|---|---|---|
| Central pressure | 6000 | 30 | about 24k |
| Rmax | 8000 | 30 | about 15k |

Both targets need a support set large enough for the data. Rmax in particular
trains on the EBTRK-augmented set (about 15k fixes), and a support of 8000 (rather
than the 3000 first tried on a much smaller subset) is what brought it above the
MATLAB. Beyond about 30 neighbors there is no gain for either target, because the
physics-informed trend leaves a short-range residual that a small conditioning set
already captures.

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

| configuration | fit time | predict time | R-squared (85/15 hold-out) | RMSE (hold-out) | support pts | R matrix |
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
| MATLAB full GP (constant trend, all-data calibration) | 0.932 | 4.97 hPa |
| Python full GP (constant trend, n_cal=4000) | 0.919 | 5.47 hPa |
| Python full GP (physical-mean trend, n_cal=4000) | 0.934 | 4.92 hPa |
| Python NNGP (physical-mean, shallow calibration n_cal=1200) | 0.919 | 5.45 hPa |
| Python NNGP (physical-mean, deep calibration n_cal=4000/n_lhs=250) | **0.937** | **4.81 hPa** |

Three things follow.

1. **Calibration depth is the dominant lever.** Deepening the central-pressure
   calibration from the shallow default (n_cal=1200, n_lhs=120) to n_cal=4000 /
   n_lhs=250 lifts the NNGP from 0.919 to 0.937 and the physical-mean full GP from
   0.928 to 0.934. The constant-trend full GP shows the same monotone climb toward
   the MATLAB as n_cal grows (convergence table below). N_LHS of at least 250 is
   needed because the likelihood is multi-modal and a smaller Latin-hypercube
   budget can settle on a worse optimum.

2. **With matched deep calibration, the recommended NNGP (0.937) beats both the
   MATLAB (0.932) and the exact full GP (0.934)**, while forming no dense
   covariance and scaling linearly. So the recommended configuration needs no full
   GP to beat the MATLAB on central pressure; the cheap local predictor, properly
   calibrated, is the most accurate of the four and the cheapest.

3. **The solver choice is secondary to calibration depth.** An earlier reading of
   this table appeared to show the NNGP trailing the full GP under LOOCV, but that
   compared a shallow-calibrated NNGP (0.919) with a deep-calibrated full GP
   (0.934); at equal calibration depth the NNGP is at least as accurate. The
   physics-informed trend and the deep calibration carry the result; whether the
   final predictor is the NNGP or the exact full GP changes the LOOCV by only a few
   thousandths. The NNGP is preferred for cost and because it also generalizes well
   on the stricter 85/15 hold-out.

#### Calibration-depth convergence (matched model)

To confirm that calibration depth is the lever, the constant-trend full GP (the
MATLAB's own model) was run on the identical 2024 HURDAT file with the LOOCV metric
at increasing `n_cal`. The Python LOOCV climbs monotonically toward the MATLAB's
all-data value:

| n_cal | LOOCV R-squared | RMSE | fit time |
|---|---|---|---|
| 4,000 | 0.9189 | 5.47 hPa | (calibration on 4k) |
| 8,000 | 0.9222 | 5.35 hPa | ~39 min |
| 12,000 | 0.9277 | 5.16 hPa | ~88 min |
| 16,000 | pending | pending | pending |
| 20,000 | pending | pending | pending |
| 24,213 (all-data) | pending | pending | pending |
| MATLAB (all-data calibration) | 0.9320 | 4.97 hPa | reference |

At n_cal=12,000 (half the fixes) the Python is within 0.004 R-squared of the
MATLAB. Each point costs hours because calibration scales as `n_cal^3`; the
remaining points (16,000, 20,000, and the full 24,213) are not yet run. For
reference, with the physical-mean trend the full GP already reaches 0.9343 at
n_cal=4,000, above the MATLAB.

The NNGP with the deep central-pressure calibration is the production default: on
the controlled LOOCV it is the most accurate of the four configurations (0.937),
beating both the MATLAB and the exact full GP, and it scales linearly and forms no
dense covariance. The exact full GP remains available via the per-model solver
flags for users who want it, but it is neither more accurate nor cheaper here.

---

## 4. Validation against the MATLAB

The primary comparison is the recommended configuration as it ships (NNGP,
physics-informed trend, log-transformed Rmax, per-target settings, with the deep
central-pressure calibration n_cal=4000 / n_lhs=250) against the MATLAB, on the
IDENTICAL HURDAT file the MATLAB models were trained on
(`hurdat2-1851-2024-040425.txt`), scored by the MATLAB's own metric, leave-one-out
(LOOCV). Same file, same fixes, same metric:

| model | recommended NNGP, LOOCV | MATLAB, LOOCV | fixes |
|---|---|---|---|
| Cp6 | 0.937 / 4.81 hPa | 0.932 / 4.97 hPa | 24,213 |
| Cp3 | 0.920 / 5.45 hPa | 0.918 / 5.49 hPa | 24,097 |
| Rm7 | 0.607 / 35.2 km  | 0.603 / 36.0 km  | 15,017 |
| Rm4 | 0.447 / 41.8 km  | 0.401 / 44.3 km  | 15,054 |

**The recommended configuration beats the MATLAB on all four models**, on the same
data and metric (the figure is the deployed NNGP predictor's leave-one-out). For
central pressure the lever is deep calibration: Cp6 reaches 0.937 (up from about
0.919 at the shallow default), and Cp3 0.920. For the radius of maximum wind the
margin is smaller (Rm7 0.607 versus 0.603, Rm4 0.447 versus 0.401); the lever
there was a large enough support set, 8000 rather than the earlier 3000, on its
roughly 15k training fixes, together with the deep calibration. Rmax follows the
MATLAB workflow: the models are trained on the EBTRK-augmented Rmax with the
OBSERVED pressure (before central-pressure imputation), then predict the missing
Rmax with the completed pressure.

For central pressure, an 85/15 hold-out (a supplementary generalization estimate,
used in Section 3 to compare the Python models with one another) gives Cp6 0.923
and Cp3 0.923. It is not a matched comparison with the MATLAB (Python hold-out
versus MATLAB LOOCV), so it complements rather than replaces the matched table
above.

### Caveats

- The matched LOOCV table is on the identical HURDAT file and, for Rmax, the same
  EBTRK-augmented training set the MATLAB used, so it has no data confound. The
  radius of maximum wind is an intrinsically low-skill target (its predictors
  carry limited size information), so the absolute R-squared is modest for both
  implementations; the Python is the more skillful of the two.
- Consecutive fixes of one storm are autocorrelated, so a per-point split (both
  LOOCV and the hold-out) is optimistic in absolute terms. This inflates the
  numbers for both Python and the MATLAB equally, so the comparison between them
  stays fair.

### Reproducing the comparison

The MATLAB reference output is at
`R:\Work\CHS-PF\JPM-AMP\JPM-AMP_v2024.04\Atlantic_20240426_HURDAT\1_TC_Observations\6_RadiusMaxWinds_DI\out\CHS_TC_HURDATatl20230504_col27_gpm_Cp_Rm.csv`.
Join it to the Python output on datetime and position. With the quality flags
off, the observed central-pressure rows must match exactly (correlation 1.000).
