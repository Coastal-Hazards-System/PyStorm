"""Universal-kriging Gaussian-process metamodel.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Python re-implementation of the Taflanidis-group GP surrogate used by the CHS
HURDAT data-imputation codes (``calibration_GP.m`` / ``sur_model_GP.m``), with
three optional upgrades over the original:

  * universal kriging - a configurable polynomial / physical trend ``b·β`` plus a
    zero-mean GP with an anisotropic **power-exponential** correlation
        R_ij = exp( -Σ_k θ_k |x_ik - x_jk|^p )
    (one shared exponent ``p``, a per-dimension weight θ_k), a **nugget** on the
    diagonal, and standardized inputs/outputs.
  * hyperparameters (θ, p, nugget) by **maximum likelihood** - the concentrated
    negative log-likelihood ``log(σ̂²) + (1/n)·log det R`` minimized with an LHS
    scan polished by **analytic-gradient** L-BFGS-B (multi-start).
  * predictions reproduce ``sur_model_GP``:  f = (b·β + γ·r)·sY + mY.

Trend (physical mean). ``trend_linear`` / ``trend_quad`` select input columns
that enter the kriging trend linearly / quadratically (on standardized inputs).
For central pressure, [vmax, vmax²] gives the GP a wind-pressure mean to model
the residual around - better tail extrapolation than a constant mean.

Scalability - two regimes:
  * ``vecchia=False`` - exact GP on a capped, response-stratified SUPPORT SET
    (``max_support`` points); prediction sums over the support.
  * ``vecchia=True`` (default) - NNGP / nearest-neighbor kriging: hyperparameters
    and the trend β are estimated on the support set, but every prediction
    conditions on its ``n_neighbors`` nearest points among ALL training fixes.
    This uses the whole dataset (not just the support) at O(n·m³) cost and is
    dimension-robust (unlike sparse-Cholesky, which fills in badly in 6-7D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, qr, solve_triangular
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Optional C++ acceleration engine (_gpm). Falls back to NumPy if unavailable.
try:
    from . import _gpm  # type: ignore
    _HAVE_CPP = True
except Exception:  # noqa: BLE001
    _gpm = None
    _HAVE_CPP = False


def have_cpp() -> bool:
    """True if the compiled _gpm engine is available."""
    return _HAVE_CPP


# Hyperparameter bounds (match set_problem_GP defaults).
_BDS_SCALE = (1e-3, 5.0)      # per-dimension correlation weight θ_k
_BDS_EXPON = (0.4, 2.0)       # shared power-exponential exponent p
_BDS_NUGGET = (1e-5, 1.0)     # nugget (relative to process variance)


def _corr(A: np.ndarray, B: np.ndarray, theta: np.ndarray, p: float) -> np.ndarray:
    """Power-exponential correlation between rows of A (m×d) and B (n×d).

    R_ij = exp( -Σ_k θ_k |A_ik - B_jk|^p ). Uses the compiled OpenMP kernel
    (``_gpm.corr``) when available, otherwise a NumPy broadcast.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if _HAVE_CPP:
        return _gpm.corr(A, B, theta, float(p))
    acc = np.zeros((A.shape[0], B.shape[0]))
    for k in range(A.shape[1]):
        d = np.abs(A[:, k][:, None] - B[:, k][None, :])
        acc += theta[k] * d ** p
    return np.exp(-acc)


def _make_basis(Xn: np.ndarray, lin: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Trend basis [1, Xn[:,lin], Xn[:,quad]²] over standardized inputs."""
    cols = [np.ones((Xn.shape[0], 1))]
    if lin.size:
        cols.append(Xn[:, lin])
    if quad.size:
        cols.append(Xn[:, quad] ** 2)
    return np.hstack(cols)


@dataclass
class GPModel:
    """A fitted GP surrogate; ``predict`` mirrors sur_model_GP (+ NNGP option)."""
    mX: np.ndarray
    sX: np.ndarray
    mY: float
    sY: float
    theta: np.ndarray         # per-dim correlation weights (d,)
    p: float                  # shared exponent
    nugget: float
    beta: np.ndarray          # trend coefficients (k,)
    sigma2: float             # process variance (normalized-y units)
    trend_lin: np.ndarray     # input indices entering the trend linearly
    trend_quad: np.ndarray    # input indices entering the trend quadratically
    mode: str = "global"      # "global" (support GP) or "nngp" (all-data NN kriging)
    transform: str = "none"   # response transform: "none", "log", or "sqrt"
    offset_trans: float = 0.0  # offset added before transform to keep positivity
    # global mode
    Xn: Optional[np.ndarray] = None       # support inputs (n×d)
    gamma: Optional[np.ndarray] = None    # correlation coefficients (n,)
    C: Optional[np.ndarray] = None        # Cholesky (variance, optional)
    Ft: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    # nngp mode - all training data
    Xn_all: Optional[np.ndarray] = None   # every training input (N×d), normalized
    resid_all: Optional[np.ndarray] = None  # y_all_norm − B_all·β (N,)
    n_neighbors: int = 30
    # diagnostics
    loocv_r2: float = np.nan
    loocv_rmse: float = np.nan
    _tree_cache: object = field(default=None, repr=False, compare=False)

    # ---- trend basis ----------------------------------------------------
    def _basis(self, Xn: np.ndarray) -> np.ndarray:
        return _make_basis(Xn, self.trend_lin, self.trend_quad)

    def _inv(self, ft: np.ndarray) -> np.ndarray:
        """Back-transform a prediction from emulator space to response space.

        Returns the MEDIAN estimate (inverse of the GP mean); for ``log`` this is
        the geometric-mean estimate, the natural point prediction for a positive,
        right-skewed target like Rmax.
        """
        if self.transform == "log":
            return np.exp(ft) - self.offset_trans
        if self.transform == "sqrt":
            return np.square(ft) - self.offset_trans
        return ft

    def _tree(self) -> cKDTree:
        if self._tree_cache is None:
            # scale by sqrt(theta) so Euclidean nearest-neighbors approximate the
            # anisotropic kernel metric (better-conditioned conditioning sets)
            self._scale = np.sqrt(self.theta)
            self._tree_cache = cKDTree(self.Xn_all * self._scale)
        return self._tree_cache

    # ---- prediction -----------------------------------------------------
    def predict(self, X: np.ndarray, *, return_std: bool = False,
                chunk: int = 2000):
        """Predict the response at rows of X (m×d)."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.mode == "nngp":
            if return_std:
                raise NotImplementedError("return_std is not available in NNGP mode.")
            return self._predict_nngp(X, chunk=chunk)
        return self._predict_global(X, return_std=return_std, chunk=chunk)

    def _predict_global(self, X, *, return_std, chunk):
        if return_std and self.C is None:
            raise ValueError("return_std requires store_variance=True.")
        m = X.shape[0]
        f = np.empty(m)
        std = np.empty(m) if return_std else None
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            xn = (X[s:e] - self.mX) / self.sX
            b = self._basis(xn)
            r = _corr(xn, self.Xn, self.theta, self.p)
            ft = (b @ self.beta + r @ self.gamma) * self.sY + self.mY
            f[s:e] = self._inv(ft)
            if return_std:
                RT = solve_triangular(self.C, r.T, lower=True)
                U = self.Ft.T @ RT - b.T
                V = solve_triangular(self.G, U)
                se = self.sigma2 * (1.0 + np.sum(V ** 2, 0) - np.sum(RT ** 2, 0))
                std[s:e] = np.sqrt(np.clip(se, 0, None)) * self.sY  # emulator-space units
        return (f, std) if return_std else f

    def _predict_nngp(self, X, *, chunk):
        """NNGP: condition each query on its nearest neighbors among all data."""
        tree = self._tree()
        m = min(self.n_neighbors, self.Xn_all.shape[0])
        theta, p, nug = self.theta, self.p, self.nugget
        eye = np.eye(m)
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            xq = (X[s:e] - self.mX) / self.sX                 # (c,d)
            _, idx = tree.query(xq * self._scale, k=m)
            idx = np.atleast_2d(idx)
            Xg = self.Xn_all[idx]                             # (c,m,d)
            rg = self.resid_all[idx]                          # (c,m)
            # local correlation among neighbors
            dgg = np.abs(Xg[:, :, None, :] - Xg[:, None, :, :])  # (c,m,m,d)
            Cgg = np.exp(-np.einsum("k,cijk->cij", theta, dgg ** p))
            Cgg += nug * eye
            # query-to-neighbor correlation
            dq = np.abs(xq[:, None, :] - Xg)                  # (c,m,d)
            cq = np.exp(-np.einsum("k,cik->ci", theta, dq ** p))
            w = np.linalg.solve(Cgg, rg[..., None])[..., 0]   # (c,m)
            rhat = np.einsum("ci,ci->c", cq, w)               # SK residual estimate
            b = self._basis(xq)
            out[s:e] = self._inv((b @ self.beta + rhat) * self.sY + self.mY)
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = dict(mX=self.mX, sX=self.sX, theta=self.theta, beta=self.beta,
                 trend_lin=self.trend_lin, trend_quad=self.trend_quad,
                 mode=np.array(self.mode), transform=np.array(self.transform),
                 offset_trans=self.offset_trans, mY=self.mY, sY=self.sY, p=self.p,
                 nugget=self.nugget, sigma2=self.sigma2,
                 n_neighbors=self.n_neighbors,
                 loocv_r2=self.loocv_r2, loocv_rmse=self.loocv_rmse)
        for k in ("Xn", "gamma", "C", "Ft", "G", "Xn_all", "resid_all"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        np.savez_compressed(path, **d)
        return path

    @classmethod
    def load(cls, path) -> "GPModel":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files}
        kw["mode"] = str(kw["mode"])
        kw["transform"] = str(kw["transform"])
        for s in ("mY", "sY", "p", "nugget", "sigma2", "offset_trans",
                  "loocv_r2", "loocv_rmse"):
            kw[s] = float(kw[s])
        kw["n_neighbors"] = int(kw["n_neighbors"])
        return cls(**kw)


# ---- internals shared by calibration and final build --------------------
def _select_support(y: np.ndarray, m: int, rng) -> np.ndarray:
    """Pick m support indices stratified across the response distribution.

    Half spread uniformly over the sorted response (covering both tails), half a
    random draw (preserving density) - better tail accuracy than random alone.
    """
    n = len(y)
    m = min(m, n)
    n_strat = m // 2
    order = np.argsort(y)
    pos = np.unique(np.linspace(0, n - 1, n_strat).round().astype(int))
    strat = order[pos]
    chosen = set(strat.tolist())
    extra = []
    for i in rng.permutation(n):
        if len(strat) + len(extra) >= m:
            break
        if i not in chosen:
            extra.append(i)
    return np.array(sorted(set(strat.tolist()) | set(extra)), dtype=int)


def _gls(R: np.ndarray, B: np.ndarray, y: np.ndarray):
    """Generalized-least-squares solve. Returns C, Ft, G, beta, rho."""
    C = cholesky(R, lower=True)
    Ft = solve_triangular(C, B, lower=True)
    Q, G = qr(Ft, mode="economic")
    Yt = solve_triangular(C, y, lower=True)
    # least-squares solve of Ft @ beta = Yt (the GLS estimate); robust to a
    # rank-deficient trend basis, where G would be singular
    beta = np.linalg.lstsq(Ft, Yt, rcond=None)[0]
    rho = Yt - Ft @ beta
    return C, Ft, G, beta, rho


def _objective(params, Xn, B, y, *, want_grad=False):
    """Concentrated negative log-likelihood ``log(σ̂²) + (1/n) log det R``.

    With ``want_grad`` returns the closed-form gradient w.r.t. the optimizer
    variables (log10 θ, p, log10 nugget).
    """
    n, d = Xn.shape
    theta = 10.0 ** params[:d]
    p = float(params[d])
    nugget = 10.0 ** params[d + 1]

    R = _corr(Xn, Xn, theta, p)
    R[np.diag_indices_from(R)] += nugget
    try:
        cf = cho_factor(R, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return (1e6, np.zeros(len(params))) if want_grad else 1e6
    C = cf[0]
    logdet = 2.0 * np.sum(np.log(np.abs(np.diag(C))))
    Rinv_B = cho_solve(cf, B, check_finite=False)
    Rinv_y = cho_solve(cf, y, check_finite=False)
    # least-squares (not solve) so a rank-deficient trend basis, e.g. a near
    # collinear feature, degrades gracefully instead of raising Singular matrix
    beta = np.linalg.lstsq(B.T @ Rinv_B, B.T @ Rinv_y, rcond=None)[0]
    r = y - B @ beta
    alpha = cho_solve(cf, r, check_finite=False)
    sigma2 = float(r @ alpha) / n
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return (1e6, np.zeros(len(params))) if want_grad else 1e6
    f = np.log(sigma2) + logdet / n
    if not want_grad:
        return f

    Rinv = cho_solve(cf, np.eye(n), check_finite=False)
    RinvR = Rinv * R
    AAR = np.outer(alpha, alpha) * R
    ln10 = np.log(10.0)
    grad = np.zeros(len(params))
    dSdp = np.zeros((n, n))
    for k in range(d):
        Dk = np.abs(Xn[:, k][:, None] - Xn[:, k][None, :])
        Ak = Dk ** p
        dF = (-np.sum(RinvR * Ak) + np.sum(AAR * Ak) / sigma2) / n
        grad[k] = ln10 * theta[k] * dF
        with np.errstate(divide="ignore", invalid="ignore"):
            lnk = np.where(Dk > 0, np.log(Dk), 0.0)
        dSdp += theta[k] * Ak * lnk
    grad[d] = (-np.sum(RinvR * dSdp) + np.sum(AAR * dSdp) / sigma2) / n
    dFn = (np.trace(Rinv) - float(alpha @ alpha) / sigma2) / n
    grad[d + 1] = ln10 * nugget * dFn
    return f, grad


def _loocv(R, B, y):
    """Leave-one-out CV response (no regression update), per GP_GainCV LOOCV."""
    C, Ft, G, beta, rho = _gls(R, B, y)
    Cinv = solve_triangular(C, np.eye(C.shape[0]), lower=True)
    Rinv = Cinv.T @ Cinv
    gamma = Rinv @ (y - B @ beta)
    err = -gamma / np.diag(Rinv)
    return y + err


def _nngp_loocv(model, chunk: int = 2000):
    """Leave-one-out prediction for the NNGP: each training fix predicted from its
    ``n_neighbors`` nearest OTHER fixes. Returns predictions in response units, or
    None if there are too few points. This is the deployed predictor's own skill,
    not the support-set diagnostic."""
    Xn = model.Xn_all
    n = len(Xn)
    m = min(model.n_neighbors, n - 1)
    if m < 1:
        return None
    tree = model._tree()
    scale = model._scale
    theta, p, nug = model.theta, model.p, model.nugget
    resid = model.resid_all
    eye = np.eye(m)
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xq = Xn[s:e]
        _, idx = tree.query(xq * scale, k=m + 1)
        idx = np.atleast_2d(idx)[:, 1:m + 1]            # drop self (nearest, distance 0)
        Xg = Xn[idx]
        rg = resid[idx]
        Cgg = np.exp(-np.einsum("k,cijk->cij", theta,
                                np.abs(Xg[:, :, None, :] - Xg[:, None, :, :]) ** p)) + nug * eye
        cq = np.exp(-np.einsum("k,cik->ci", theta, np.abs(xq[:, None, :] - Xg) ** p))
        w = np.linalg.solve(Cgg, rg[..., None])[..., 0]
        rhat = np.einsum("ci,ci->c", cq, w)
        b = _make_basis(xq, model.trend_lin, model.trend_quad)
        out[s:e] = model._inv((b @ model.beta + rhat) * model.sY + model.mY)
    return out


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    trend_linear=(),
    trend_quad=(),
    transform: str = "none",
    vecchia: bool = True,
    n_neighbors: int = 30,
    max_support: int = 3000,
    n_cal: int = 1200,
    n_lhs: int = 120,
    n_polish: int = 2,
    maxiter: int = 60,
    optimize_nugget: bool = True,
    seed: int = 0,
    store_variance: bool = False,
    loocv: bool = True,
) -> GPModel:
    """Calibrate and build a universal-kriging GP for inputs X, response y.

    Hyperparameters and the trend β are estimated on a capped support set; with
    ``vecchia`` the predictive model conditions on all training data via nearest
    neighbors (NNGP), otherwise predictions sum over the support set.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    rng = np.random.default_rng(seed)
    trend_lin = np.asarray(trend_linear, dtype=int)
    trend_quad = np.asarray(trend_quad, dtype=int)

    # de-duplicate inputs
    _, uniq = np.unique(X, axis=0, return_index=True)
    uniq = np.sort(uniq)
    X, y = X[uniq], y[uniq]
    n_all, d = X.shape

    # response transform (offset keeps the argument positive), per set_problem_GP
    offset_trans = 0.0
    if transform in ("log", "sqrt"):
        offset_trans = max(0.0, 1e-3 - float(y.min()))
        arg = y + offset_trans
        y_work = np.log(arg) if transform == "log" else np.sqrt(arg)
    else:
        y_work = y

    # global standardization (from all data)
    mX, sX = X.mean(0), X.std(0)
    sX[sX < 1e-12] = 1.0
    mY = float(y_work.mean())
    sY = float(y_work.std())
    if sY < 1e-4:
        sY = 1.0
    Xn_all = (X - mX) / sX
    yn_all = (y_work - mY) / sY

    # support set (capped, response-stratified)
    supp = _select_support(y, max_support, rng) if n_all > max_support else np.arange(n_all)
    Xs, ys = Xn_all[supp], yn_all[supp]
    ns = len(Xs)
    Bs = _make_basis(Xs, trend_lin, trend_quad)

    # calibration subset of the support
    if ns <= n_cal:
        Xc, yc, Bc = Xs, ys, Bs
    else:
        csel = rng.choice(ns, size=n_cal, replace=False)
        Xc, yc, Bc = Xs[csel], ys[csel], Bs[csel]

    # optimizer bounds: log10(θ), p, log10(nugget)
    lo = [np.log10(_BDS_SCALE[0])] * d + [_BDS_EXPON[0]]
    hi = [np.log10(_BDS_SCALE[1])] * d + [_BDS_EXPON[1]]
    if optimize_nugget:
        lo.append(np.log10(_BDS_NUGGET[0])); hi.append(np.log10(_BDS_NUGGET[1]))
    else:
        lo.append(np.log10(1e-6)); hi.append(np.log10(1e-6))
    lo, hi = np.array(lo), np.array(hi)

    # LHS scan (value-only) then analytic-gradient polish
    npar = len(lo)
    cut = (np.arange(n_lhs)[:, None] + rng.random((n_lhs, npar))) / n_lhs
    lhs = lo + (hi - lo) * np.array([rng.permutation(cut[:, j]) for j in range(npar)]).T
    fvals = np.array([_objective(lhs[i], Xc, Bc, yc) for i in range(n_lhs)])
    starts = lhs[np.argsort(fvals)[:max(1, n_polish)]]
    best_x, best_f = starts[0], np.inf
    bnds = list(zip(lo, hi))
    for x0 in starts:
        res = minimize(lambda pp: _objective(pp, Xc, Bc, yc, want_grad=True),
                       x0, jac=True, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": maxiter})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x

    theta = 10.0 ** best_x[:d]
    p = float(best_x[d])
    nugget = float(10.0 ** best_x[d + 1])

    # final build on the support set: β, γ, σ² (+ variance machinery)
    R = _corr(Xs, Xs, theta, p)
    R[np.diag_indices_from(R)] += nugget
    C, Ft, G, beta, rho = _gls(R, Bs, ys)
    gamma = solve_triangular(C, rho, lower=True, trans="T")
    sigma2 = float(rho @ rho) / ns

    # LOOCV (back-transformed to real units), reported as the DEPLOYED predictor's
    # skill: the support-set leave-one-out for the full GP (below), or the
    # nearest-neighbor leave-one-out over all data for the NNGP (after the model
    # is built). Optional: it can be skipped (e.g. for benchmarking).
    loocv_r2 = loocv_rmse = np.nan
    if loocv and not vecchia:
        yhat_t = _loocv(R, Bs, ys) * sY + mY
        if transform == "log":
            yhat = np.exp(yhat_t) - offset_trans
        elif transform == "sqrt":
            yhat = np.square(yhat_t) - offset_trans
        else:
            yhat = yhat_t
        yreal = y[supp]
        ss = np.sum((yreal - yreal.mean()) ** 2)
        loocv_r2 = float(1.0 - np.sum((yreal - yhat) ** 2) / ss) if ss > 0 else np.nan
        loocv_rmse = float(np.sqrt(np.mean((yreal - yhat) ** 2)))

    model = GPModel(
        mX=mX, sX=sX, mY=mY, sY=sY, theta=theta, p=p, nugget=nugget,
        beta=beta, sigma2=sigma2, trend_lin=trend_lin, trend_quad=trend_quad,
        transform=transform, offset_trans=offset_trans,
        Xn=Xs, gamma=gamma,
        C=C if store_variance else None,
        Ft=Ft if store_variance else None,
        G=G if store_variance else None,
        loocv_r2=loocv_r2, loocv_rmse=loocv_rmse,
        n_neighbors=n_neighbors,
    )
    if vecchia:
        # NNGP: keep all data; residual around the (global-β) trend
        B_all = _make_basis(Xn_all, trend_lin, trend_quad)
        model.mode = "nngp"
        model.Xn_all = Xn_all
        model.resid_all = yn_all - B_all @ beta
        if loocv:
            # Deployed-predictor leave-one-out: each fix predicted from its m
            # nearest OTHER fixes (the metric the NNGP actually achieves).
            pred = _nngp_loocv(model)
            if pred is not None:
                yreal = model._inv(yn_all * sY + mY)
                ss = float(np.sum((yreal - yreal.mean()) ** 2))
                model.loocv_r2 = float(1.0 - np.sum((yreal - pred) ** 2) / ss) if ss > 0 else np.nan
                model.loocv_rmse = float(np.sqrt(np.mean((yreal - pred) ** 2)))
    return model
