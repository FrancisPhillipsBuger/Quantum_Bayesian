#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Fast kernel ridge regression primitives used by the model selector:
RBF / Nystrom kernels, GCV ridge path and lightweight predictors.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.linalg.blas import dsyrk

from common.report import log


#%% 1. Low-level kernels and solvers
def scale_gamma(X):

    """

    RBF bandwidth by the sklearn 'scale' rule: 1 / (n_features * Var(X)).

    :param1 X: (n, d) feature matrix.

    :return: scalar gamma.

    """

    n_feat = X.shape[1]
    x_var = float(np.var(X))
    return 1.0 / (n_feat * x_var) if x_var > 0 else 1.0 / n_feat


def sample_landmarks(X, m, random_state):

    """

    Draw m Nystrom landmark rows from X without replacement.

    :param1 X:            (n, d) feature matrix.
    :param2 m:            landmark count (<= n).
    :param3 random_state: RNG seed.

    :return: (m, d) contiguous landmark matrix.

    """

    rng = np.random.RandomState(random_state)
    return np.ascontiguousarray(X[rng.permutation(X.shape[0])[:m]])


def rbf_kernel_inplace(X, Y, gamma):

    """

    RBF kernel K(X, Y) = exp(-gamma * ||x - y||^2) with reused buffers.

    :param1 X:     (n, d) array.
    :param2 Y:     (m, d) array (may be the same object as X).
    :param3 gamma: RBF bandwidth.

    :return: (n, m) kernel matrix.

    """

    X_norm = np.einsum('ij,ij->i', X, X)[:, None]
    if Y is X:
        Y_norm = X_norm.T
    else:
        Y_norm = np.einsum('ij,ij->i', Y, Y)[None, :]
    sq = X @ Y.T
    sq *= -2.0
    sq += X_norm
    sq += Y_norm
    np.maximum(sq, 0.0, out=sq)
    sq *= -float(gamma)
    return np.exp(sq, out=sq)


def whiten_kernel(K_bb, rtol=1e-8):

    """

    Rank-truncated inverse square root of a Nystrom basis kernel.

    :param1 K_bb: (m, m) symmetric PSD basis kernel.
    :param2 rtol: eigenvalues below rtol * max eigenvalue are discarded.

    :return: (m, r) whitening matrix with r <= m.

    """

    w, V = np.linalg.eigh((K_bb + K_bb.T) / 2.0)
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0:
        return np.zeros((K_bb.shape[0], 1))
    keep = w > (w_max * rtol)
    if not np.any(keep):
        keep = np.zeros_like(w, dtype=bool)
        keep[-1] = True
    return V[:, keep] / np.sqrt(w[keep])


def normal_equations(F_or_K, y_c, normalization=None):

    """

    Build the ridge normal equations without forming the whitened design matrix.

    :param1 F_or_K:        (n, m) kernel block K, or design matrix F if normalization is None.
    :param2 y_c:           length-n centred target.
    :param3 normalization: (m, r) whitening matrix, or None.

    :return: (G, b) with G = FᵀF and b = Fᵀy_c.

    """

    A = np.asfortranarray(F_or_K)
    M = dsyrk(1.0, A, trans=1, lower=1)
    M = M + np.tril(M, -1).T          # DSYRK only fills the lower triangle
    v = A.T @ y_c
    if normalization is None:
        return M, v
    return normalization.T @ M @ normalization, normalization.T @ v


def ridge_gcv_path(G, b, ss_total, n, alphas):

    """

    Solve ridge for every alpha with one eigendecomposition; keep the GCV optimum.

    :param1 G:        (m, m) Gram matrix FᵀF.
    :param2 b:        (m,) right-hand side Fᵀy_c.
    :param3 ss_total: y_cᵀy_c.
    :param4 n:        sample count.
    :param5 alphas:   iterable of ridge strengths (>= 0).

    :return: dict {coef, alpha, ss_residual, dof, sigma_gcv_sq, gcv_ok}.

    """

    lam, V = np.linalg.eigh((G + G.T) / 2.0)
    lam = np.maximum(lam, 0.0)
    bt = V.T @ b
    bt2 = bt ** 2

    best = None
    for alpha in alphas:
        a = float(alpha)
        denom = lam + a
        safe = denom > 0
        shrink = np.zeros_like(lam)
        shrink[safe] = 1.0 / denom[safe]

        # SSR = ||y_c||^2 - 2 bᵀw + wᵀGw in the eigenbasis
        ss_residual = float(
            ss_total - np.sum(bt2 * shrink * (lam + 2.0 * a) * shrink)
        )
        ss_residual = max(ss_residual, ss_total * 1e-14)
        dof = float(np.sum(lam * shrink)) + 1.0   # +1 for the intercept
        sigma_gcv, gcv_ok = gcv_sigma_sq(ss_residual, n, dof)
        if best is None or sigma_gcv < best['sigma_gcv_sq']:
            best = {
                'alpha': a, 'ss_residual': ss_residual, 'dof': dof,
                'sigma_gcv_sq': sigma_gcv, 'gcv_ok': gcv_ok,
                'shrink': shrink,
            }

    best['coef'] = V @ (bt * best.pop('shrink'))
    return best


def gcv_sigma_sq(ss_residual, n, df):

    """

    GCV residual variance (SSR / n) / (1 - df / n)^2; charges the fit for its dof.

    :param1 ss_residual: residual sum of squares.
    :param2 n:           sample count.
    :param3 df:          effective degrees of freedom (incl. intercept).

    :return: (sigma_gcv_sq, is_valid); falls back to SSR / n when df >= n.

    """

    mle = ss_residual / n
    denom = 1.0 - (df / n)
    if not np.isfinite(denom) or denom <= 1e-6:
        return mle, False
    return mle / (denom ** 2), True


#%% 2. Lightweight predictor classes
class FastNystroemRBF:

    """

    RBF Nystrom feature map (eigh whitening, in-place kernel, no sklearn overhead).


    """

    __slots__ = ("gamma", "components_", "normalization_")

    def __init__(self, gamma, components_, normalization_):

        """

        Store the fitted Nystrom map.

        :param1 gamma:          RBF bandwidth.
        :param2 components_:    (m, d) landmarks.
        :param3 normalization_: (m, r) whitening matrix.

        :return: None.

        """

        self.gamma = gamma
        self.components_ = components_
        self.normalization_ = normalization_

    @classmethod
    def fit_kernel(cls, X, gamma, n_components, random_state):

        """

        Sample landmarks, whiten K_bb and return the raw kernel block.

        :param1 X:            (n, d) array.
        :param2 gamma:        RBF bandwidth.
        :param3 n_components: landmark count.
        :param4 random_state: RNG seed.

        :return: (FastNystroemRBF, (n, m) kernel block K_xb).

        """

        n_components = max(1, min(int(n_components), X.shape[0]))
        basis = sample_landmarks(X, n_components, random_state)
        normalization = whiten_kernel(rbf_kernel_inplace(basis, basis, gamma))
        return cls(gamma, basis, normalization), rbf_kernel_inplace(X, basis, gamma)

    def transform(self, X):

        """

        Map X to Nystrom features.

        :param1 X: (n, d) feature matrix.

        :return: (n, r) feature matrix.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        K = rbf_kernel_inplace(X, self.components_, self.gamma)
        return K @ self.normalization_


class FastNystroemRidge:

    """

    Nystrom + ridge predictor (or pure linear ridge).


    """

    __slots__ = ("nystroem", "coef_", "intercept_", "linear")

    def __init__(self, nystroem, coef_, intercept_, linear=False):

        """

        Store the fitted feature map and ridge coefficients.

        :param1 nystroem:   fitted FastNystroemRBF (None for linear).
        :param2 coef_:      ridge coefficient vector.
        :param3 intercept_: intercept (target mean).
        :param4 linear:     predict on raw X instead of Nystrom features.

        :return: None.

        """

        self.nystroem = nystroem
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.linear = linear

    def predict(self, X):

        """

        Predict targets for X.

        :param1 X: (n, d) feature matrix.

        :return: length-n prediction vector.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        if self.linear or self.nystroem is None:
            features = X
        else:
            features = self.nystroem.transform(X)
        return features @ self.coef_ + self.intercept_


class FastKernelRidge:

    """

    Full-kernel RBF ridge predictor (no Nystrom).


    """

    __slots__ = ("X_fit", "gamma", "dual", "intercept_")

    def __init__(self, X_fit, gamma, dual, intercept_):

        """

        Store the training data and dual coefficients.

        :param1 X_fit:      (n, d) training features.
        :param2 gamma:      RBF bandwidth.
        :param3 dual:       length-n dual coefficients.
        :param4 intercept_: intercept (target mean).

        :return: None.

        """

        self.X_fit = X_fit
        self.gamma = gamma
        self.dual = dual
        self.intercept_ = intercept_

    def predict(self, X):

        """

        Predict targets for X against the stored training kernel.

        :param1 X: (n, d) feature matrix.

        :return: length-n prediction vector.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        K = rbf_kernel_inplace(X, self.X_fit, self.gamma)
        return K @ self.dual + self.intercept_


#%% 3. Shared regressor scaffolding
def select_subset(X, feature_indices):

    """

    Extract the requested columns; reject a sub-matrix whose columns are all constant.

    :param1 X:               (n, d) feature matrix.
    :param2 feature_indices: column indices, or None for all.

    :return: (n, k) contiguous float64 sub-matrix, or None when degenerate.

    """

    if feature_indices is not None:
        X_subset = np.ascontiguousarray(X[:, list(feature_indices)], dtype=np.float64)
    else:
        X_subset = np.ascontiguousarray(X, dtype=np.float64)
    if not np.any(np.ptp(X_subset, axis=0) > 1e-10):
        return None
    return X_subset


def fit_statistics(ss_residual, ss_total, n, p):

    """

    Goodness-of-fit scalars reported by every regressor backend.

    :param1 ss_residual: residual sum of squares.
    :param2 ss_total:    total (centred) sum of squares.
    :param3 n:           sample count.
    :param4 p:           effective parameter count.

    :return: dict {mse, sigma_ml_sq, r_squared, adj_r_squared}.

    """

    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    return {
        'mse': ss_residual / max(n - p, 1),
        'sigma_ml_sq': ss_residual / n,
        'r_squared': r_squared,
        'adj_r_squared': 1 - ((1 - r_squared) * (n - 1) / max(n - p - 1, 1)),
    }


class BaseRidgeRegression:

    """

    Ridge-strength bookkeeping shared by the kernel-ridge backends.


    """

    def __init__(self, alpha=1.0, alpha_grid=None):

        """

        :param1 alpha:      L2 regularisation strength.
        :param2 alpha_grid: optional ridge grid scanned by GCV per subset; None → [alpha].

        :return: None.

        """

        self.alpha = alpha
        self.alpha_grid = (
            None if alpha_grid is None
            else [float(a) for a in np.atleast_1d(alpha_grid)]
        )
        self.model = None
        self.feature_indices = None

    def alphas(self):

        """Return the ridge strengths to scan."""

        return self.alpha_grid if self.alpha_grid else [float(self.alpha)]

    def take_subset(self, X, feature_indices):

        """

        Record the fitted subset and return its validated sub-matrix.

        :param1 X:               (n, d) feature matrix.
        :param2 feature_indices: column indices, or None.

        :return: (n, k) sub-matrix, or None when degenerate.

        """

        self.feature_indices = feature_indices
        return select_subset(X, feature_indices)


#%% 4. RBF kernel ridge regression
class KernelRidgeRegression(BaseRidgeRegression):

    """

    Kernel ridge regression with an RBF kernel (full, Nystrom or linear).


    """

    def __init__(self, alpha=1.0, gamma=None, full_kernel_gcv_max_n=4000,
                 alpha_grid=None):

        """

        :param1 alpha:                 L2 regularisation strength.
        :param2 gamma:                 RBF bandwidth; None → scale_gamma per subset.
        :param3 full_kernel_gcv_max_n: largest n for the O(n^3) GCV trace on the full-kernel path.
        :param4 alpha_grid:            optional ridge grid; GCV-best alpha per subset.

        :return: None.

        """

        super().__init__(alpha=alpha, alpha_grid=alpha_grid)
        self.gamma = gamma
        self.full_kernel_gcv_max_n = int(full_kernel_gcv_max_n)

    def fit(self, X, y, feature_indices=None, use_nystroem=False,
            nystroem_components=100, nystroem_random_state=0, linear=False,
            need_predictions=True):

        """

        Fit a KRR model on the requested feature subset.

        :param1 X:                     (n, d) feature matrix.
        :param2 y:                     length-n target vector.
        :param3 feature_indices:       optional column indices.
        :param4 use_nystroem:          enable the Nystrom approximation.
        :param5 nystroem_components:   Nystrom landmark count.
        :param6 nystroem_random_state: RNG seed.
        :param7 linear:                fit a pure linear ridge (no kernel).
        :param8 need_predictions:      materialise y_pred / residuals (the subset search does not).

        :return: dict with model + diagnostics, or None on failure.

        """

        try:
            X_subset = self.take_subset(X, feature_indices)
            if X_subset is None:
                return None

            if linear:
                use_nystroem = False

            gamma = float(self.gamma) if self.gamma is not None else scale_gamma(X_subset)

            y = np.asarray(y, dtype=np.float64).ravel()
            n = y.shape[0]
            y_mean = y.mean()
            y_c = y - y_mean

            if linear:
                design, normalization = X_subset, None
                nystroem = None
                n_components = None
            elif use_nystroem:
                if nystroem_components is None:
                    n_components = min(100, X_subset.shape[0])
                else:
                    n_components = min(int(nystroem_components), X_subset.shape[0])
                if n_components < 1:
                    return None
                nystroem, design = FastNystroemRBF.fit_kernel(
                    X_subset, gamma, n_components, nystroem_random_state
                )
                normalization = nystroem.normalization_
            else:
                # Full dual KRR (alpha I + K) c = y_c; keeps the single configured alpha
                full_alpha = float(self.alpha)
                K = rbf_kernel_inplace(X_subset, X_subset, gamma)
                K.flat[::K.shape[0] + 1] += full_alpha
                try:
                    c, lower = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
                    dual = cho_solve((c, lower), y_c, check_finite=False)
                except Exception:
                    dual = np.linalg.solve(K, y_c)

                self.model = FastKernelRidge(X_subset, gamma, dual, y_mean)
                y_pred = self.model.predict(X_subset)
                residuals = y - y_pred
                p = X_subset.shape[1]
                ss_residual = float(residuals @ residuals)
                sigma_gcv_sq, gcv_ok = ss_residual / n, False
                if n <= self.full_kernel_gcv_max_n:
                    try:
                        L_inv = solve_triangular(
                            c, np.eye(n), lower=lower, check_finite=False,
                        )
                        df = n - full_alpha * float(np.sum(L_inv ** 2)) + 1.0
                        sigma_gcv_sq, gcv_ok = gcv_sigma_sq(ss_residual, n, df)
                    except Exception:
                        pass
                return {
                    'model': self.model,
                    'y_pred': y_pred,
                    'residuals': residuals,
                    **fit_statistics(ss_residual, float(y_c @ y_c), n, p),
                    'sigma_gcv_sq': sigma_gcv_sq,
                    'sigma_estimator': 'gcv' if gcv_ok else 'in-sample',
                    'alpha': full_alpha,
                    'gamma': gamma,
                    'kernel': 'rbf',
                    'n_features': p,
                    'linear': linear,
                    'use_nystroem': False,
                    'nystroem_components': None,
                }

            ss_total = float(y_c @ y_c)
            G, b = normal_equations(design, y_c, normalization=normalization)
            best = ridge_gcv_path(G, b, ss_total, n, self.alphas())
            coef = best['coef']

            self.model = FastNystroemRidge(
                nystroem=nystroem,
                coef_=coef,
                intercept_=y_mean,
                linear=linear,
            )

            if need_predictions:
                y_pred = design @ coef + y_mean if normalization is None \
                    else (design @ normalization) @ coef + y_mean
                residuals = y - y_pred
            else:
                y_pred = None
                residuals = None

            p = (normalization.shape[1] if normalization is not None
                 else X_subset.shape[1])
            ss_residual = best['ss_residual']

            return {
                'model': self.model,
                'y_pred': y_pred,
                'residuals': residuals,
                **fit_statistics(ss_residual, ss_total, n, p),
                'sigma_gcv_sq': best['sigma_gcv_sq'],
                'sigma_estimator': 'gcv' if best['gcv_ok'] else 'in-sample',
                'dof': best['dof'],
                'alpha': best['alpha'],
                'gamma': gamma,
                'kernel': 'rbf',
                'n_features': p,
                'linear': linear,
                'use_nystroem': use_nystroem,
                'nystroem_components': n_components if use_nystroem else None,
                'nystroem_rank': (
                    int(normalization.shape[1]) if normalization is not None else None
                ),
            }

        except Exception as e:
            log("Warning", f"KRR fit failed: {e}")
            return None
