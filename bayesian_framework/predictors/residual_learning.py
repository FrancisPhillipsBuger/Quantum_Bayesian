#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tier-2 residual learning on top of the Tier-1 ensemble prediction: exact GP,
Nystrom + BayesianRidge, or exact linear ridge with GCV alpha.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from common.kernel_ridge import (normal_equations, ridge_gcv_path, sample_landmarks,
                                 scale_gamma, whiten_kernel)
from common.report import log


#%% 1. Solvers and feature-map helpers
class LinearRidgeGCV:

    """

    Exact linear ridge E = w . Phi(A) with GCV-chosen alpha; the O(n d^2) equivalent
    of a DotProduct GP posterior mean, used for large n.


    """

    __slots__ = ("coef_", "intercept_", "alpha_", "dof_", "sigma_", "mean_")

    def __init__(self):

        """Create an unfitted linear ridge."""

        self.coef_ = None
        self.intercept_ = 0.0
        self.alpha_ = None
        self.dof_ = None
        self.sigma_ = None
        self.mean_ = None

    def fit(self, X, y, alphas):

        """

        Fit the ridge, selecting alpha by GCV.

        :param1 X:      (n, d) feature matrix.
        :param2 y:      length-n target.
        :param3 alphas: ridge strengths to scan.

        :return: self.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = y.shape[0]
        self.intercept_ = float(y.mean())
        y_c = y - self.intercept_

        G, b = normal_equations(X, y_c)
        best = ridge_gcv_path(G, b, float(y_c @ y_c), n, list(alphas))
        self.coef_ = best['coef']
        self.alpha_ = best['alpha']
        self.dof_ = best['dof']
        self.sigma_ = float(np.sqrt(max(best['sigma_gcv_sq'], 0.0)))
        return self

    def predict(self, X, return_std=False):

        """

        Predict targets and optionally the homoscedastic std.

        :param1 X:          (n, d) feature matrix.
        :param2 return_std: also return the predictive std.

        :return: predictions, or (predictions, std).

        """

        if self.coef_ is None:
            raise ValueError("LinearRidgeGCV is not fitted")
        pred = np.ascontiguousarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
        if return_std:
            return pred, np.full(pred.shape[0], self.sigma_)
        return pred


class CappedLBFGSB:

    """

    Picklable L-BFGS-B optimiser for GaussianProcessRegressor, capped at max_iter.


    """

    __slots__ = ("max_iter",)

    def __init__(self, max_iter):

        """

        :param1 max_iter: iteration cap (>= 1).

        :return: None.

        """

        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1 when provided")
        self.max_iter = max_iter

    def __call__(self, obj_func, initial_theta, bounds):

        """Run capped L-BFGS-B on the GP marginal-likelihood objective."""

        result = minimize(
            obj_func, initial_theta, method="L-BFGS-B", jac=True,
            bounds=bounds, options={"maxiter": self.max_iter},
        )
        return result.x, result.fun


class KernelNystroem:

    """

    Nystrom map x -> k(x, B) K_BB^{-1/2} for any vectorised kernel k(A, B)
    (Matern or fidelity quantum kernels stay vectorised, unlike sklearn's Nystroem).


    """

    __slots__ = ("kernel_fn", "components_", "normalization_")

    def __init__(self, kernel_fn, components_, normalization_):

        """

        :param1 kernel_fn:      vectorised callable k(A, B) -> Gram matrix.
        :param2 components_:    (m, d) landmarks.
        :param3 normalization_: (m, r) whitening matrix.

        :return: None.

        """

        self.kernel_fn = kernel_fn
        self.components_ = components_
        self.normalization_ = normalization_

    @classmethod
    def fit_and_transform(cls, X, kernel_fn, n_components, random_state=None):

        """

        Sample landmarks, whiten K_BB and map X.

        :param1 X:            (n, d) feature matrix.
        :param2 kernel_fn:    vectorised kernel callable.
        :param3 n_components: landmark count (capped at n).
        :param4 random_state: RNG seed.

        :return: (KernelNystroem, (n, r) feature matrix).

        """

        m = max(1, min(int(n_components), X.shape[0]))
        basis = sample_landmarks(X, m, random_state)
        normalization = whiten_kernel(
            np.asarray(kernel_fn(basis, basis), dtype=np.float64)
        )
        instance = cls(kernel_fn, basis, normalization)
        return instance, instance.transform(X)

    def transform(self, X):

        """

        Map X against the stored landmarks.

        :param1 X: (n, d) feature matrix.

        :return: (n, r) feature matrix.

        """

        K_xb = np.asarray(
            self.kernel_fn(np.ascontiguousarray(X, dtype=np.float64),
                           self.components_),
            dtype=np.float64,
        )
        return K_xb @ self.normalization_


#%% 2. GP backbone
class ClassicalGaussianProcess:

    """

    GP backbone with three solvers, recorded in effective_backend_: 'exact-gp'
    (O(n^3), n <= exact_gp_max_samples), 'nystroem' (same kernel + BayesianRidge)
    and 'linear-ridge' (linear kernel, large n).


    """

    VALID_KERNEL_TYPES = {"linear", "rbf", "matern"}

    # Exact-GP settings that the Nystrom + BayesianRidge path cannot honour
    NYSTROEM_INACTIVE_PARAMS = ("sigma_noise", "kernel_variance",
                                "optimize", "n_restarts")

    def __init__(self, length_scale=1.0, kernel_variance=1.0, sigma_noise=0.1,
                 random_state=None, use_nystroem=True,
                 nystroem_components=512, linear=False, kernel_type="rbf",
                 nu=2.5, exact_gp_max_samples=2000, linear_alphas=None):

        """

        :param1 length_scale:          kernel length scale.
        :param2 kernel_variance:       ConstantKernel amplitude.
        :param3 sigma_noise:           observation noise (std).
        :param4 random_state:          RNG seed.
        :param5 use_nystroem:          use the Nystrom solver.
        :param6 nystroem_components:   Nystrom landmark count.
        :param7 linear:                shortcut for kernel_type='linear'.
        :param8 kernel_type:           'linear' | 'rbf' | 'matern'.
        :param9 nu:                    Matern smoothness (0.5, 1.5, 2.5 or inf).
        :param10 exact_gp_max_samples: largest n for the exact GP; above it the Nystrom
                                       (or linear-ridge) solver is used.
        :param11 linear_alphas:        ridge grid of the linear path; None → logspace(-4, 6, 21).

        :return: None.

        """

        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if kernel_variance <= 0:
            raise ValueError("kernel_variance must be positive")
        if sigma_noise < 0:
            raise ValueError("sigma_noise must be non-negative")
        if nu not in {0.5, 1.5, 2.5} and not (nu == float("inf")):
            raise ValueError("nu must be 0.5, 1.5, 2.5, or inf")

        if not use_nystroem:
            if length_scale < 1e-10 or length_scale > 1e5:
                warnings.warn(
                    f"length_scale={length_scale} outside optimisation bounds [1e-10, 1e5]."
                )
        if kernel_variance < 1e-5 or kernel_variance > 1e5:
            warnings.warn(
                f"kernel_variance={kernel_variance} outside optimisation bounds [1e-5, 1e5]."
            )

        self.length_scale = float(length_scale)
        self.kernel_variance = float(kernel_variance)
        self.sigma_noise = float(sigma_noise)
        self.random_state = random_state
        self.use_nystroem = bool(use_nystroem)
        self.nystroem_components = nystroem_components
        self.linear = bool(linear)
        self.kernel_type = "linear" if self.linear else str(kernel_type).lower()
        if self.kernel_type not in self.VALID_KERNEL_TYPES:
            raise ValueError(f"kernel_type must be one of: {sorted(self.VALID_KERNEL_TYPES)}")
        if self.kernel_type == "linear":
            self.use_nystroem = False
        self.nu = float(nu)
        self.exact_gp_max_samples = int(exact_gp_max_samples)
        self.linear_alphas = (
            np.logspace(-4, 6, 21) if linear_alphas is None
            else np.asarray(linear_alphas, dtype=np.float64)
        )

        self.gp = None
        self.feature_map = None
        self.X_features_train_ = None      # cached training features
        self.effective_backend_ = None     # solver actually used by fit()
        self.effective_length_scale_ = None

    #%% 2.1 Kernel and optimiser construction
    def kernel_label(self):

        """Return the configured kernel's name."""

        if self.kernel_type == "matern":
            return f"matern(nu={self.nu:g})"
        return self.kernel_type

    def build_kernel(self):

        """Return the sklearn GP kernel for the configured kernel_type."""

        if self.kernel_type == "linear":
            base_kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
        elif self.kernel_type == "matern":
            base_kernel = Matern(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-10, 1e5),
                nu=self.nu,
            )
        else:
            base_kernel = RBF(self.length_scale, length_scale_bounds=(0.01, 100.0))
        return ConstantKernel(self.kernel_variance, constant_value_bounds=(1e-5, 1e5)) * base_kernel

    def build_gp(self, optimize=True, kernel=None, n_restarts=0, max_iter=100):

        """

        Build an unfitted GaussianProcessRegressor.

        :param1 optimize:   enable hyper-parameter optimisation.
        :param2 kernel:     prebuilt kernel; None → build_kernel().
        :param3 n_restarts: optimiser restart count.
        :param4 max_iter:   optimiser iteration cap; None → sklearn default.

        :return: GaussianProcessRegressor.

        """

        if kernel is None:
            kernel = self.build_kernel()
        if not optimize:
            optimizer = None
        elif max_iter is None:
            optimizer = "fmin_l_bfgs_b"
        else:
            optimizer = CappedLBFGSB(max_iter)
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(self.sigma_noise ** 2, 1e-10),
            normalize_y=True,
            n_restarts_optimizer=n_restarts if optimize else 0,
            random_state=self.random_state,
            optimizer=optimizer,
        )

    #%% 2.2 Feature map
    def build_nystroem_kernel_fn(self, X):

        """

        Kernel callable for the Nystrom map, with the 'scale' length-scale heuristic.

        :param1 X: (n, d) training features.

        :return: (callable k(A, B), label).

        """

        self.effective_length_scale_ = float(np.sqrt(1.0 / (2.0 * scale_gamma(X))))
        self.length_scale = self.effective_length_scale_
        if self.kernel_type == "matern":
            kernel = Matern(length_scale=self.effective_length_scale_, nu=self.nu)
            label = f"matern(nu={self.nu:g}, ls={self.effective_length_scale_:.4g})"
        else:
            kernel = RBF(length_scale=self.effective_length_scale_)
            label = f"rbf(ls={self.effective_length_scale_:.4g})"
        return kernel, label

    def fit_feature_map(self, X, show_progress=True):

        """

        Fit the Nystrom map when enabled.

        :param1 X:             (n, d) feature matrix.
        :param2 show_progress: print status lines.

        :return: mapped (n, m) features, or X unchanged.

        """

        if not self.use_nystroem:
            self.feature_map = None
            return X

        n_samples = X.shape[0]
        n_requested = int(self.nystroem_components if self.nystroem_components is not None else 512)
        n_components = max(1, min(n_requested, n_samples))

        if show_progress and n_components < n_requested:
            log("Warning", f"Nystrom landmarks capped at n_samples ({n_components} of {n_requested})")

        kernel_fn, label = self.build_nystroem_kernel_fn(X)
        if show_progress:
            log("Tier-2", f"Nystrom feature map: {label}, {n_components} landmarks")

        self.feature_map, F = KernelNystroem.fit_and_transform(
            X, kernel_fn, n_components, random_state=self.random_state,
        )
        self.nystroem_components = n_components
        return F

    def transform_features(self, X):

        """Map X through the fitted feature map (identity when off)."""

        if self.feature_map is None:
            return X
        return self.feature_map.transform(X)

    def sync_kernel_params(self):

        """Copy optimised kernel hyper-parameters back onto this instance."""

        if self.gp is None:
            return
        kernel = self.gp.kernel_
        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            if isinstance(kernel.k1, ConstantKernel):
                self.kernel_variance = float(kernel.k1.constant_value)
            elif isinstance(kernel.k2, ConstantKernel):
                self.kernel_variance = float(kernel.k2.constant_value)
            for side in (kernel.k1, kernel.k2):
                if isinstance(side, (RBF, Matern)):
                    self.length_scale = float(side.length_scale)
                    break

    #%% 2.3 Fit / predict
    def fit(self, X, y, optimize=True, n_restarts=0, max_iter=100,
            show_progress=True, optimize_subset_size=None):

        """

        Fit on (X, y) with the solver implied by kernel_type, n and use_nystroem.

        :param1 X:                    (n, d) feature matrix.
        :param2 y:                    length-n target.
        :param3 optimize:             tune kernel hyper-parameters (exact GP).
        :param4 n_restarts:           L-BFGS-B restart count.
        :param5 max_iter:             optimiser iteration cap.
        :param6 show_progress:        print status lines.
        :param7 optimize_subset_size: hyper-parameter subsample; None → min(800, n).

        :return: self.

        """

        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples = X.shape[0]

        if self.kernel_type == "linear":
            use_map = False
            if n_samples > self.exact_gp_max_samples:
                self.use_nystroem = False
                self.feature_map = None
                self.X_features_train_ = X
                self.effective_backend_ = "linear-ridge"
                self.gp = LinearRidgeGCV().fit(X, y, self.linear_alphas)
                if show_progress:
                    log("Tier-2", f"Solver: exact linear ridge, GCV alpha {self.gp.alpha_:.4g} "
                                  f"of {len(self.linear_alphas)}, dof {self.gp.dof_:.1f}")
                return self
        elif self.use_nystroem:
            use_map = True
        elif n_samples > self.exact_gp_max_samples:
            use_map = True
            if show_progress:
                log("Tier-2", f"n = {n_samples} exceeds exact_gp_max_samples = "
                              f"{self.exact_gp_max_samples}; using the Nystrom solver")
        else:
            use_map = False

        self.use_nystroem = use_map
        if use_map and self.nystroem_components is None:
            self.nystroem_components = 1024

        X_features = self.fit_feature_map(X, show_progress=show_progress)
        self.X_features_train_ = X_features

        if self.use_nystroem:
            self.effective_backend_ = "nystroem"
            if show_progress:
                log("Tier-2", f"Solver: Nystrom + BayesianRidge (inactive: "
                              f"{', '.join(self.NYSTROEM_INACTIVE_PARAMS)})")
            br_max_iter = 300 if max_iter is None else max(1, int(max_iter))
            self.gp = BayesianRidge(max_iter=br_max_iter, fit_intercept=True)
            self.gp.fit(X_features, y)
            return self

        self.effective_backend_ = "exact-gp"
        if show_progress:
            log("Tier-2", f"Solver: exact GP ({self.kernel_label()})")

        if optimize_subset_size is None:
            optimize_subset_size = min(800, n_samples)

        if optimize and optimize_subset_size < n_samples:
            # Tune hyper-parameters on a subsample, then refit on all samples
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_samples, size=optimize_subset_size, replace=False)
            if show_progress:
                log("Tier-2", f"Hyper-parameters tuned on {optimize_subset_size} of "
                              f"{n_samples} samples")
            tmp_gp = self.build_gp(optimize=True, n_restarts=n_restarts, max_iter=max_iter)
            tmp_gp.fit(X_features[idx], y[idx])
            self.gp = self.build_gp(
                optimize=False, kernel=tmp_gp.kernel_, n_restarts=0, max_iter=max_iter,
            )
        else:
            self.gp = self.build_gp(
                optimize=optimize,
                n_restarts=n_restarts if optimize else 0,
                max_iter=max_iter,
            )
        self.gp.fit(X_features, y)
        self.X_features_train_ = X_features

        self.sync_kernel_params()
        return self

    def describe_backend(self):

        """Return a label for the solver fit() actually used."""

        if self.effective_backend_ is None:
            return "unfitted"
        detail = self.kernel_label()
        if self.effective_backend_ == "nystroem":
            return (f"{detail} Nystrom({self.nystroem_components}) "
                    f"+ BayesianRidge")
        if self.effective_backend_ == "linear-ridge":
            return f"exact linear ridge (GCV alpha={self.gp.alpha_:.4g})"
        return f"exact GP ({detail})"

    def predict(self, X, return_std=True):

        """

        Predict targets (and optionally std) for new samples.

        :param1 X:          (n, d) feature matrix.
        :param2 return_std: also return the predictive std.

        :return: predictions, or (predictions, std).

        """

        if self.gp is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        return self.gp.predict(self.transform_features(X), return_std=return_std)

    def predict_train(self, return_std=True):

        """

        Predict on the cached training features.

        :param1 return_std: also return the predictive std.

        :return: training-set predictions, or (predictions, std).

        """

        if self.gp is None or self.X_features_train_ is None:
            raise ValueError("Model not fitted or features not cached.")
        return self.gp.predict(self.X_features_train_, return_std=return_std)


#%% 3. Residual learner
class ResidualLearning:

    """

    GP residual learner; subclasses swap the backbone via the hooks in section 3.2.


    """

    PREFERRED_RESIDUAL_DIM = None       # Tier-2 projection width; None → no projection
    LABEL = "GP residual"

    def __init__(self, sigma_noise=0.01, optimize=True,
                 n_restarts=1, max_iter=100, optimize_subset_size=None,
                 random_state=None, length_scale=1.0, kernel_variance=1.0,
                 use_nystroem=False, nystroem_components=512, linear=False,
                 kernel_type="matern", nu=2.5, exact_gp_max_samples=2000):

        """

        :param1 sigma_noise:           GP observation noise.
        :param2 optimize:              tune kernel hyper-parameters.
        :param3 n_restarts:            optimiser restart count.
        :param4 max_iter:              optimiser iteration cap.
        :param5 optimize_subset_size:  hyper-parameter subsample; None → auto.
        :param6 random_state:          RNG seed.
        :param7 length_scale:          kernel length scale.
        :param8 kernel_variance:       ConstantKernel amplitude.
        :param9 use_nystroem:          use the Nystrom solver.
        :param10 nystroem_components:  Nystrom landmark count.
        :param11 linear:               shortcut for kernel_type='linear'.
        :param12 kernel_type:          'linear' | 'rbf' | 'matern'.
        :param13 nu:                   Matern smoothness.
        :param14 exact_gp_max_samples: largest n for the exact GP.

        :return: None.

        """

        self.sigma_noise = sigma_noise
        self.optimize = optimize
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.optimize_subset_size = optimize_subset_size
        self.random_state = random_state
        self.length_scale = length_scale
        self.kernel_variance = kernel_variance
        self.use_nystroem = use_nystroem
        self.nystroem_components = nystroem_components
        self.linear = bool(linear)
        self.kernel_type = "linear" if self.linear else str(kernel_type).lower()
        if self.kernel_type not in ClassicalGaussianProcess.VALID_KERNEL_TYPES:
            raise ValueError(
                f"kernel_type must be one of: {sorted(ClassicalGaussianProcess.VALID_KERNEL_TYPES)}"
            )
        if self.kernel_type == "linear":
            self.use_nystroem = False
        self.nu = float(nu)
        self.exact_gp_max_samples = int(exact_gp_max_samples)
        self.gp = None
        self.feature_indices = None
        self.baseline_std_ = None
        self.baseline_std_in_sample_ = None
        self.residual_target_source_ = None
        self.n_input_features_ = None
        self.feature_scaler_ = None

    #%% 3.1 Shared steps
    def shared_kwargs(self):

        """

        Constructor kwargs, used to clone this learner as a quantum subclass.

        :return: dict of constructor kwargs.

        """

        return dict(
            sigma_noise=self.sigma_noise,
            optimize=self.optimize,
            n_restarts=self.n_restarts,
            max_iter=self.max_iter,
            optimize_subset_size=self.optimize_subset_size,
            random_state=self.random_state,
            length_scale=self.length_scale,
            kernel_variance=self.kernel_variance,
            use_nystroem=self.use_nystroem,
            nystroem_components=self.nystroem_components,
            linear=self.linear,
            kernel_type=self.kernel_type,
            nu=self.nu,
            exact_gp_max_samples=self.exact_gp_max_samples,
        )

    def prepare_residuals(self, y, y_pred, y_pred_oof=None, show_progress=True):

        """

        Residual target for Tier 2; out-of-fold Tier-1 predictions avoid the
        downward bias of in-sample residuals.

        :param1 y:             length-n true target.
        :param2 y_pred:        length-n in-sample Tier-1 prediction.
        :param3 y_pred_oof:    optional length-n out-of-fold Tier-1 prediction.
        :param4 show_progress: print status lines.

        :return: (residuals, baseline_mse).

        """

        y = np.asarray(y).ravel()
        in_sample = y - np.asarray(y_pred).ravel()
        self.baseline_std_in_sample_ = float(np.sqrt(np.mean(in_sample ** 2)))

        if y_pred_oof is None:
            residuals = in_sample
            self.residual_target_source_ = "in-sample"
        else:
            residuals = y - np.asarray(y_pred_oof).ravel()
            self.residual_target_source_ = "out-of-fold"

        baseline_mse = float(np.mean(residuals ** 2))
        self.baseline_std_ = float(np.sqrt(baseline_mse))
        if show_progress:
            extra = ("" if y_pred_oof is None else
                     f" (in-sample {self.baseline_std_in_sample_:.4f} eV)")
            log("Tier-2", f"Residual target ({self.residual_target_source_}): "
                          f"RMSE {self.baseline_std_:.4f} eV{extra}")
        return residuals, baseline_mse

    def prepare_features(self, X, best_model, feature_indices):

        """

        Select the Tier-2 columns and standardise them.

        :param1 X:               (n, d) feature matrix.
        :param2 best_model:      selector result (default feature source).
        :param3 feature_indices: explicit columns, overriding best_model.

        :return: standardised (n, k) matrix.

        """

        if feature_indices is None:
            feature_indices = best_model['feature_indices']
        self.feature_indices = sorted(feature_indices)
        self.n_input_features_ = int(X.shape[1])    # disambiguates predict() inputs
        self.feature_scaler_ = StandardScaler()
        return self.feature_scaler_.fit_transform(X[:, self.feature_indices])

    #%% 3.2 Backbone hooks
    def build_backbone(self):

        """Construct the unfitted Tier-2 regressor."""

        return ClassicalGaussianProcess(
            length_scale=self.length_scale,
            kernel_variance=self.kernel_variance,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state,
            use_nystroem=self.use_nystroem,
            nystroem_components=self.nystroem_components,
            linear=self.linear,
            kernel_type=self.kernel_type,
            nu=self.nu,
            exact_gp_max_samples=self.exact_gp_max_samples,
        )

    def store_backbone(self, backbone):

        """Attach the backbone to this learner."""

        self.gp = backbone

    def backbone(self):

        """Return the fitted backbone, or None."""

        return self.gp

    def fit_backbone(self, backbone, X_subset, residuals, show_progress):

        """

        Train the backbone on the residual.

        :param1 backbone:      object from build_backbone().
        :param2 X_subset:      (n, k) standardised Tier-2 features.
        :param3 residuals:     length-n residual target.
        :param4 show_progress: print status lines.

        :return: (residual_pred, residual_std) on the training set.

        """

        backbone.fit(
            X_subset, residuals,
            optimize=self.optimize,
            n_restarts=self.n_restarts,
            max_iter=self.max_iter,
            show_progress=show_progress,
            optimize_subset_size=self.optimize_subset_size,
        )
        return backbone.predict_train(return_std=True)

    def describe_training(self):

        """Return the model description logged before the Tier-2 fit."""

        return f"GP residual, kernel {self.kernel_type}"

    def report_fit(self, rmse, mae, residual_std):

        """Log the post-fit quality."""

        log("Tier-2", f"{self.LABEL} [{self.gp.describe_backend()}]: RMSE {rmse:.4f} eV | "
                      f"MAE {mae:.4f} eV | mean std {np.mean(residual_std):.4f} eV")

    def backbone_metadata(self):

        """Return the backend-specific part of the residual_model dict."""

        return {
            'gp': self.gp,
            'length_scale': self.gp.length_scale,
            'kernel_variance': self.gp.kernel_variance,
            'sigma_noise': self.gp.sigma_noise,
            'kernel_type': self.gp.kernel_type,
            'nu': self.gp.nu,
            'use_nystroem': self.gp.use_nystroem,
            'nystroem_components': self.gp.nystroem_components,
            'backend': self.gp.describe_backend(),
        }

    #%% 3.3 Fit / predict
    def fit(self, X, y, y_pred, best_model, feature_indices=None,
            show_progress=True, y_pred_oof=None):

        """

        Fit the Tier-2 backbone on y - y_pred (or the out-of-fold residual).

        :param1 X:               (n, d) Tier-2 feature matrix.
        :param2 y:               length-n true target.
        :param3 y_pred:          length-n Tier-1 prediction.
        :param4 best_model:      selector result (default features).
        :param5 feature_indices: explicit columns.
        :param6 show_progress:   print status lines.
        :param7 y_pred_oof:      optional out-of-fold Tier-1 prediction.

        :return: (y_pred + residual_pred, residual_model dict).

        """

        residuals, baseline_mse = self.prepare_residuals(
            y, y_pred, y_pred_oof=y_pred_oof, show_progress=show_progress,
        )
        X_subset = self.prepare_features(X, best_model, feature_indices)

        if show_progress:
            log("Tier-2", f"Training {self.describe_training()}: "
                          f"{X_subset.shape[0]} samples x {X_subset.shape[1]} features")

        backbone = self.build_backbone()
        self.store_backbone(backbone)
        residual_pred, residual_std = self.fit_backbone(
            backbone, X_subset, residuals, show_progress,
        )
        final_pred = y_pred + residual_pred

        mse = np.mean((y - final_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - final_pred))

        if show_progress:
            self.report_fit(rmse, mae, residual_std)

        residual_model = {
            'model': self,
            'feature_indices': self.feature_indices,
            'residual_pred': residual_pred,
            'residual_std': residual_std,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'n_features': X_subset.shape[1],
            'residual_target_source': self.residual_target_source_,
            'baseline_mse': baseline_mse,
            'baseline_std': self.baseline_std_,
            'baseline_std_in_sample': self.baseline_std_in_sample_,
        }
        residual_model.update(self.backbone_metadata())

        return final_pred, residual_model

    def resolve_predict_features(self, X):

        """

        Select and scale the columns predict() uses (full or pre-selected input).

        :param1 X: (n, d) full or pre-selected feature matrix.

        :return: standardised (n, k) matrix.

        """

        n_selected = len(self.feature_indices)
        if X.shape[1] == self.n_input_features_:
            X_subset = X[:, list(self.feature_indices)]
        elif X.shape[1] == n_selected:
            X_subset = X
        else:
            raise ValueError(
                f"X has {X.shape[1]} features; expected "
                f"{self.n_input_features_} (full, as passed to fit) or "
                f"{n_selected} (pre-selected)."
            )
        if self.feature_scaler_ is not None:
            X_subset = self.feature_scaler_.transform(X_subset)
        return X_subset

    def predict(self, X, return_std=False, combine_baseline_uncertainty=False):

        """

        Predict residuals for new samples.

        :param1 X:                            (n, d) feature matrix.
        :param2 return_std:                   also return the predictive std.
        :param3 combine_baseline_uncertainty: add the baseline std in quadrature.

        :return: residual predictions, or (predictions, std).

        """

        backbone = self.backbone()
        if backbone is None or self.feature_indices is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")

        X_subset = self.resolve_predict_features(X)

        if return_std:
            pred, std = backbone.predict(X_subset, return_std=True)
            if combine_baseline_uncertainty and self.baseline_std_ is not None:
                return pred, np.sqrt(std ** 2 + self.baseline_std_ ** 2)
            return pred, std
        return backbone.predict(X_subset, return_std=False)
