#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Quantum-kernel ridge regression: KernelRidgeRegression with the RBF kernel
replaced by a fidelity quantum kernel (full or Nystrom).

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler

from common.kernel_ridge import (BaseRidgeRegression, fit_statistics, gcv_sigma_sq,
                                 normal_equations, ridge_gcv_path, sample_landmarks,
                                 whiten_kernel)
from common.quantum_kernel import QuantumKernel
from common.report import log


#%% 1. Lightweight predictors
class QuantumNystroemRidge:

    """

    Quantum-Nystrom feature map + linear ridge predictor.


    """

    __slots__ = ("scaler", "kernel", "landmarks", "normalization_", "coef_", "intercept_")

    def __init__(self, scaler, kernel, landmarks, normalization_, coef_, intercept_):

        """

        Store the fitted quantum-Nystrom map and ridge coefficients.

        :param1 scaler:         fitted StandardScaler.
        :param2 kernel:         fidelity kernel object.
        :param3 landmarks:      (m, d) standardised landmarks.
        :param4 normalization_: (m, r) whitening matrix.
        :param5 coef_:          ridge coefficient vector.
        :param6 intercept_:     intercept (target mean).

        :return: None.

        """

        self.scaler = scaler
        self.kernel = kernel
        self.landmarks = landmarks
        self.normalization_ = normalization_
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, X):

        """

        Predict targets for X via the quantum-Nystrom feature map.

        :param1 X: (n, d) raw feature matrix.

        :return: length-n prediction vector.

        """

        Xs = self.scaler.transform(np.ascontiguousarray(X, dtype=np.float64))
        K_xm = self.kernel.kernel(Xs, self.landmarks)
        return (K_xm @ self.normalization_) @ self.coef_ + self.intercept_


class QuantumFullRidge:

    """

    Full dual quantum-kernel ridge predictor (no Nystrom).


    """

    __slots__ = ("scaler", "kernel", "X_fit", "dual", "intercept_")

    def __init__(self, scaler, kernel, X_fit, dual, intercept_):

        """

        Store the training data and dual coefficients.

        :param1 scaler:     fitted StandardScaler.
        :param2 kernel:     fidelity kernel object.
        :param3 X_fit:      (n, d) standardised training features.
        :param4 dual:       length-n dual coefficients.
        :param5 intercept_: intercept (target mean).

        :return: None.

        """

        self.scaler = scaler
        self.kernel = kernel
        self.X_fit = X_fit
        self.dual = dual
        self.intercept_ = intercept_

    def predict(self, X):

        """

        Predict targets for X against the stored training kernel.

        :param1 X: (n, d) raw feature matrix.

        :return: length-n prediction vector.

        """

        Xs = self.scaler.transform(np.ascontiguousarray(X, dtype=np.float64))
        K = self.kernel.kernel(Xs, self.X_fit)
        return K @ self.dual + self.intercept_


#%% 2. Quantum-kernel ridge regression
class QuantumKernelRidgeRegression(BaseRidgeRegression):

    """

    Kernel ridge regression with a fidelity quantum kernel.


    """

    def __init__(self, alpha=1.0, feature_map="zz", n_qubits=6, n_layers=1,
                 scale=1.0, alpha_grid=None, kernel=None, kernel_label=None):

        """

        :param1 alpha:        L2 regularisation strength.
        :param2 feature_map:  'zz' or 'z' feature map.
        :param3 n_qubits:     qubit cap.
        :param4 n_layers:     encoding layers.
        :param5 scale:        angle scaling on standardised inputs.
        :param6 alpha_grid:   optional ridge grid; GCV-best alpha per subset (Nystrom path).
        :param7 kernel:       prebuilt kernel exposing kernel(A, B); None → QuantumKernel.
        :param8 kernel_label: name reported in the result dict; None → 'quantum:<map>'.

        :return: None.

        """

        super().__init__(alpha=alpha, alpha_grid=alpha_grid)
        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scale = scale
        self.kernel = kernel if kernel is not None else QuantumKernel(
            feature_map=feature_map, n_qubits=n_qubits,
            n_layers=n_layers, scale=scale,
        )
        self.kernel_label = (kernel_label if kernel_label is not None
                             else f"quantum:{feature_map}")

    def fit(self, X, y, feature_indices=None, use_nystroem=False,
            nystroem_components=100, nystroem_random_state=0, linear=False,
            need_predictions=True):

        """

        Fit a quantum-kernel ridge model on one feature subset (KRR-compatible signature).

        :param1 X:                     (n, d) feature matrix.
        :param2 y:                     length-n target vector.
        :param3 feature_indices:       optional column indices.
        :param4 use_nystroem:          enable the quantum-Nystrom approximation.
        :param5 nystroem_components:   landmark count.
        :param6 nystroem_random_state: landmark-sampling seed.
        :param7 linear:                ignored (signature parity).
        :param8 need_predictions:      materialise y_pred / residuals.

        :return: dict with model + diagnostics, or None on failure.

        """

        try:
            X_subset = self.take_subset(X, feature_indices)
            if X_subset is None:
                return None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)

            y = np.asarray(y, dtype=np.float64).ravel()
            n = y.shape[0]
            y_mean = y.mean()
            y_c = y - y_mean
            ss_total = float(y_c @ y_c)

            if use_nystroem:
                m = min(int(nystroem_components if nystroem_components is not None else 100), n)
                if m < 1:
                    return None
                landmarks = sample_landmarks(X_scaled, m, nystroem_random_state)
                normalization = whiten_kernel(self.kernel.kernel(landmarks, landmarks))

                K_xm = self.kernel.kernel(X_scaled, landmarks)
                G, b = normal_equations(K_xm, y_c, normalization=normalization)
                best = ridge_gcv_path(G, b, ss_total, n, self.alphas())
                coef = best['coef']

                self.model = QuantumNystroemRidge(
                    scaler, self.kernel, landmarks, normalization, coef, y_mean,
                )
                y_pred = ((K_xm @ normalization) @ coef + y_mean
                          if need_predictions else None)
                residuals = (y - y_pred) if y_pred is not None else None
                ss_residual = best['ss_residual']
                p = int(normalization.shape[1])
                n_components = m
                dof = best['dof']
                chosen_alpha = best['alpha']
            else:
                # Full dual KRR keeps the single configured alpha
                K = self.kernel.kernel(X_scaled, X_scaled)
                K.flat[::n + 1] += float(self.alpha)
                try:
                    c, lower = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
                    dual = cho_solve((c, lower), y_c, check_finite=False)
                except Exception:
                    dual = np.linalg.solve(K, y_c)

                self.model = QuantumFullRidge(scaler, self.kernel, X_scaled, dual, y_mean)
                y_pred = self.model.predict(X_subset)
                residuals = y - y_pred
                ss_residual = float(residuals @ residuals)
                p = X_subset.shape[1]
                n_components = None
                dof = None          # no cheap trace: score on the in-sample variance
                chosen_alpha = float(self.alpha)

            stats = fit_statistics(ss_residual, ss_total, n, p)
            if dof is None:
                sigma_gcv_sq, gcv_ok = stats['sigma_ml_sq'], False
            else:
                sigma_gcv_sq, gcv_ok = gcv_sigma_sq(ss_residual, n, dof)

            return {
                'model': self.model,
                'y_pred': y_pred,
                'residuals': residuals,
                **stats,
                'sigma_gcv_sq': sigma_gcv_sq,
                'sigma_estimator': 'gcv' if gcv_ok else 'in-sample',
                'dof': dof,
                'alpha': chosen_alpha,
                'gamma': None,
                'kernel': self.kernel_label,
                'n_features': p,
                'linear': False,
                'use_nystroem': bool(use_nystroem),
                'nystroem_components': n_components,
            }

        except Exception as e:
            log("Warning", f"Quantum KRR fit failed: {e}")
            return None
