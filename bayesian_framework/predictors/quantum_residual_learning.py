#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tier-2 GP residual learning with a fidelity quantum kernel.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel

from bayesian_framework.predictors.residual_learning import ClassicalGaussianProcess, ResidualLearning
from common.quantum_kernel import QuantumKernel


#%% 1. scikit-learn kernel wrapper
class QuantumGPKernel(Kernel):

    """

    scikit-learn Kernel wrapping a fidelity kernel; it has no hyper-parameters,
    so the gradient is empty (an outer ConstantKernel carries the amplitude).


    """

    def __init__(self, feature_map="zz", n_qubits=6, n_layers=1, scale=1.0,
                 qk=None):

        """

        :param1 feature_map: 'zz' or 'z' feature map.
        :param2 n_qubits:    qubit cap.
        :param3 n_layers:    encoding layers.
        :param4 scale:       angle scaling on standardised inputs.
        :param5 qk:          prebuilt kernel exposing kernel(A, B); None → QuantumKernel.

        :return: None.

        """

        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scale = scale
        self.qk = qk if qk is not None else QuantumKernel(
            feature_map=feature_map, n_qubits=n_qubits,
            n_layers=n_layers, scale=scale,
        )

    def __call__(self, X, Y=None, eval_gradient=False):

        """

        Evaluate the fidelity kernel matrix (sklearn Kernel protocol).

        :param1 X:             (n, d) array.
        :param2 Y:             (m, d) array; None → X.
        :param3 eval_gradient: also return the (empty) gradient.

        :return: (n, m) kernel matrix [, zero-width gradient].

        """

        K = self.qk.kernel(X, Y)
        if eval_gradient:
            return K, np.empty((np.shape(X)[0], np.shape(X)[0], 0))
        return K

    def diag(self, X):

        """Return the kernel diagonal, identically 1."""

        return np.ones(np.shape(X)[0], dtype=np.float64)

    def is_stationary(self):

        """Return False; the fidelity kernel is not stationary."""

        return False


#%% 2. Quantum GP backbone
class QuantumGaussianProcess(ClassicalGaussianProcess):

    """

    GP backbone whose kernel (exact or Nystrom) is a fidelity quantum kernel.


    """

    def __init__(self, sigma_noise=0.1, kernel_variance=1.0, random_state=None,
                 qk_feature_map="zz", qk_n_qubits=6, qk_n_layers=1, qk_scale=1.0,
                 use_nystroem=False, nystroem_components=512,
                 exact_gp_max_samples=2000, qk=None, qk_label=None):

        """

        :param1 sigma_noise:           GP observation noise.
        :param2 kernel_variance:       ConstantKernel amplitude.
        :param3 random_state:          RNG seed.
        :param4 qk_feature_map:        'zz' or 'z' feature map.
        :param5 qk_n_qubits:           qubit cap.
        :param6 qk_n_layers:           encoding layers.
        :param7 qk_scale:              angle scaling on standardised inputs.
        :param8 use_nystroem:          use the quantum-Nystrom solver.
        :param9 nystroem_components:   Nystrom landmark count.
        :param10 exact_gp_max_samples: largest n for the exact GP.
        :param11 qk:                   prebuilt kernel; None → QuantumKernel.
        :param12 qk_label:             kernel name for logs.

        :return: None.

        """

        super().__init__(
            length_scale=1.0, kernel_variance=kernel_variance,
            sigma_noise=sigma_noise, random_state=random_state,
            use_nystroem=use_nystroem, nystroem_components=nystroem_components,
            linear=False, kernel_type="rbf",
            exact_gp_max_samples=exact_gp_max_samples,
        )
        self.qk_feature_map = qk_feature_map
        self.qk_n_qubits = int(qk_n_qubits)
        self.qk_n_layers = int(qk_n_layers)
        self.qk_scale = float(qk_scale)
        self.qk = qk if qk is not None else QuantumKernel(
            feature_map=qk_feature_map, n_qubits=self.qk_n_qubits,
            n_layers=self.qk_n_layers, scale=self.qk_scale,
        )
        self.qk_label = (qk_label if qk_label is not None
                         else f"quantum:{qk_feature_map}")

    def build_nystroem_kernel_fn(self, X):

        """

        Return the fidelity kernel for the Nystrom map (bound method, so it pickles).

        :param1 X: (n, d) training features (unused).

        :return: (callable k(A, B), label).

        """

        self.effective_length_scale_ = None
        return (self.qk.kernel,
                f"{self.qk_label}"
                f"(q<={self.qk_n_qubits}, L={self.qk_n_layers}, "
                f"scale={self.qk_scale:g})")

    def kernel_label(self):

        """Return the fidelity-kernel name."""

        return self.qk_label

    def build_kernel(self):

        """Return ConstantKernel * QuantumGPKernel."""

        quantum = QuantumGPKernel(
            feature_map=self.qk_feature_map, n_qubits=self.qk_n_qubits,
            n_layers=self.qk_n_layers, scale=self.qk_scale, qk=self.qk,
        )
        return ConstantKernel(
            self.kernel_variance, constant_value_bounds=(1e-5, 1e5),
        ) * quantum

    def fit(self, X, y, optimize=True, n_restarts=0, max_iter=100,
            show_progress=True, optimize_subset_size=None):

        """

        Fit the GP; optimisation is always off (no length scale to tune).

        :param1 X: (n, d) feature matrix.
        :param2 y: length-n target vector.

        :return: self.

        """

        return super().fit(
            X, y, optimize=False, n_restarts=0, max_iter=max_iter,
            show_progress=show_progress, optimize_subset_size=optimize_subset_size,
        )


#%% 3. Quantum residual learner
class QuantumResidualLearning(ResidualLearning):

    """

    Residual learner with a quantum-kernel GP backbone.


    """

    PREFERRED_RESIDUAL_DIM = 16
    LABEL = "Quantum GP residual"

    def __init__(self, qk_feature_map="zz", qk_n_qubits=6, qk_n_layers=1,
                 qk_scale=1.0, qk=None, qk_label=None, **kwargs):

        """

        :param1 qk_feature_map: 'zz' or 'z' feature map.
        :param2 qk_n_qubits:    qubit cap.
        :param3 qk_n_layers:    encoding layers.
        :param4 qk_scale:       angle scaling on standardised inputs.
        :param5 qk:             prebuilt kernel (e.g. ReuploadingFidelityKernel); None → ZZ.
        :param6 qk_label:       kernel name for logs.
        :param7 kwargs:         ResidualLearning arguments (kernel_type / nu / linear ignored).

        :return: None.

        """

        super().__init__(**kwargs)
        self.qk_feature_map = qk_feature_map
        self.qk_n_qubits = int(qk_n_qubits)
        self.qk_n_layers = int(qk_n_layers)
        self.qk_scale = float(qk_scale)
        self.qk = qk
        self.qk_label = qk_label

    def build_backbone(self):

        """Return the quantum-kernel GP backbone."""

        return QuantumGaussianProcess(
            sigma_noise=self.sigma_noise,
            kernel_variance=self.kernel_variance,
            random_state=self.random_state,
            qk_feature_map=self.qk_feature_map,
            qk_n_qubits=self.qk_n_qubits,
            qk_n_layers=self.qk_n_layers,
            qk_scale=self.qk_scale,
            use_nystroem=self.use_nystroem,
            nystroem_components=self.nystroem_components,
            exact_gp_max_samples=self.exact_gp_max_samples,
            qk=self.qk, qk_label=self.qk_label,
        )

    def describe_training(self):

        """Return the model description logged before the Tier-2 fit."""

        label = self.qk_label or ("feature_map=" + str(self.qk_feature_map))
        return f"quantum-kernel GP residual, {label}"

    def backbone_metadata(self):

        """

        Classical metadata with the fidelity-kernel name; landmark count only when used.

        :return: backend-specific part of the residual_model dict.

        """

        meta = super().backbone_metadata()
        meta['kernel_type'] = self.gp.kernel_label()
        meta['nystroem_components'] = (
            self.gp.nystroem_components if self.gp.use_nystroem else None
        )
        return meta
