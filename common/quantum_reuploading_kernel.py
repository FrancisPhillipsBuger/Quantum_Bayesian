#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Fidelity kernel of the re-uploading circuit's own encoding at frozen (initial)
theta: the implicit counterpart of DataReuploadingRegressor over the same encoding.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np

from common.quantum_kernel import FidelityKernel
from common.quantum_kernel_ridge import QuantumKernelRidgeRegression
from common.quantum_reuploading import DataReuploadingRegressor


#%% 1. Fidelity kernel over the re-uploading encoding
class ReuploadingFidelityKernel(FidelityKernel):

    """

    k(x, y) = |<psi(x)|psi(y)>|^2 for the re-uploading circuit at its initial theta;
    reuses the regressor's configure(), init_params() and _statevector().


    """

    def __init__(self, n_qubits=6, n_layers=3, scale=1.0, random_state=0):

        """

        :param1 n_qubits:     qubit cap (mirrors the regressor).
        :param2 n_layers:     uploads per feature (mirrors the regressor).
        :param3 scale:        initial encoding scale.
        :param4 random_state: seed of the frozen theta draw.

        :return: None.

        """

        self.n_qubits = max(1, int(n_qubits))
        self.n_layers = max(1, int(n_layers))
        self.scale = float(scale)
        self.random_state = random_state
        self._cache = {}        # d -> (configured regressor, frozen params)

    def circuit_for(self, d):

        """

        Return the configured regressor and frozen parameters for d features.

        :param1 d: input feature count.

        :return: (DataReuploadingRegressor, (P,) parameter vector).

        """

        hit = self._cache.get(d)
        if hit is not None:
            return hit

        vqc = DataReuploadingRegressor(
            n_qubits=self.n_qubits, n_layers=self.n_layers, scale=self.scale,
            random_state=self.random_state, laplace=False,
        ).configure(d)
        params = vqc.init_params(np.random.RandomState(self.random_state))

        self._cache[d] = (vqc, params)
        return vqc, params

    def statevectors(self, X):

        """

        Map a batch of inputs to the circuit's statevectors.

        :param1 X: (n, d) array of standardised features.

        :return: (n, 2^q) complex statevector batch.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        vqc, params = self.circuit_for(X.shape[1])
        return vqc._statevector(X, params)

    def label(self):

        """Return a human-readable kernel name."""

        return (f"reuploading-fidelity[init]"
                f"(q<={self.n_qubits}, uploads={self.n_layers}, "
                f"scale={self.scale:g})")


#%% 2. Kernel ridge over that kernel
class ReuploadingKernelRidgeRegression(QuantumKernelRidgeRegression):

    """

    Kernel ridge regression with the re-uploading encoding's fidelity kernel.


    """

    def __init__(self, alpha=1.0, n_qubits=6, n_layers=3, scale=1.0,
                 random_state=0, alpha_grid=None):

        """

        :param1 alpha:        L2 regularisation strength.
        :param2 n_qubits:     qubit cap (mirror the explicit model).
        :param3 n_layers:     uploads per feature (mirror the explicit model).
        :param4 scale:        encoding scale (mirror the explicit model).
        :param5 random_state: seed of the frozen theta draw.
        :param6 alpha_grid:   optional ridge grid (GCV-best per subset).

        :return: None.

        """

        kernel = ReuploadingFidelityKernel(
            n_qubits=n_qubits, n_layers=n_layers, scale=scale,
            random_state=random_state,
        )
        super().__init__(
            alpha=alpha, alpha_grid=alpha_grid, kernel=kernel,
            kernel_label=kernel.label(),
        )
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.scale = float(scale)
        self.random_state = random_state
