#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

QAOA subset search (QuantumModelSelector) with every subset fitted by
fidelity quantum-kernel ridge regression instead of the RBF KRR.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from bayesian_framework.selectors.quantum import QuantumModelSelector
from common.quantum_kernel_ridge import QuantumKernelRidgeRegression


#%% 1. Quantum-kernel selector
class QuantumKernelModelSelector(QuantumModelSelector):

    """

    Drop-in QuantumModelSelector with a quantum-kernel KRR backend.


    """

    def __init__(self, qk_feature_map='zz', qk_n_qubits=6, qk_n_layers=1,
                 qk_scale=1.0, **kwargs):

        """

        :param1 qk_feature_map: 'zz' or 'z' feature map.
        :param2 qk_n_qubits:    qubit cap.
        :param3 qk_n_layers:    encoding layers.
        :param4 qk_scale:       angle scaling on standardised inputs.
        :param5 kwargs:         QuantumModelSelector arguments.

        :return: None.

        """

        super().__init__(**kwargs)
        self.qk_feature_map = qk_feature_map
        self.qk_n_qubits = int(qk_n_qubits)
        self.qk_n_layers = int(qk_n_layers)
        self.qk_scale = float(qk_scale)

        self.regression = QuantumKernelRidgeRegression(
            alpha=self.alpha,
            alpha_grid=self.alpha_grid,
            feature_map=qk_feature_map,
            n_qubits=self.qk_n_qubits,
            n_layers=self.qk_n_layers,
            scale=self.qk_scale,
        )
