#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

BayesianFrameworkQuantum whose subset KRR (and optionally the Tier-2 GP)
uses a fidelity quantum kernel.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from bayesian_framework.frameworks.quantum import BayesianFrameworkQuantum
from bayesian_framework.selectors.quantum_kernel import QuantumKernelModelSelector
from bayesian_framework.predictors.quantum_residual_learning import QuantumResidualLearning


#%% 1. Framework (quantum-kernel variant)
class BayesianFrameworkQuantumKernel(BayesianFrameworkQuantum):

    """

    QAOA subset selection with quantum-kernel KRR and quantum-kernel Tier 2.


    """

    def __init__(self, qk_feature_map='zz', qk_n_qubits=6, qk_n_layers=1,
                 qk_scale=1.0, quantum_residual=True, **kwargs):

        """

        :param1 qk_feature_map:   'zz' or 'z' feature map.
        :param2 qk_n_qubits:      qubit cap.
        :param3 qk_n_layers:      encoding layers.
        :param4 qk_scale:         angle scaling on standardised inputs.
        :param5 quantum_residual: also use the quantum kernel in the Tier-2 GP.
        :param6 kwargs:           BayesianFrameworkQuantum arguments.

        :return: None.

        """

        super().__init__(**kwargs)
        qk = dict(qk_feature_map=qk_feature_map, qk_n_qubits=int(qk_n_qubits),
                  qk_n_layers=int(qk_n_layers), qk_scale=float(qk_scale))

        self.model_selector = QuantumKernelModelSelector(
            **self.model_selector.shared_kwargs(), **qk,
        )
        if quantum_residual:
            self.residual_learner = QuantumResidualLearning(
                **self.residual_learner.shared_kwargs(), **qk,
            )
