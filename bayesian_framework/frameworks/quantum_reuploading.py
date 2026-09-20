#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

BayesianFrameworkQuantum with a re-uploading selector (kernel proxy + circuit
refit) and a Tier-2 variational re-uploading residual learner (Jerbi et al.).

Created on: Mon Jun 30 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from bayesian_framework.frameworks.quantum import BayesianFrameworkQuantum
from bayesian_framework.predictors.quantum_reuploading_residual import ReuploadingResidualLearning
from bayesian_framework.predictors.quantum_residual_learning import QuantumResidualLearning
from bayesian_framework.selectors.quantum_reuploading import ReuploadingModelSelector
from common.quantum_reuploading_kernel import ReuploadingFidelityKernel


#%% 1. Framework (variational re-uploading variant)
class BayesianFrameworkQuantumReupload(BayesianFrameworkQuantum):

    """

    QAOA selection and Tier-2 residual learning driven by the re-uploading model.


    """

    def __init__(self, ru_n_qubits=4, ru_n_layers=3, ru_scale=1.0,
                 ru_learning_rate=0.05, ru_epochs=30, ru_batch_size=256,
                 ru_l2=0.0, ru_selection_epochs=20, ru_selection_max_samples=400,
                 ru_patience=5, ru_rescore_top_n=64,
                 ru_readout_learning_rate=None,
                 ru_proxy_backend="quantum_kernel", ru_proxy_feature_map="zz",
                 ru_proxy_n_qubits=6, ru_proxy_n_layers=1, ru_proxy_scale=1.0,
                 ru_residual_backend="explicit", ru_laplace=True,
                 ru_laplace_prior_precision=1.0, **kwargs):

        """

        :param1 ru_*:                       circuit geometry and training settings
                                            (see ReuploadingResidualLearning).
        :param2 ru_selection_*:             per-finalist budget in the selector.
        :param3 ru_proxy_*:                 stage-A proxy backend and its kernel settings.
        :param4 ru_residual_backend:        Tier 2: 'explicit' (circuit), 'implicit' (its
                                            fidelity kernel) or 'classical'.
        :param5 ru_laplace:                 linearised-Laplace predictive std.
        :param6 ru_laplace_prior_precision: prior precision of that posterior.
        :param7 kwargs:                     BayesianFrameworkQuantum arguments.

        :return: None.

        """

        super().__init__(**kwargs)

        self.ru_n_qubits = int(ru_n_qubits)
        self.ru_n_layers = int(ru_n_layers)
        self.ru_scale = float(ru_scale)
        self.ru_residual_backend = str(ru_residual_backend).lower()
        if self.ru_residual_backend not in {"explicit", "implicit", "classical"}:
            raise ValueError(
                "ru_residual_backend must be 'explicit', 'implicit' or 'classical'"
            )

        self.model_selector = ReuploadingModelSelector(
            **self.model_selector.shared_kwargs(),
            ru_n_qubits=self.ru_n_qubits,
            ru_n_layers=self.ru_n_layers,
            ru_scale=self.ru_scale,
            ru_learning_rate=float(ru_learning_rate),
            ru_batch_size=int(ru_batch_size),
            ru_l2=float(ru_l2),
            ru_selection_epochs=int(ru_selection_epochs),
            ru_selection_max_samples=ru_selection_max_samples,
            ru_random_state=self.qubo_seed,
            ru_patience=ru_patience,
            ru_rescore_top_n=int(ru_rescore_top_n),
            ru_readout_learning_rate=ru_readout_learning_rate,
            proxy_backend=str(ru_proxy_backend).lower(),
            proxy_feature_map=ru_proxy_feature_map,
            proxy_n_qubits=int(ru_proxy_n_qubits),
            proxy_n_layers=int(ru_proxy_n_layers),
            proxy_scale=float(ru_proxy_scale),
        )

        rl = self.residual_learner
        if self.ru_residual_backend == "explicit":
            self.residual_learner = ReuploadingResidualLearning(
                **rl.shared_kwargs(),
                ru_n_qubits=self.ru_n_qubits,
                ru_n_layers=self.ru_n_layers,
                ru_scale=self.ru_scale,
                ru_learning_rate=float(ru_learning_rate),
                ru_epochs=int(ru_epochs),
                ru_batch_size=int(ru_batch_size),
                ru_l2=float(ru_l2),
                ru_patience=ru_patience,
                ru_readout_learning_rate=ru_readout_learning_rate,
                ru_laplace=bool(ru_laplace),
                ru_laplace_prior_precision=float(ru_laplace_prior_precision),
            )
        elif self.ru_residual_backend == "implicit":
            self.residual_learner = self.build_implicit_residual(rl)

    def build_implicit_residual(self, rl):

        """

        GP residual learner on the circuit's own fidelity kernel at frozen theta,
        pinned to the explicit learner's Tier-2 width for a controlled comparison.

        :param1 rl: classical ResidualLearning to copy settings from.

        :return: QuantumResidualLearning instance.

        """

        kernel = ReuploadingFidelityKernel(
            n_qubits=self.ru_n_qubits,
            n_layers=self.ru_n_layers,
            scale=self.ru_scale,
            random_state=self.qubo_seed,
        )
        learner = QuantumResidualLearning(
            **rl.shared_kwargs(),
            qk=kernel,
            qk_label=kernel.label(),
        )
        learner.PREFERRED_RESIDUAL_DIM = ReuploadingResidualLearning.PREFERRED_RESIDUAL_DIM
        return learner
