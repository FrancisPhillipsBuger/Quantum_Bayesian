#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tier-2 residual learning with the explicit (variational) data re-uploading
circuit in place of the GP backbone.

Created on: Mon Jun 30 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np

from bayesian_framework.predictors.residual_learning import ResidualLearning
from common.quantum_reuploading import DataReuploadingRegressor
from common.report import log


#%% 1. Variational re-uploading residual learner
class ReuploadingResidualLearning(ResidualLearning):

    """

    Drop-in ResidualLearning whose backbone is a DataReuploadingRegressor.


    """

    PREFERRED_RESIDUAL_DIM = 48
    LABEL = "Re-uploading residual"

    def __init__(self, ru_n_qubits=4, ru_n_layers=3, ru_scale=1.0,
                 ru_learning_rate=0.05, ru_epochs=30, ru_batch_size=256,
                 ru_l2=0.0, ru_patience=5, ru_readout_learning_rate=None,
                 ru_laplace=True, ru_laplace_prior_precision=1.0, **kwargs):

        """

        :param1 ru_n_qubits:                 qubit cap.
        :param2 ru_n_layers:                 re-uploading layers.
        :param3 ru_scale:                    initial encoding scale.
        :param4 ru_learning_rate:            Adam step size for circuit parameters.
        :param5 ru_epochs:                   training epochs.
        :param6 ru_batch_size:               mini-batch size.
        :param7 ru_l2:                       L2 penalty on circuit parameters.
        :param8 ru_patience:                 early-stopping patience (None disables).
        :param9 ru_readout_learning_rate:    Adam step size for the readout.
        :param10 ru_laplace:                 linearised-Laplace predictive std.
        :param11 ru_laplace_prior_precision: prior precision of that posterior.
        :param12 kwargs:                     ResidualLearning arguments (GP knobs ignored).

        :return: None.

        """

        super().__init__(**kwargs)
        self.ru_n_qubits = int(ru_n_qubits)
        self.ru_n_layers = int(ru_n_layers)
        self.ru_scale = float(ru_scale)
        self.ru_learning_rate = float(ru_learning_rate)
        self.ru_epochs = int(ru_epochs)
        self.ru_batch_size = int(ru_batch_size)
        self.ru_l2 = float(ru_l2)
        self.ru_patience = ru_patience
        self.ru_readout_learning_rate = ru_readout_learning_rate
        self.ru_laplace = bool(ru_laplace)
        self.ru_laplace_prior_precision = float(ru_laplace_prior_precision)
        self.model_ = None

    #%% 1.1 Backbone hooks
    def build_backbone(self):

        """Return the untrained re-uploading circuit."""

        return DataReuploadingRegressor(
            n_qubits=self.ru_n_qubits,
            n_layers=self.ru_n_layers,
            scale=self.ru_scale,
            learning_rate=self.ru_learning_rate,
            epochs=self.ru_epochs,
            batch_size=self.ru_batch_size,
            l2=self.ru_l2,
            random_state=self.random_state if self.random_state is not None else 0,
            patience=self.ru_patience,
            readout_learning_rate=self.ru_readout_learning_rate,
            laplace=self.ru_laplace,
            laplace_prior_precision=self.ru_laplace_prior_precision,
        )

    def store_backbone(self, backbone):

        """Keep the circuit under model_."""

        self.model_ = backbone

    def backbone(self):

        """Return the trained circuit, or None."""

        return self.model_

    def fit_backbone(self, backbone, X_subset, residuals, show_progress):

        """

        Train the circuit and return its training-set fit.

        :param1 backbone:      DataReuploadingRegressor.
        :param2 X_subset:      (n, k) standardised Tier-2 features.
        :param3 residuals:     length-n residual target.
        :param4 show_progress: show the epoch progress bar.

        :return: (residual_pred, residual_std) on the training set.

        """

        backbone.fit(X_subset, residuals, show_progress=show_progress)
        return backbone.predict(X_subset, return_std=True)

    def describe_training(self):

        """Return the model description logged before the Tier-2 fit."""

        return (f"variational re-uploading residual, "
                f"q <= {self.ru_n_qubits}, L = {self.ru_n_layers}")

    def report_fit(self, rmse, mae, residual_std):

        """Log fit quality, uncertainty source and circuit size."""

        unc = ("Laplace" if self.model_.posterior_cov_ is not None
               else "constant training RMSE")
        log("Tier-2", f"{self.LABEL}: RMSE {rmse:.4f} eV | MAE {mae:.4f} eV")
        log("Tier-2", f"Predictive std ({unc}): mean {np.mean(residual_std):.4f}, "
                      f"range [{np.min(residual_std):.4f}, {np.max(residual_std):.4f}] eV")
        log("Tier-2", f"Circuit: {self.model_.q_} qubits x {self.model_.n_layers_} layers, "
                      f"{self.model_._n_params()} parameters")

    def backbone_metadata(self):

        """

        Circuit metadata; GP fields kept as None so the schema matches other backends.

        :return: backend-specific part of the residual_model dict.

        """

        return {
            'gp': None,
            'length_scale': None,
            'kernel_variance': None,
            'sigma_noise': self.sigma_noise,
            'kernel_type': 'variational_reuploading',
            'uncertainty': ('linearised-laplace'
                            if self.model_.posterior_cov_ is not None
                            else 'constant-train-rmse'),
            'n_params': int(self.model_._n_params()),
            'nu': None,
            'use_nystroem': False,
            'nystroem_components': None,
            'n_qubits': self.model_.q_,
            'n_layers': self.model_.n_layers_,
            'backend': 'variational re-uploading circuit',
        }
