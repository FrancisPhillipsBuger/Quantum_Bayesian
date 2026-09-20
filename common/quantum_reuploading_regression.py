#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Adapter giving DataReuploadingRegressor the KernelRidgeRegression.fit interface,
so the QUBO/QAOA selector can score subsets with a small per-subset circuit.

Created on: Mon Jun 30 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler

from common.kernel_ridge import fit_statistics, select_subset
from common.quantum_reuploading import DataReuploadingRegressor
from common.report import log


#%% 1. Predictor wrapper
class ReuploadingPredictor:

    """

    Feature standardiser + trained re-uploading circuit.


    """

    __slots__ = ("scaler", "vqc")

    def __init__(self, scaler, vqc):

        """

        :param1 scaler: fitted StandardScaler.
        :param2 vqc:    trained DataReuploadingRegressor.

        :return: None.

        """

        self.scaler = scaler
        self.vqc = vqc

    def predict(self, X):

        """

        Standardise X and predict with the circuit.

        :param1 X: (n, d) raw feature matrix.

        :return: length-n prediction vector.

        """

        Xs = self.scaler.transform(np.ascontiguousarray(X, dtype=np.float64))
        return self.vqc.predict(Xs)


#%% 2. Regression adapter
class ReuploadingRegression:

    """

    Variational re-uploading regressor with a KernelRidge-style fit().


    """

    def __init__(self, n_qubits=4, n_layers=3, scale=1.0, learning_rate=0.05,
                 epochs=20, batch_size=256, l2=0.0, random_state=0,
                 max_samples=400, patience=5, readout_learning_rate=None):

        """

        :param1 n_qubits:               qubit cap.
        :param2 n_layers:               re-uploading layers.
        :param3 scale:                  initial encoding scale.
        :param4 learning_rate:          Adam step size for circuit parameters.
        :param5 epochs:                 epochs per subset.
        :param6 batch_size:             mini-batch size.
        :param7 l2:                     L2 penalty on circuit parameters.
        :param8 random_state:           RNG seed.
        :param9 max_samples:            training subsample cap per subset (None → all);
                                        the mBIC variance is scored on the full set.
        :param10 patience:              early-stopping patience (None disables).
        :param11 readout_learning_rate: Adam step size for the readout (None → default ratio).

        :return: None.

        """

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.scale = float(scale)
        self.learning_rate = float(learning_rate)
        self.readout_learning_rate = readout_learning_rate
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.random_state = random_state
        self.max_samples = max_samples
        self.patience = patience
        self.model = None
        self.feature_indices = None

    def fit(self, X, y, feature_indices=None, use_nystroem=False,
            nystroem_components=100, nystroem_random_state=0, linear=False,
            need_predictions=True):

        """

        Train the circuit on one feature subset; kernel arguments are ignored.

        :param1 X:               (n, d) feature matrix.
        :param2 y:               length-n target vector.
        :param3 feature_indices: optional column indices.

        :return: dict with model + diagnostics, or None on failure.

        """

        try:
            self.feature_indices = feature_indices
            X_subset = select_subset(X, feature_indices)
            if X_subset is None:
                return None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            y = np.asarray(y, dtype=np.float64).ravel()
            n = y.shape[0]
            p = X_subset.shape[1]

            held_out = self.max_samples is not None and n > self.max_samples
            if held_out:
                rng = np.random.RandomState(self.random_state)
                tr = rng.choice(n, int(self.max_samples), replace=False)
            else:
                tr = np.arange(n)

            vqc = DataReuploadingRegressor(
                n_qubits=self.n_qubits, n_layers=self.n_layers, scale=self.scale,
                learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size, l2=self.l2,
                random_state=self.random_state, patience=self.patience,
                readout_learning_rate=self.readout_learning_rate,
                laplace=False,          # subset scoring never reads a predictive std
            )
            vqc.fit(X_scaled[tr], y[tr], show_progress=False)

            self.model = ReuploadingPredictor(scaler, vqc)
            y_pred = vqc.predict(X_scaled)

            residuals = y - y_pred
            ss_residual = float(residuals @ residuals)
            y_c = y - y.mean()
            stats = fit_statistics(ss_residual, float(y_c @ y_c), n, p)

            return {
                'model': self.model,
                'y_pred': y_pred,
                'residuals': residuals,
                **stats,
                'sigma_gcv_sq': stats['sigma_ml_sq'],    # no closed form, no GCV correction
                'sigma_estimator': 'partly-held-out' if held_out else 'in-sample',
                'alpha': None,
                'gamma': None,
                'kernel': 'variational_reuploading',
                'n_params': int(vqc._n_params()),       # mBIC charges every circuit weight
                'n_features': p,
                'linear': False,
                'use_nystroem': False,
                'nystroem_components': None,
            }

        except Exception as e:
            log("Warning", f"Re-uploading regression fit failed: {e}")
            return None
