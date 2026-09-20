#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

QAOA subset search with two-stage scoring: a closed-form kernel-ridge proxy
ranks the pool (stage A), then the top-N subsets are refit with the variational
re-uploading circuit and rescored (stage B); the A/B rank agreement is reported.

Created on: Mon Jun 30 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from scipy.stats import spearmanr

from bayesian_framework.selectors.quantum import QuantumModelSelector
from common.quantum_kernel_ridge import QuantumKernelRidgeRegression
from common.quantum_reuploading_kernel import ReuploadingKernelRidgeRegression
from common.quantum_reuploading_regression import ReuploadingRegression
from common.report import log

PROXY_LABELS = {'quantum_kernel': 'quantum-kernel',
                'reuploading_kernel': 'reuploading-kernel',
                'classical': 'classical RBF'}


#%% 1. Proxy / circuit agreement
def rank_agreement(proxy_mbic, circuit_mbic):

    """

    Compare the stage-A proxy ranking with the stage-B circuit ranking.

    :param1 proxy_mbic:   dict {subset key: proxy mBIC}.
    :param2 circuit_mbic: dict {subset key: circuit mBIC}.

    :return: (Spearman rho or None, proxy rank of the circuit winner or None, n compared).

    """

    keys = [k for k in circuit_mbic
            if k in proxy_mbic
            and np.isfinite(circuit_mbic[k]) and np.isfinite(proxy_mbic[k])]
    if len(keys) < 3:
        return None, None, len(keys)

    a = np.asarray([proxy_mbic[k] for k in keys], dtype=np.float64)
    b = np.asarray([circuit_mbic[k] for k in keys], dtype=np.float64)

    rho = None
    if np.ptp(a) > 0 and np.ptp(b) > 0:
        try:
            rho = float(spearmanr(a, b).statistic)
        except Exception:
            rho = None

    winner = keys[int(np.argmin(b))]
    proxy_order = sorted(proxy_mbic, key=lambda k: proxy_mbic[k])
    return rho, proxy_order.index(winner) + 1, len(keys)


#%% 2. Re-uploading selector
class ReuploadingModelSelector(QuantumModelSelector):

    """

    QAOA proposer with a kernel-ridge proxy (quantum kernel by default) and a
    variational re-uploading refit + mBIC rescore of the finalists.


    """

    def __init__(self, ru_n_qubits=4, ru_n_layers=3, ru_scale=1.0,
                 ru_learning_rate=0.05, ru_batch_size=256, ru_l2=0.0,
                 ru_selection_epochs=20, ru_selection_max_samples=400,
                 ru_random_state=0, ru_patience=5, ru_rescore_top_n=64,
                 ru_readout_learning_rate=None,
                 proxy_backend="quantum_kernel", proxy_feature_map="zz",
                 proxy_n_qubits=6, proxy_n_layers=1, proxy_scale=1.0, **kwargs):

        """

        :param1 ru_n_qubits:               circuit qubit cap.
        :param2 ru_n_layers:               re-uploading layers.
        :param3 ru_scale:                  initial encoding scale.
        :param4 ru_learning_rate:          Adam step size.
        :param5 ru_batch_size:             mini-batch size.
        :param6 ru_l2:                     L2 penalty on circuit parameters.
        :param7 ru_selection_epochs:       epochs per finalist.
        :param8 ru_selection_max_samples:  training subsample per finalist.
        :param9 ru_random_state:           circuit RNG seed.
        :param10 ru_patience:              early-stopping patience (None disables).
        :param11 ru_rescore_top_n:         finalists refit in stage B.
        :param12 ru_readout_learning_rate: Adam step size for the readout.
        :param13 proxy_backend:            'quantum_kernel' | 'reuploading_kernel' | 'classical'.
        :param14 proxy_feature_map:        stage-A feature map ('zz' or 'z').
        :param15 proxy_n_qubits:           stage-A qubit cap.
        :param16 proxy_n_layers:           stage-A encoding layers.
        :param17 proxy_scale:              stage-A angle scaling.
        :param18 kwargs:                   QuantumModelSelector arguments.

        :return: None.

        """

        super().__init__(**kwargs)

        self.proxy_backend = str(proxy_backend).lower()
        if self.proxy_backend not in PROXY_LABELS:
            raise ValueError(f"proxy_backend must be one of {sorted(PROXY_LABELS)}")
        self.proxy_feature_map = proxy_feature_map
        self.proxy_n_qubits = int(proxy_n_qubits)
        self.proxy_n_layers = int(proxy_n_layers)
        self.proxy_scale = float(proxy_scale)

        # Stage A backend; 'classical' keeps the inherited RBF KRR
        if self.proxy_backend == "quantum_kernel":
            self.regression = QuantumKernelRidgeRegression(
                alpha=self.alpha, alpha_grid=self.alpha_grid,
                feature_map=self.proxy_feature_map,
                n_qubits=self.proxy_n_qubits,
                n_layers=self.proxy_n_layers,
                scale=self.proxy_scale,
            )
        elif self.proxy_backend == "reuploading_kernel":
            # Implicit counterpart of stage B: same encoding geometry as the circuit
            self.regression = ReuploadingKernelRidgeRegression(
                alpha=self.alpha, alpha_grid=self.alpha_grid,
                n_qubits=ru_n_qubits, n_layers=ru_n_layers, scale=ru_scale,
                random_state=ru_random_state,
            )
        self.krr_regression = self.regression
        self.proxy_agreement_ = None        # (rho, winner proxy rank, n compared)
        self.ru_regression = ReuploadingRegression(
            n_qubits=ru_n_qubits, n_layers=ru_n_layers, scale=ru_scale,
            learning_rate=ru_learning_rate, batch_size=ru_batch_size, l2=ru_l2,
            epochs=ru_selection_epochs, max_samples=ru_selection_max_samples,
            random_state=ru_random_state, patience=ru_patience,
            readout_learning_rate=ru_readout_learning_rate,
        )
        self.rescore_top_n = max(1, int(ru_rescore_top_n))

    #%% 2.1 Two-stage selection driver
    def feature_selection(self, X, y):

        """

        Stage A: inherited QUBO/QAOA search scored by the proxy. Stage B: refit
        the top-N proxy subsets with the circuit and rescore with mBIC.

        :param1 X: (n, p) reduced feature matrix.
        :param2 y: length-n target.

        :return: selection result dict; every model is a re-uploading circuit.

        """

        self.regression = self.krr_regression
        proxy = super().feature_selection(X, y)
        if not proxy['all_models']:
            return proxy

        ranked = sorted(proxy['all_models'].values(), key=lambda m: m['mbic'])
        finalists = [m['feature_indices'] for m in ranked[:self.rescore_top_n]]
        proxy_mbic = {self.key(m['feature_indices']): m['mbic'] for m in ranked}

        proxy_label = PROXY_LABELS[self.proxy_backend]
        log("Selection", f"Stage 4: re-uploading refit of the top {len(finalists)} of "
                         f"{len(ranked)} subsets ({proxy_label} proxy ranking)")
        self.regression = self.ru_regression
        all_models = {}
        self.register(all_models, self.fit_subsets(X, y, finalists, "Re-uploading finalists"))
        if not all_models:
            log("Warning", "Every re-uploading refit failed; using the proxy pool")
            return proxy

        result = self.finalise(all_models, len(y), label="re-uploading models")

        rho, winner_rank, n_cmp = rank_agreement(
            proxy_mbic, {k: m['mbic'] for k, m in all_models.items()},
        )
        self.proxy_agreement_ = (rho, winner_rank, n_cmp)
        rho_txt = "n/a" if rho is None else f"{rho:+.3f}"
        rank_txt = "n/a" if winner_rank is None else f"{winner_rank} of {len(finalists)}"
        log("Selection", f"Proxy agreement ({proxy_label}): Spearman rho {rho_txt} over "
                         f"{n_cmp} finalists; winner proxy rank {rank_txt}")
        if winner_rank is not None and winner_rank > self.rescore_top_n // 2:
            log("Warning", f"Winner ranked in the lower half of the refit band; "
                           f"consider raising rescore_top_n ({self.rescore_top_n})")

        return result
