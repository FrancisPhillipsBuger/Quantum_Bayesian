#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Exhaustive subset KRR search scored by a per-sample modified BIC:
mBIC = ln(sigma^2 / sigma_0^2) + (k_B ln n) / n + (2 gamma / n) k_S ln(p / k_S),
where k_S = d + 1 counts the subset and k_B the fitted parameters (k_B = k_S
for the KRR candidates, N_theta for a variational re-uploading finalist).

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import multiprocessing
from itertools import chain, combinations
from math import comb

import numpy as np
from joblib import Parallel, delayed

from common.kernel_ridge import KernelRidgeRegression
from common.report import log, progress


#%% 1. Selector
class ModelSelector:

    """

    Exhaustive subset KRR fitter scored by mBIC; base class of every selector.


    """

    def __init__(self, alpha=1.5, gamma_krr=None, gamma=1.0, use_nystroem=True,
                 nystroem_components=200, nystroem_random_state=0, linear=False,
                 n_jobs=None, max_features=None, alpha_grid=None):

        """

        :param1 alpha:                 KRR ridge strength.
        :param2 gamma_krr:             KRR RBF bandwidth; None → auto.
        :param3 gamma:                 mBIC sparsity penalty.
        :param4 use_nystroem:          use Nystrom KRR.
        :param5 nystroem_components:   Nystrom landmark count.
        :param6 nystroem_random_state: Nystrom RNG seed.
        :param7 linear:                pure linear ridge (no kernel).
        :param8 n_jobs:                joblib workers (None → all cores).
        :param9 max_features:          cap on subset size (None → all).
        :param10 alpha_grid:           optional ridge grid; GCV-best alpha per subset.

        :return: None.

        """

        self.alpha_grid = alpha_grid
        self.regression = KernelRidgeRegression(
            alpha=alpha, gamma=gamma_krr, alpha_grid=alpha_grid,
        )
        self.alpha = alpha
        self.gamma_krr = gamma_krr
        self.sigma_0_squared = None
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        self.gamma = gamma
        self.p = None
        self.use_nystroem = use_nystroem
        self.nystroem_components = nystroem_components
        self.nystroem_random_state = nystroem_random_state
        self.linear = linear
        self.max_features = max_features

    def shared_kwargs(self):

        """

        Constructor kwargs, used to clone this selector as a subclass.

        :return: dict of constructor kwargs.

        """

        return dict(
            alpha=self.alpha, gamma_krr=self.gamma_krr, gamma=self.gamma,
            use_nystroem=self.use_nystroem,
            nystroem_components=self.nystroem_components,
            nystroem_random_state=self.nystroem_random_state,
            linear=self.linear, n_jobs=self.n_jobs,
            max_features=self.max_features, alpha_grid=self.alpha_grid,
        )

    #%% 1.1 Scoring
    def compute_mbic(self, sigma_ml_sq, k, n, sigma_0_squared=None,
                     k_params=None):

        """

        mBIC of one model; the BIC term counts fitted parameters (k_params), the
        sparsity term counts the subset size k.

        :param1 sigma_ml_sq:     residual variance of the candidate.
        :param2 k:               subset size (incl. intercept).
        :param3 n:               sample count.
        :param4 sigma_0_squared: reference variance; None → self.sigma_0_squared.
        :param5 k_params:        free-parameter count; None → k.

        :return: scalar mBIC.

        """

        if sigma_0_squared is None:
            sigma_0_squared = (
                self.sigma_0_squared if self.sigma_0_squared is not None else sigma_ml_sq
            )
        if k_params is None:
            k_params = k
        variance_ratio = sigma_ml_sq / sigma_0_squared
        return (
            np.log(variance_ratio)
            + (k_params * np.log(n) / n)
            + (2.0 * self.gamma / n) * k * np.log(self.p / k)
        )

    def fit_single_combination(self, X, y, feature_indices):

        """

        Fit one feature subset.

        :param1 X:               (n, p) feature matrix.
        :param2 y:               length-n target.
        :param3 feature_indices: tuple of column indices.

        :return: model_info dict, or None on failure.

        """

        try:
            X_subset = X[:, list(feature_indices)]
            model_result = self.regression.fit(
                X_subset,
                y,
                feature_indices=None,
                use_nystroem=self.use_nystroem,
                nystroem_components=self.nystroem_components,
                nystroem_random_state=self.nystroem_random_state,
                linear=self.linear,
                need_predictions=False,
            )
            if model_result is not None:
                in_sample = model_result.get('sigma_ml_sq', model_result['mse'])
                return {
                    'feature_indices': feature_indices,
                    'model_result': model_result,
                    'mse': model_result['mse'],
                    # Score on the GCV variance; the raw MLE always favours larger subsets
                    'sigma_ml_sq': model_result.get('sigma_gcv_sq', in_sample),
                    'sigma_in_sample_sq': in_sample,
                    'sigma_estimator': model_result.get(
                        'sigma_estimator', 'in-sample'
                    ),
                    'alpha': model_result.get('alpha'),
                    'n_params': model_result.get('n_params'),   # None → subset size
                    'mbic': None,
                }
            return None
        except Exception:
            return None

    #%% 1.2 Shared search helpers
    @staticmethod
    def key(feature_indices):

        """Return the all_models key of a subset, e.g. '0_3_7'."""

        return '_'.join(map(str, feature_indices))

    def fit_subsets(self, X, y, subsets, desc=None, total=None):

        """

        Fit many subsets in parallel (single-thread BLAS per worker).

        :param1 X:       (n, p) feature matrix.
        :param2 y:       length-n target.
        :param3 subsets: iterable of index tuples.
        :param4 desc:    progress-bar label; None → no bar (short jobs).
        :param5 total:   subset count when subsets has no len().

        :return: list of model_info dicts (None for failed fits), in input order.

        """

        return Parallel(n_jobs=self.n_jobs, backend='loky', inner_max_num_threads=1)(
            delayed(self.fit_single_combination)(X, y, fi)
            for fi in (subsets if desc is None else progress(subsets, desc, total=total))
        )

    @classmethod
    def register(cls, all_models, results):

        """Add successful fits to all_models, keyed by their indices."""

        for model_info in results:
            if model_info is not None:
                all_models[cls.key(model_info['feature_indices'])] = model_info

    def subset_cap(self, n_features):

        """Return the largest subset size allowed by max_features."""

        if self.max_features is None:
            return n_features
        max_k = min(n_features, int(self.max_features))
        log("Selection", f"Subset size limited to {max_k} of {n_features} features")
        return max_k

    #%% 1.3 Rescoring
    def rescore(self, all_models, n, label="models"):

        """

        Set sigma_0^2, score every candidate and pick the best; also stamps
        n_samples and bic = n * mBIC (the scale BMA weights use).

        :param1 all_models: dict of model_info entries (mutated in place).
        :param2 n:          sample count of the fits.
        :param3 label:      noun used in the log line.

        :return: (best_model, best_features, best_mbic).

        """

        if not all_models:
            self.sigma_0_squared = 1.0
            log("Warning", "No model survived; nothing to rescore")
            return None, None, float('inf')

        min_sigma = min(m['sigma_ml_sq'] for m in all_models.values())
        self.sigma_0_squared = min_sigma if np.isfinite(min_sigma) else 1.0

        estimators = {m.get('sigma_estimator', 'in-sample')
                      for m in all_models.values()}
        log("Selection", f"Rescoring {len(all_models)} {label}: sigma0^2 = "
                         f"{self.sigma_0_squared:.4e} ({'/'.join(sorted(estimators))})")

        best_model, best_features, best_mbic = None, None, float('inf')
        for model_key, model_info in all_models.items():
            k = len(model_info['feature_indices']) + 1
            mbic = self.compute_mbic(
                model_info['sigma_ml_sq'], k, n, self.sigma_0_squared,
                k_params=model_info.get('n_params'),
            )
            model_info['mbic'] = mbic
            model_info['n_samples'] = int(n)
            model_info['bic'] = float(mbic) * int(n)

            model_result = model_info.get('model_result')
            if model_result is not None:
                model_result['y_pred'] = None
                model_result['residuals'] = None

            if np.isfinite(mbic):
                if mbic < best_mbic:
                    best_mbic = mbic
                    best_model = model_info
                    best_features = model_info['feature_indices']
            else:
                log("Warning", f"Invalid mBIC for subset {model_key}: {mbic}")

        return best_model, best_features, best_mbic

    @staticmethod
    def sanitise(X, y):

        """

        Replace NaN / Inf in features and targets.

        :param1 X: (n, p) feature matrix.
        :param2 y: length-n target.

        :return: (X, y) with non-finite entries substituted.

        """

        if np.isnan(X).any() or np.isinf(X).any():
            log("Warning", "Features contain NaN/Inf; replaced")
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        if np.isnan(y).any() or np.isinf(y).any():
            log("Warning", "Targets contain NaN/Inf; replaced")
            y = np.nan_to_num(y, nan=np.nanmean(y), posinf=1e10, neginf=-1e10)
        return X, y

    def finalise(self, all_models, n, label="models"):

        """

        Rescore a candidate pool and package the common selection result.

        :param1 all_models: dict of model_info entries.
        :param2 n:          sample count of the fits.
        :param3 label:      noun used in the log line.

        :return: dict {all_models, best_model, best_features, best_mbic, sigma_0_squared}.

        """

        best_model, best_features, best_mbic = self.rescore(all_models, n, label=label)
        return {
            'all_models': all_models,
            'best_model': best_model,
            'best_features': best_features,
            'best_mbic': best_mbic if best_mbic != float('inf') else None,
            'sigma_0_squared': self.sigma_0_squared,
        }

    #%% 1.4 Search driver
    def feature_selection(self, X, y):

        """

        Fit every subset of size 1..max_k and rescore with mBIC.

        :param1 X: (n, p) reduced feature matrix.
        :param2 y: length-n target.

        :return: dict {all_models, best_model, best_features, best_mbic, sigma_0_squared}.

        """

        n_features = X.shape[1]
        self.p = n_features
        all_models = {}

        X, y = self.sanitise(X, y)

        max_k = self.subset_cap(n_features)
        total = sum(comb(n_features, k) for k in range(1, max_k + 1))
        log("Selection", f"Exhaustive search: {total} subsets (k = 1-{max_k})")

        subsets = chain.from_iterable(
            combinations(range(n_features), k) for k in range(1, max_k + 1)
        )
        self.register(all_models, self.fit_subsets(X, y, subsets, "Subsets", total=total))

        if not all_models and n_features > 0:
            log("Warning", "No subset survived; retrying single features")
            singles = [(i,) for i in range(n_features)]
            self.register(all_models, self.fit_subsets(X, y, singles))

        return self.finalise(all_models, len(y), label="subsets")
