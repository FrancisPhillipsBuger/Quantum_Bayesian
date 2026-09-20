#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Bayesian model averaging over mBIC-scored subset models.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np


#%% 1. BMA
class BayesianModelAveraging:

    """

    Weighted combination of subset models; posterior weights w_i ∝ exp(-ΔBIC_i / 2)
    on the total BIC scale (BIC = n * mBIC).


    """

    @staticmethod
    def total_bic(model_info):

        """

        Return a model's BIC on the total scale.

        :param1 model_info: one entry of the selector's all_models dict.

        :return: scalar BIC (n * mBIC; falls back to mBIC for old checkpoints).

        """

        bic = model_info.get('bic')
        if bic is not None:
            return float(bic)
        return float(model_info['mbic']) * float(model_info.get('n_samples') or 1)

    @classmethod
    def bic_weights(cls, top_models):

        """

        Posterior weights exp(-ΔBIC / 2), normalised over the given models.

        :param1 top_models: list of model_info dicts.

        :return: length-k weight vector.

        """

        scores = np.array([cls.total_bic(m) for m in top_models], dtype=np.float64)
        delta = scores - np.min(scores)
        weights = np.exp(-delta / 2.0)
        return weights / np.sum(weights)

    @classmethod
    def combine(cls, X, top_models, weights):

        """

        Weighted sum of the models' predictions with the weights chosen at fit time.

        :param1 X:          (n, p) feature matrix.
        :param2 top_models: list of model_info dicts.
        :param3 weights:    length-k weight vector; None → BIC weights.

        :return: length-n prediction.

        """

        if weights is None:
            weights = cls.bic_weights(top_models)
        weights = np.asarray(weights, dtype=np.float64)
        y_pred = np.zeros(X.shape[0])
        for w, m in zip(weights, top_models):
            y_pred += w * m['model_result']['model'].predict(
                X[:, list(m['feature_indices'])]
            )
        return y_pred
