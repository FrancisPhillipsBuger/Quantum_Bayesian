#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Supervised (PLS) reducer exposing the PCA attributes the framework uses.

Created on: Wed Aug 20 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from sklearn.cross_decomposition import PLSRegression


#%% 1. PLS reducer
class PLSReducer:

    """

    Partial-least-squares reducer with transform(), n_components_ and components_.


    """

    __slots__ = ("pls", "n_components_", "components_", "explained_variance_ratio_")

    def __init__(self, pls, explained_variance_ratio=None):

        """

        Wrap a fitted PLSRegression.

        :param1 pls:                      fitted sklearn PLSRegression.
        :param2 explained_variance_ratio: optional per-component input-variance share.

        :return: None.

        """

        self.pls = pls
        self.components_ = np.asarray(pls.x_rotations_).T
        self.n_components_ = self.components_.shape[0]
        self.explained_variance_ratio_ = (
            np.zeros(self.n_components_) if explained_variance_ratio is None
            else np.asarray(explained_variance_ratio)
        )

    @classmethod
    def fit(cls, X_scaled, y, n_components):

        """

        Fit PLS on standardised features.

        :param1 X_scaled:     (n, d) standardised feature matrix.
        :param2 y:            length-n target the components should track.
        :param3 n_components: number of latent components to keep.

        :return: (PLSReducer, (n, n_components) scores).

        """

        n_components = int(min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1))
        n_components = max(n_components, 1)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_scaled, np.asarray(y, dtype=np.float64).ravel())
        scores = pls.transform(X_scaled)

        total_var = float(np.sum(np.var(X_scaled, axis=0)))
        ratio = (np.var(scores, axis=0) / total_var) if total_var > 0 else None
        return cls(pls, explained_variance_ratio=ratio), scores

    def transform(self, X):

        """

        Project standardised features onto the fitted latent components.

        :param1 X: (n, d) standardised feature matrix.

        :return: (n, n_components) scores.

        """

        return self.pls.transform(X)
