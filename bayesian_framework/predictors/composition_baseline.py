#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

RidgeCV energy baseline on per-element atom counts, subtracted before the
kernel framework is trained on the residual.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from sklearn.linear_model import RidgeCV

from common.report import log


#%% 1. Baseline learner
class CompositionBaseline:

    """

    Atomic-count → energy RidgeCV baseline.


    """

    def __init__(self, alphas=None):

        """

        :param1 alphas: RidgeCV alpha grid; None → logspace(-6, 6, 25).

        :return: None.

        """

        self.alphas = alphas
        self.model = None
        self.species = None

    def fit(self, mol_atomic_numbers, y):

        """

        Fit RidgeCV on the (n_samples, n_species) count matrix.

        :param1 mol_atomic_numbers: list of (n_atoms_i,) atomic-number arrays.
        :param2 y:                  length-n target vector.

        :return: length-n in-sample prediction.

        """

        X_counts = self.counts(mol_atomic_numbers, fit=True)
        alphas = self.alphas if self.alphas is not None else np.logspace(-6, 6, 25)
        self.model = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))
        self.model.fit(X_counts, y)
        y_pred = self.model.predict(X_counts)

        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y - y_pred)))
        r2 = float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        log("Baseline", f"Composition RidgeCV: R2 {r2:.4f} | RMSE {rmse:.4f} eV | "
                        f"MAE {mae:.4f} eV | alpha {self.model.alpha_:.4g}")
        return y_pred

    def predict(self, mol_atomic_numbers):

        """

        Predict baseline energies; zeros when untrained.

        :param1 mol_atomic_numbers: list of (n_atoms_i,) atomic-number arrays.

        :return: length-n prediction.

        """

        if self.model is None:
            return np.zeros(len(mol_atomic_numbers), dtype=np.float64)
        return self.model.predict(self.counts(mol_atomic_numbers))

    def counts(self, mol_atomic_numbers, fit=False):

        """

        Build the atom-count matrix; unseen elements count as the nearest trained one.

        :param1 mol_atomic_numbers: list of (n_atoms_i,) atomic-number arrays.
        :param2 fit:                refresh self.species from the data.

        :return: (n_samples, n_species) count matrix.

        """

        if fit or self.species is None:
            self.species = sorted({int(z) for arr in mol_atomic_numbers
                                   for z in np.asarray(arr).ravel()})

        index = {int(z): i for i, z in enumerate(self.species)}
        known = np.asarray(self.species, dtype=np.int64)
        n_rows = len(mol_atomic_numbers)
        counts = np.zeros((n_rows, len(self.species)), dtype=np.float64)
        unknown_rows = np.zeros(n_rows, dtype=bool)
        unknown = set()
        for row, arr in enumerate(mol_atomic_numbers):
            for z in np.asarray(arr).astype(np.int64).ravel():
                z = int(z)
                col = index.get(z)
                if col is None:
                    unknown.add(z)
                    unknown_rows[row] = True
                    if len(known):
                        col = int(np.argmin(np.abs(known - z)))
                    else:
                        continue
                counts[row, col] += 1.0
        if unknown:
            substitutes = {
                z: int(known[int(np.argmin(np.abs(known - z)))]) for z in sorted(unknown)
            } if len(known) else {}
            log("Warning", f"Composition baseline: unseen elements {sorted(unknown)} in "
                           f"{int(unknown_rows.sum())} samples mapped to {substitutes}")
        return counts
