#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Atom-level Nystrom RBF feature map with per-molecule sum pooling
(Artrith et al. 2017): Phi(A) = sum_{i in A} phi(sigma_i).

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

from common.kernel_ridge import scale_gamma
from common.report import log


#%% 1. Pooler
class AtomicNystroemPooler:

    """

    StandardScaler → Nystrom(RBF) → per-molecule sum pool. With ads_decay_length set,
    a second block sum_i exp(-d_i / lambda) phi_i (d_i = distance to the nearest
    adsorbate atom) is concatenated to the global pool.


    """

    def __init__(self, n_components: int = 2048, random_state: int = 0,
                 ads_decay_length: float = None):

        """

        :param1 n_components:     Nystrom landmark count.
        :param2 random_state:     RNG seed.
        :param3 ads_decay_length: lambda (Å) of the adsorbate-distance weight;
                                  None → global pool only (width n_components).

        :return: None.

        """

        self.n_components = int(n_components)
        self.random_state = random_state
        self.ads_decay_length = (
            float(ads_decay_length) if ads_decay_length is not None else None
        )
        self.scaler = None
        self.nystroem = None
        self.transform_atom_budget = 200_000

    def fit(self, mol_atoms_list, ads_dist_list=None):

        """

        Fit scaler + Nystrom on all atoms and return the pooled molecules.

        :param1 mol_atoms_list: list of (n_atoms_i, n_descriptor) arrays.
        :param2 ads_dist_list:  optional list of (n_atoms_i,) adsorbate distances (or None).

        :return: (n_molecules, n_components [x2 with ads_decay_length]) matrix.

        """

        nonempty = [m for m in mol_atoms_list if m.ndim == 2 and m.shape[0] > 0]
        if not nonempty:
            raise ValueError("Empty atom-pool fit: every molecule has zero atoms")
        X_atoms = np.vstack(nonempty)

        self.scaler = StandardScaler()
        X_atoms_scaled = self.scaler.fit_transform(X_atoms)
        gamma = scale_gamma(X_atoms_scaled)

        n_comp = min(self.n_components, X_atoms_scaled.shape[0])
        self.nystroem = Nystroem(
            kernel='rbf', gamma=gamma, n_components=n_comp, random_state=self.random_state,
        )
        self.nystroem.fit(X_atoms_scaled)
        mode = ("global + adsorbate-weighted" if self.ads_decay_length is not None
                else "global sum")
        log("Pooling", f"Nystrom RBF: {X_atoms.shape[0]} atoms -> {n_comp} features "
                       f"(gamma {gamma:.4g}, {mode})")

        return self.transform(mol_atoms_list, ads_dist_list)

    def transform(self, mol_atoms_list, ads_dist_list=None):

        """

        Pool a list of per-atom feature matrices in atom-budgeted blocks.

        :param1 mol_atoms_list: list of (n_atoms_i, n_descriptor) arrays.
        :param2 ads_dist_list:  optional list of (n_atoms_i,) adsorbate distances.

        :return: (n_molecules, n_components [x2 with ads_decay_length]) matrix.

        """

        if self.nystroem is None or self.scaler is None:
            raise ValueError("AtomicNystroemPooler not fitted; call fit() first.")
        n_comp = self.nystroem.components_.shape[0]
        use_local = self.ads_decay_length is not None
        out_dim = n_comp * (2 if use_local else 1)
        n_mol = len(mol_atoms_list)
        pooled = np.zeros((n_mol, out_dim), dtype=np.float64)
        atom_budget = max(int(self.transform_atom_budget), n_comp)
        start = 0
        while start < n_mol:
            counts, total, end = [], 0, start
            while end < n_mol:
                mat = np.asarray(mol_atoms_list[end], dtype=np.float64)
                k = mat.shape[0] if mat.ndim == 2 else 0
                if counts and total + k > atom_budget:
                    break
                counts.append(k)
                total += k
                end += 1

            if total == 0:
                start = end
                continue

            nz = [j for j in range(start, end) if counts[j - start] > 0]
            block = np.vstack([
                np.asarray(mol_atoms_list[j], dtype=np.float64) for j in nz
            ])
            phi = self.nystroem.transform(self.scaler.transform(block))
            offs = np.cumsum([0] + [counts[j - start] for j in nz])

            if use_local:
                w = np.ones(total, dtype=np.float64)    # no adsorbate → weight 1
                for pos, j in enumerate(nz):
                    d = None if ads_dist_list is None else ads_dist_list[j]
                    if d is not None:
                        w[offs[pos]:offs[pos + 1]] = np.exp(
                            -np.asarray(d, dtype=np.float64).ravel()
                            / self.ads_decay_length
                        )
                phi_w = phi * w[:, None]

            for pos, j in enumerate(nz):
                a, b = int(offs[pos]), int(offs[pos + 1])
                g = phi[a:b].sum(axis=0)
                pooled[j] = (np.concatenate([g, phi_w[a:b].sum(axis=0)])
                             if use_local else g)
            start = end
        return pooled
