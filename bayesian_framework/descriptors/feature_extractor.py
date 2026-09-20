#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pseudospin-weighted per-atom descriptor (Artrith et al. 2017, PRB 96, 014112)
with an optional 4-body torsional block: structural and pseudospin-weighted
RDF / ADF / TDF, dimension 2 * (rdf + adf + tdf).

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

from common.arrays import to_numpy
from common.report import log
from bayesian_framework.descriptors.numba_kernels import (
    chebyshev_recursive_numba,
    compute_rdf_with_weights_numba,
    compute_adf_with_weights_numba_unified,
    compute_tdf_with_weights_numba,
)


#%% 1. Feature extractor
class FeatureExtractor:

    """

    Pseudospin-weighted RDF / ADF / (TDF) extractor; width independent of element count.


    """

    ELEMENT_MAP = {
        1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',   9: 'F',  10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi',
    }

    def __init__(self, rdf_n_basis: int = 11, adf_n_basis: int = 11,
                 tdf_n_basis: int = 0,
                 cutoff_radius: float = 6.0, species: Optional[List[int]] = None,
                 use_pbc: bool = True, pbc: Optional[List[bool]] = None,
                 n_jobs: int = 1):

        """

        :param1 rdf_n_basis:   RDF Chebyshev basis count.
        :param2 adf_n_basis:   ADF Chebyshev basis count.
        :param3 tdf_n_basis:   TDF basis count (0 disables 4-body term).
        :param4 cutoff_radius: cut-off radius in Å.
        :param5 species:       optional iterable of atomic numbers; None → auto-detect.
        :param6 use_pbc:       enable PBC distance evaluation.
        :param7 pbc:           3-bool list for periodic directions.
        :param8 n_jobs:        joblib worker count (threads back-end; numba releases GIL).

        :return: None.

        """

        self.rdf_n_basis = rdf_n_basis
        self.adf_n_basis = adf_n_basis
        self.tdf_n_basis = tdf_n_basis
        self.cutoff_radius = cutoff_radius
        self.n_jobs = n_jobs
        self.use_pbc = use_pbc
        self.pbc = np.array(pbc if pbc is not None else [True, True, True], dtype=bool)

        if species is not None:
            filtered = [z for z in species if z in self.ELEMENT_MAP]
            ignored  = [z for z in species if z not in self.ELEMENT_MAP]
            if ignored:
                log("Warning", f"Ignoring elements outside ELEMENT_MAP: {ignored}")
            self.species = sorted(filtered)
        else:
            self.species = None

        self.pseudospin_weights = None
        self.weights_map = None
        self.initialized = self.species is not None

        if self.initialized:
            self.setup_pseudospin_weights()

        self.pi_rc = np.pi / cutoff_radius

    #%% 1.1 Pseudospin weight setup
    def setup_pseudospin_weights(self):

        """

        Assign Ising-style pseudospin weights {-h, ..., h} (0 dropped for even counts).

        :return: None.

        """

        n_species = len(self.species)
        if n_species % 2 == 1:
            half = n_species // 2
            weights = list(range(-half, half + 1))
        else:
            half = n_species // 2
            weights = list(range(-half, 0)) + list(range(1, half + 1))

        self.pseudospin_weights = {z: w for z, w in zip(self.species, weights)}

        # Dense Z -> weight lookup over the whole ELEMENT_MAP for numba
        max_z = max(self.ELEMENT_MAP.keys())
        self.weights_map = np.zeros(max_z + 1, dtype=np.float64)
        for z, w in self.pseudospin_weights.items():
            self.weights_map[z] = w

    def get_weights(self, use_weights: bool) -> np.ndarray:

        """Return the pseudospin weight lookup, or all-ones when unweighted."""

        if use_weights:
            return self.weights_map
        return np.ones(len(self.weights_map), dtype=np.float64)

    #%% 1.2 Basis / cut-off helpers
    def cutoff_function(self, r: np.ndarray) -> np.ndarray:

        """

        Cosine cut-off f_c(r) = 0.5 * (cos(pi r / R_c) + 1) for r < R_c, else 0
        (the paper's "cos(...) - 1" is a typo).

        :param1 r: input distances.

        :return: cut-off values.

        """

        return np.where(
            r < self.cutoff_radius,
            0.5 * (np.cos(self.pi_rc * r) + 1.0),
            0.0
        )

    def chebyshev_basis(self, x: np.ndarray, n_max: int,
                        x_min: float = 0.0, x_max: float = 1.0) -> np.ndarray:

        """

        Evaluate the first n_max Chebyshev polynomials on rescaled inputs.

        :param1 x:     input values.
        :param2 n_max: basis-function count.
        :param3 x_min: lower bound of the input range.
        :param4 x_max: upper bound of the input range.

        :return: (len(x), n_max) basis-value array.

        """

        x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        x_scaled = np.clip(x_scaled, -1.0, 1.0)
        return chebyshev_recursive_numba(x_scaled, n_max)

    def compute_distance_matrix_fast(self, pos: np.ndarray) -> np.ndarray:

        """

        Compute the non-periodic pairwise distance matrix.

        :param1 pos: (n_atoms, 3) positions.

        :return: (n_atoms, n_atoms) distance matrix.

        """

        n_atoms = len(pos)
        if n_atoms <= 1:
            return np.zeros((n_atoms, n_atoms))
        try:
            distances = squareform(pdist(pos, metric='euclidean'))
        except Exception:
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
        return distances

    def compute_distance_vectors_pbc(self, pos: np.ndarray, cell: np.ndarray,
                                     pbc: np.ndarray) -> np.ndarray:

        """

        Compute minimum-image distance vectors under periodic boundaries.

        :param1 pos:  (n_atoms, 3) positions.
        :param2 cell: (3, 3) lattice matrix.
        :param3 pbc:  3-bool periodic-direction mask.

        :return: (n_atoms, n_atoms, 3) distance-vector array.

        """

        inv_cell = np.linalg.inv(cell)
        frac = pos @ inv_cell
        delta_frac = frac[np.newaxis, :, :] - frac[:, np.newaxis, :]
        for dim in range(3):
            if pbc[dim]:
                delta_frac[..., dim] -= np.round(delta_frac[..., dim])
        return np.matmul(delta_frac, cell)

    #%% 1.3 Single descriptor blocks
    def compute_rdf_features(self, distances: np.ndarray, fc_matrix: np.ndarray,
                             atomic_numbers: np.ndarray,
                             use_weights: bool = False) -> np.ndarray:

        """

        Compute per-atom RDF descriptor blocks.

        :param1 distances:      (n, n) distance matrix.
        :param2 fc_matrix:      (n, n) cut-off function values.
        :param3 atomic_numbers: (n,) atomic numbers.
        :param4 use_weights:    apply pseudospin weights (compositional block).

        :return: (n, rdf_n_basis) feature block.

        """

        n_atoms = distances.shape[0]

        mask = (distances > 0) & (distances < self.cutoff_radius)
        if np.sum(mask) > 0:
            valid_basis = self.chebyshev_basis(
                distances[mask], self.rdf_n_basis,
                x_min=0.0, x_max=self.cutoff_radius
            )
            basis_all = np.zeros((n_atoms, n_atoms, self.rdf_n_basis))
            basis_all[mask] = valid_basis

            max_z = len(self.weights_map) - 1
            weights = self.get_weights(use_weights)
            return compute_rdf_with_weights_numba(
                distances, fc_matrix, atomic_numbers, weights,
                basis_all, self.rdf_n_basis, self.cutoff_radius, max_z
            )
        return np.zeros((n_atoms, self.rdf_n_basis))

    def compute_adf_features(self, pos: np.ndarray, distances: np.ndarray,
                             fc_matrix: np.ndarray, atomic_numbers: np.ndarray,
                             use_weights: bool = False,
                             distance_vectors: Optional[np.ndarray] = None) -> np.ndarray:

        """

        Compute per-atom ADF (3-body angular) descriptor blocks.

        :param1 pos:              (n, 3) positions.
        :param2 distances:        (n, n) distance matrix.
        :param3 fc_matrix:        (n, n) cut-off function values.
        :param4 atomic_numbers:   (n,) atomic numbers.
        :param5 use_weights:      apply pseudospin weights (compositional block).
        :param6 distance_vectors: optional precomputed (n, n, 3) vectors.

        :return: (n, adf_n_basis) feature block.

        """

        max_z = len(self.weights_map) - 1
        weights = self.get_weights(use_weights)
        if distance_vectors is None:
            distance_vectors = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        return compute_adf_with_weights_numba_unified(
            distances, distance_vectors, fc_matrix, atomic_numbers, weights,
            self.adf_n_basis, self.cutoff_radius, max_z
        )

    def compute_tdf_features(self, pos: np.ndarray, distances: np.ndarray,
                             fc_matrix: np.ndarray, atomic_numbers: np.ndarray,
                             use_weights: bool = False,
                             distance_vectors: Optional[np.ndarray] = None) -> np.ndarray:

        """

        Compute per-atom TDF (4-body torsional) descriptor blocks.

        :param1 pos:              (n, 3) positions.
        :param2 distances:        (n, n) distance matrix.
        :param3 fc_matrix:        (n, n) cut-off function values.
        :param4 atomic_numbers:   (n,) atomic numbers.
        :param5 use_weights:      apply pseudospin weights (compositional block).
        :param6 distance_vectors: optional precomputed (n, n, 3) vectors.

        :return: (n, tdf_n_basis) feature block.

        """

        max_z = len(self.weights_map) - 1
        weights = self.get_weights(use_weights)
        if distance_vectors is None:
            distance_vectors = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        return compute_tdf_with_weights_numba(
            distances, distance_vectors, fc_matrix, atomic_numbers, weights,
            self.tdf_n_basis, self.cutoff_radius, max_z
        )

    #%% 1.4 Per-molecule and batch entry points
    def extract_single_molecule(self, pos: np.ndarray, atomic_numbers: np.ndarray,
                                cell: Optional[np.ndarray] = None,
                                pbc: Optional[np.ndarray] = None,
                                tags: Optional[np.ndarray] = None):

        """

        Build the full per-atom descriptor for one molecule.

        :param1 pos:            (n_atoms, 3) positions.
        :param2 atomic_numbers: (n_atoms,) atomic numbers.
        :param3 cell:           optional (3, 3) lattice.
        :param4 pbc:            optional 3-bool array.
        :param5 tags:           optional OC22 site tags (2 = adsorbate).

        :return: (atom_features (n_atoms, 2*(rdf+adf+tdf)), ads_dist (n_atoms,) distance
                 to the nearest adsorbate atom, or None without adsorbate).

        """

        n_atoms = len(pos)
        if n_atoms == 0:
            return np.zeros((0, self.get_n_features())), None

        use_pbc = self.use_pbc and cell is not None
        if use_pbc:
            pbc_mask = self.pbc if pbc is None else pbc
            distance_vectors = self.compute_distance_vectors_pbc(pos, cell, pbc_mask)
            distances = np.linalg.norm(distance_vectors, axis=2)
        else:
            distance_vectors = None
            distances = self.compute_distance_matrix_fast(pos)

        fc_matrix = self.cutoff_function(distances)

        rdf_structural = self.compute_rdf_features(
            distances, fc_matrix, atomic_numbers, use_weights=False
        )
        adf_structural = self.compute_adf_features(
            pos, distances, fc_matrix, atomic_numbers,
            use_weights=False, distance_vectors=distance_vectors
        )
        rdf_compositional = self.compute_rdf_features(
            distances, fc_matrix, atomic_numbers, use_weights=True
        )
        adf_compositional = self.compute_adf_features(
            pos, distances, fc_matrix, atomic_numbers,
            use_weights=True, distance_vectors=distance_vectors
        )

        feature_blocks = [
            rdf_structural,
            rdf_compositional,
            adf_structural,
            adf_compositional,
        ]

        if self.tdf_n_basis > 0:
            if distance_vectors is None:
                distance_vectors = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
            tdf_structural = self.compute_tdf_features(
                pos, distances, fc_matrix, atomic_numbers,
                use_weights=False, distance_vectors=distance_vectors
            )
            tdf_compositional = self.compute_tdf_features(
                pos, distances, fc_matrix, atomic_numbers,
                use_weights=True, distance_vectors=distance_vectors
            )
            feature_blocks.append(tdf_structural)
            feature_blocks.append(tdf_compositional)

        atom_features = np.concatenate(feature_blocks, axis=1)

        if np.isnan(atom_features).any() or np.isinf(atom_features).any():
            atom_features = np.nan_to_num(atom_features, nan=0.0, posinf=1e10, neginf=-1e10)

        ads_dist = None
        if tags is not None:
            ads_mask = np.asarray(tags).ravel() == 2
            if ads_mask.any():
                ads_dist = distances[:, ads_mask].min(axis=1)

        return atom_features, ads_dist

    def initialize_from_batch(self, atomic_numbers_np: np.ndarray):

        """

        Set the species list from atomic numbers and build pseudospin weights.

        :param1 atomic_numbers_np: (n,) atomic numbers.

        :return: None.

        """

        if self.initialized:
            return

        unique_elements = np.unique(atomic_numbers_np)
        self.species = sorted([z for z in unique_elements.tolist() if z in self.ELEMENT_MAP])
        ignored = [z for z in unique_elements.tolist() if z not in self.ELEMENT_MAP]
        if ignored:
            log("Warning", f"Ignoring elements outside ELEMENT_MAP: {ignored}")
        if not self.species:
            raise ValueError("No valid species detected.")

        self.setup_pseudospin_weights()
        self.initialized = True

    def extract_features(self, batch):

        """

        Extract per-atom features for a PyG batch (or single sample).

        :param1 batch: PyG Batch or Data object.

        :return: (per-molecule (n_atoms_i, F) features, atomic-number arrays,
                 nearest-adsorbate distances or None).

        """

        pos_np = to_numpy(batch.pos)
        try:
            atomic_numbers_np = to_numpy(batch.atomic_numbers)
            if not np.issubdtype(atomic_numbers_np.dtype, np.integer):
                atomic_numbers_np = np.round(atomic_numbers_np).astype(np.int64)
        except Exception:
            atomic_numbers_np = np.ones(len(pos_np), dtype=np.int64)
        atomic_numbers_np = atomic_numbers_np.ravel()

        self.initialize_from_batch(atomic_numbers_np)

        cell_np = to_numpy(batch.cell) if hasattr(batch, 'cell') else None
        pbc_np = to_numpy(batch.pbc) if hasattr(batch, 'pbc') else None
        tags_np = to_numpy(batch.tags).ravel() if hasattr(batch, 'tags') else None

        if hasattr(batch, 'batch') and batch.batch is not None:
            batch_indices_np = to_numpy(batch.batch).astype(np.int64).ravel()
            batch_size = int(batch_indices_np.max()) + 1

            # PyG stores atoms contiguously per molecule: one O(N) split
            counts = np.bincount(batch_indices_np, minlength=batch_size)
            split_points = np.cumsum(counts)[:-1]
            pos_split = np.split(pos_np, split_points)
            atomic_numbers_split = np.split(atomic_numbers_np, split_points)
            tags_split = (
                np.split(tags_np, split_points) if tags_np is not None
                else [None] * batch_size
            )

            mol_args = []
            for i in range(batch_size):
                cell_i = None
                if cell_np is not None:
                    cell_i = cell_np[i] if cell_np.ndim == 3 else cell_np
                pbc_i = None
                if pbc_np is not None:
                    pbc_i = pbc_np[i] if pbc_np.ndim == 2 else pbc_np
                mol_args.append((pos_split[i], atomic_numbers_split[i], cell_i, pbc_i, tags_split[i]))

            if self.n_jobs != 1 and batch_size > 1:
                # numba releases the GIL, so threads suffice
                results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(self.extract_single_molecule)(p, z, cell=c, pbc=pb, tags=t)
                    for p, z, c, pb, t in mol_args
                )
            else:
                results = [
                    self.extract_single_molecule(p, z, cell=c, pbc=pb, tags=t)
                    for p, z, c, pb, t in mol_args
                ]

            all_atom_features = [r[0] for r in results]
            all_ads_dist = [r[1] for r in results]
            return all_atom_features, atomic_numbers_split, all_ads_dist

        atom_features, ads_dist = self.extract_single_molecule(
            pos_np, atomic_numbers_np, cell=cell_np, pbc=pbc_np, tags=tags_np
        )
        return atom_features, [atomic_numbers_np], [ads_dist]

    #%% 1.5 Feature metadata
    def get_n_features(self) -> int:

        """

        Descriptor width.

        :return: 2 * (rdf_n_basis + adf_n_basis + tdf_n_basis).

        """

        if not self.initialized:
            raise ValueError("FeatureExtractor not initialised.")
        return 2 * (self.rdf_n_basis + self.adf_n_basis + self.tdf_n_basis)
