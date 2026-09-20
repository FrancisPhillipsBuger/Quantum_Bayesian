#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Numba-compiled inner loops for pseudospin-weighted RDF / ADF / TDF
descriptors (Artrith et al. 2017, PRB 96, 014112).

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import warnings

import numpy as np

try:
    from numba import njit, prange
except ImportError as _numba_exc:                      # noqa: N816

    def njit(func=None, parallel=False):

        """No-op numba.njit stand-in used when numba is not importable."""

        if func is None:
            return lambda f: f
        return func

    prange = range

    warnings.warn(
        f"numba unavailable ({_numba_exc}); using pure-Python descriptor kernels (~100-1000x slower).",
        RuntimeWarning, stacklevel=2,
    )


#%% 1. Chebyshev basis (T_n recurrence)
@njit
def chebyshev_recursive_numba(x: np.ndarray, n_max: int) -> np.ndarray:

    """

    Chebyshev first-kind polynomials T_0..T_{n_max-1} via three-term recurrence.

    :param1 x:     (n,) input values, assumed pre-mapped to [-1, 1].
    :param2 n_max: number of basis terms.

    :return: (n, n_max) basis matrix.

    """

    basis = np.zeros((len(x), n_max))

    if n_max > 0:
        basis[:, 0] = 1.0
    if n_max > 1:
        basis[:, 1] = x

    for n in range(2, n_max):
        basis[:, n] = 2.0 * x * basis[:, n-1] - basis[:, n-2]

    return basis


#%% 2. RDF (2-body)
@njit(parallel=True)
def compute_rdf_with_weights_numba(distances: np.ndarray, fc_matrix: np.ndarray,
                                   atomic_numbers: np.ndarray, weights_map: np.ndarray,
                                   basis_all: np.ndarray, rdf_n_basis: int,
                                   cutoff_radius: float, max_z: int) -> np.ndarray:

    """

    Per-atom RDF: c_alpha^(2) = sum_j phi_alpha(R_ij) * f_c(R_ij) * w_{t_j}.

    :param1 distances:      (n_atoms, n_atoms).
    :param2 fc_matrix:      (n_atoms, n_atoms) smooth cut-off values.
    :param3 atomic_numbers: (n_atoms,) atomic numbers.
    :param4 weights_map:    (max_z+1,) Z → pseudospin weight lookup.
    :param5 basis_all:      (n_atoms, n_atoms, rdf_n_basis) Chebyshev values.
    :param6 rdf_n_basis:    basis count.
    :param7 cutoff_radius:  cut-off radius (Å).
    :param8 max_z:          maximum supported atomic number.

    :return: (n_atoms, rdf_n_basis) RDF features.

    """

    n_atoms = distances.shape[0]
    rdf_features = np.zeros((n_atoms, rdf_n_basis))

    for i in prange(n_atoms):
        for j in range(n_atoms):
            if i != j and distances[i, j] < cutoff_radius:
                Z_j = atomic_numbers[j]
                if Z_j <= max_z:
                    w_t = weights_map[Z_j]
                    fc_val = fc_matrix[i, j]
                    for k in range(rdf_n_basis):
                        rdf_features[i, k] += basis_all[i, j, k] * fc_val * w_t

    return rdf_features


#%% 3. ADF (3-body)
@njit(parallel=True)
def compute_adf_with_weights_numba_unified(distances: np.ndarray, distance_vectors: np.ndarray,
                                           fc_matrix: np.ndarray, atomic_numbers: np.ndarray,
                                           weights_map: np.ndarray, adf_n_basis: int,
                                           cutoff_radius: float, max_z: int) -> np.ndarray:

    """

    Per-atom ADF: c_alpha^(3) = sum_{j!=i, k!=i,j} phi(theta_ijk) f_c(R_ij) f_c(R_ik) w_j w_k.

    :param1 distances:        (n_atoms, n_atoms).
    :param2 distance_vectors: (n_atoms, n_atoms, 3) with [i,j] = r_j - r_i.
    :param3 fc_matrix:        cut-off matrix.
    :param4 atomic_numbers:   (n_atoms,).
    :param5 weights_map:      (max_z+1,) pseudospin weights.
    :param6 adf_n_basis:      basis count.
    :param7 cutoff_radius:    cut-off radius (Å).
    :param8 max_z:            maximum supported Z.

    :return: (n_atoms, adf_n_basis).

    """

    n_atoms = distances.shape[0]
    adf_features = np.zeros((n_atoms, adf_n_basis))

    for i in prange(n_atoms):
        for j in range(n_atoms):
            if j == i or distances[i, j] >= cutoff_radius:
                continue
            Z_j = atomic_numbers[j]
            if Z_j > max_z:
                continue
            fc_ij = fc_matrix[i, j]
            w_j = weights_map[Z_j]

            for k in range(n_atoms):
                if k == i or k == j or distances[i, k] >= cutoff_radius:
                    continue
                Z_k = atomic_numbers[k]
                if Z_k > max_z:
                    continue
                fc_ik = fc_matrix[i, k]
                w_k = weights_map[Z_k]

                dot = 0.0
                for d in range(3):
                    dot += distance_vectors[i, j, d] * distance_vectors[i, k, d]
                cos_theta = dot / (distances[i, j] * distances[i, k])
                if cos_theta > 1.0:
                    cos_theta = 1.0
                elif cos_theta < -1.0:
                    cos_theta = -1.0

                w = fc_ij * fc_ik * w_j * w_k

                # Chebyshev recurrence T_{n+1} = 2x T_n - T_{n-1}
                if adf_n_basis > 0:
                    adf_features[i, 0] += w
                if adf_n_basis > 1:
                    adf_features[i, 1] += cos_theta * w
                if adf_n_basis > 2:
                    t_prev2 = 1.0
                    t_prev1 = cos_theta
                    for n in range(2, adf_n_basis):
                        t_curr = 2.0 * cos_theta * t_prev1 - t_prev2
                        adf_features[i, n] += t_curr * w
                        t_prev2 = t_prev1
                        t_prev1 = t_curr

    return adf_features


#%% 4. TDF (4-body dihedrals)
@njit(parallel=True)
def compute_tdf_with_weights_numba(distances: np.ndarray, distance_vectors: np.ndarray,
                                   fc_matrix: np.ndarray, atomic_numbers: np.ndarray,
                                   weights_map: np.ndarray, tdf_n_basis: int,
                                   cutoff_radius: float, max_z: int) -> np.ndarray:

    """

    Per-atom TDF over chains i-j-k-l inside the cut-off, with the dihedral
    cos(phi) = (n1 . n2) / (|n1| |n2|), n1 = b1 x b2, n2 = b2 x b3.

    :param1 distances:        (n_atoms, n_atoms).
    :param2 distance_vectors: (n_atoms, n_atoms, 3) with [i,j] = r_j - r_i.
    :param3 fc_matrix:        cut-off matrix.
    :param4 atomic_numbers:   (n_atoms,).
    :param5 weights_map:      (max_z+1,) pseudospin weights.
    :param6 tdf_n_basis:      basis count.
    :param7 cutoff_radius:    cut-off radius (Å).
    :param8 max_z:            maximum supported Z.

    :return: (n_atoms, tdf_n_basis).

    """

    n_atoms = distances.shape[0]
    tdf_features = np.zeros((n_atoms, tdf_n_basis))

    for i in prange(n_atoms):
        for j in range(n_atoms):
            if j == i or distances[i, j] >= cutoff_radius:
                continue
            Z_j = atomic_numbers[j]
            if Z_j > max_z:
                continue
            fc_ij = fc_matrix[i, j]
            w_j = weights_map[Z_j]

            b1x = distance_vectors[i, j, 0]
            b1y = distance_vectors[i, j, 1]
            b1z = distance_vectors[i, j, 2]

            for k in range(n_atoms):
                if k == i or k == j or distances[j, k] >= cutoff_radius:
                    continue
                Z_k = atomic_numbers[k]
                if Z_k > max_z:
                    continue
                fc_jk = fc_matrix[j, k]
                w_k = weights_map[Z_k]

                b2x = distance_vectors[j, k, 0]
                b2y = distance_vectors[j, k, 1]
                b2z = distance_vectors[j, k, 2]

                n1x = b1y * b2z - b1z * b2y
                n1y = b1z * b2x - b1x * b2z
                n1z = b1x * b2y - b1y * b2x
                n1_norm_sq = n1x * n1x + n1y * n1y + n1z * n1z
                if n1_norm_sq < 1e-12:
                    # i-j-k collinear: dihedral undefined
                    continue
                n1_norm = np.sqrt(n1_norm_sq)

                for l in range(n_atoms):
                    if l == i or l == j or l == k or distances[k, l] >= cutoff_radius:
                        continue
                    Z_l = atomic_numbers[l]
                    if Z_l > max_z:
                        continue
                    fc_kl = fc_matrix[k, l]
                    w_l = weights_map[Z_l]

                    b3x = distance_vectors[k, l, 0]
                    b3y = distance_vectors[k, l, 1]
                    b3z = distance_vectors[k, l, 2]

                    n2x = b2y * b3z - b2z * b3y
                    n2y = b2z * b3x - b2x * b3z
                    n2z = b2x * b3y - b2y * b3x
                    n2_norm_sq = n2x * n2x + n2y * n2y + n2z * n2z
                    if n2_norm_sq < 1e-12:
                        continue
                    n2_norm = np.sqrt(n2_norm_sq)

                    cos_phi = (n1x * n2x + n1y * n2y + n1z * n2z) / (n1_norm * n2_norm)
                    if cos_phi > 1.0:
                        cos_phi = 1.0
                    elif cos_phi < -1.0:
                        cos_phi = -1.0

                    w = fc_ij * fc_jk * fc_kl * w_j * w_k * w_l

                    if tdf_n_basis > 0:
                        tdf_features[i, 0] += w
                    if tdf_n_basis > 1:
                        tdf_features[i, 1] += cos_phi * w
                    if tdf_n_basis > 2:
                        t_prev2 = 1.0
                        t_prev1 = cos_phi
                        for n in range(2, tdf_n_basis):
                            t_curr = 2.0 * cos_phi * t_prev1 - t_prev2
                            tdf_features[i, n] += t_curr * w
                            t_prev2 = t_prev1
                            t_prev1 = t_curr

    return tdf_features
