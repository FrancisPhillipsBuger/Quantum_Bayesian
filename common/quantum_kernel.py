#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Fidelity quantum kernel K(x, y) = |<phi(x)|phi(y)>|^2 with the ZZ (or Z) feature
map, simulated exactly as batched numpy statevectors.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from math import ceil

import numpy as np


#%% 1. Shared statevector primitives
def ring_pairs(q):

    """

    Entangling-ring qubit pairs, shared by the ZZ map and the re-uploading circuit.

    :param1 q: qubit count.

    :return: list of (control, target) index pairs.

    """

    if q < 2:
        return []
    if q == 2:
        return [(0, 1)]
    return [(i, (i + 1) % q) for i in range(q)]


def apply_1q(psi, U, t, n, q, dim):

    """

    Apply a 1-qubit gate to qubit t of a batched statevector.

    :param1 psi: (n, 2^q) complex statevector batch.
    :param2 U:   (n, 2, 2) per-row or (2, 2) shared gate matrix.
    :param3 t:   target qubit (qiskit little-endian order).
    :param4 n:   batch size.
    :param5 q:   qubit count.
    :param6 dim: 2^q.

    :return: (n, dim) updated statevector batch.

    """

    hi, lo = 1 << (q - 1 - t), 1 << t
    v = psi.reshape(n, hi, 2, lo)
    sub = 'nab' if U.ndim == 3 else 'ab'
    return np.einsum(f'{sub},nhbl->nhal', U, v, optimize=True).reshape(n, dim)


class FidelityKernel:

    """

    Gram matrix k(x, y) = |<psi(x)|psi(y)>|^2 over any statevector encoding;
    subclasses implement statevectors(X).


    """

    def statevectors(self, X):

        """Map a batch of inputs to statevectors (implemented by subclasses)."""

        raise NotImplementedError

    def kernel(self, Xa, Xb=None):

        """

        Fidelity Gram matrix K(Xa, Xb).

        :param1 Xa: (na, d) array.
        :param2 Xb: (nb, d) array; None → Xb = Xa.

        :return: (na, nb) real kernel matrix.

        """

        psi_a = self.statevectors(Xa)
        psi_b = psi_a if Xb is None else self.statevectors(Xb)
        overlap = psi_a @ psi_b.conj().T
        return overlap.real ** 2 + overlap.imag ** 2


#%% 2. Quantum kernel
class QuantumKernel(FidelityKernel):

    """

    Statevector fidelity kernel with the Havlicek 'zz' map or the product 'z' map.

    Each layer applies a Hadamard wall, rz(2 a_i) per qubit and, for 'zz', a
    data-dependent rzz(2 (pi - a_c)(pi - a_t)) on every ring pair (a_i = scale * x_f).


    """

    VALID_FEATURE_MAPS = {"z", "zz"}

    def __init__(self, feature_map="zz", n_qubits=6, n_layers=1, scale=1.0):

        """

        :param1 feature_map: 'zz' (entangled, second order) or 'z' (product).
        :param2 n_qubits:    qubit cap (statevector size 2^q).
        :param3 n_layers:    encoding layers (raised so every feature is encoded).
        :param4 scale:       angle scaling on standardised inputs.

        :return: None.

        """

        fm = str(feature_map).lower()
        if fm not in self.VALID_FEATURE_MAPS:
            raise ValueError(f"feature_map must be one of: {sorted(self.VALID_FEATURE_MAPS)}")
        self.feature_map = fm
        self.n_qubits = max(1, int(n_qubits))
        self.n_layers = max(1, int(n_layers))
        self.scale = float(scale)

    #%% 2.1 Geometry
    def circuit_shape(self, d):

        """Return (n_qubits, n_layers) for a d-feature input."""

        q = max(1, min(d, self.n_qubits))
        n_layers = max(self.n_layers, ceil(d / q))
        return q, n_layers

    #%% 2.2 Statevectors
    def statevectors(self, X):

        """

        Evolve a batch of inputs through the feature map (exact, qiskit conventions).

        :param1 X: (n, d) array of standardised features.

        :return: (n, 2^q) complex statevector batch.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        n, d = X.shape
        q, n_layers = self.circuit_shape(d)
        dim = 1 << q

        psi = np.zeros((n, dim), dtype=np.complex128)
        psi[:, 0] = 1.0  # |0...0>

        idx = np.arange(dim)
        parity_sign = {}

        def zz_sign(c, t):

            """Cached (-1)^(b_c XOR b_t) eigenvalue of Z_c Z_t per basis state."""

            sign = parity_sign.get((c, t))
            if sign is None:
                sign = 1.0 - 2.0 * (((idx >> c) ^ (idx >> t)) & 1)
                parity_sign[(c, t)] = sign.astype(np.float64)
            return parity_sign[(c, t)]

        H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]],
                                            dtype=np.complex128)

        pairs = ring_pairs(q)
        entangle = self.feature_map == "zz"
        for layer in range(n_layers):
            # Hadamard wall in every layer, otherwise all layers collapse into one diagonal
            for t in range(q):
                psi = apply_1q(psi, H, t, n, q, dim)
            angles = [self.scale * X[:, (layer * q + i) % d] for i in range(q)]
            for i in range(q):
                # rz(2 a_i) = diag(e^{-i a_i}, e^{+i a_i})
                phase = np.exp(-1.0j * angles[i])
                RZ = np.zeros((n, 2, 2), dtype=np.complex128)
                RZ[:, 0, 0] = phase; RZ[:, 1, 1] = phase.conj()
                psi = apply_1q(psi, RZ, i, n, q, dim)
            if not entangle:
                continue
            for c, t in pairs:
                # rzz(2 (pi - a_c)(pi - a_t)) as a diagonal phase
                theta = (np.pi - angles[c]) * (np.pi - angles[t])
                psi *= np.exp(-1.0j * theta[:, None] * zz_sign(c, t)[None, :])
        return psi
