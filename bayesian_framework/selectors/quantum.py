#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

QUBO-driven subset selection solved with gate-model QAOA on a local qiskit
simulator, followed by exact refit and mBIC rescoring.

Created on: Thu Jun 11 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import warnings
from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from bayesian_framework.selectors.baseline import ModelSelector
from common.report import log, progress
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


#%% 1. Sampler backends
def _make_statevector_sampler():

    """

    Build the pure-Python StatevectorSampler fallback.

    :return: (sampler, None, backend_label).

    """

    from qiskit.primitives import StatevectorSampler
    return StatevectorSampler(), None, 'StatevectorSampler'


def _make_qaoa_sampler(seed=None):

    """

    Build the qiskit-aer SamplerV2 (serial shots, avoids an aer race) with a
    basis-gate pass manager; falls back to StatevectorSampler without aer.

    :param1 seed: simulator seed (None → aer default).

    :return: (sampler, pass manager or None, backend_label).

    """

    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        pm = generate_preset_pass_manager(
            optimization_level=1,
            basis_gates=['rx', 'ry', 'rz', 'rzz', 'cx', 'h'],
        )
        sampler = AerSampler(
            seed=None if seed is None else int(seed),
            options=dict(backend_options=dict(
                max_parallel_shots=1, max_parallel_experiments=1,
            )),
        )
        return sampler, pm, 'qiskit-aer SamplerV2'
    except ImportError:
        return _make_statevector_sampler()


#%% 2. Quantum selector
class QuantumModelSelector(ModelSelector):

    """

    QUBO proposer + exact mBIC rescorer. The surrogate energy is
    E(z) = sum_i a_i z_i + sum_{i<j} b_ij z_i z_j from single / pair log-variances,
    plus a cardinality anchor lam * (sum_i z_i - k_t)^2 swept over targets k_t.


    """

    def __init__(self, qubo_cardinality_penalty=0.5, qubo_cardinality_grid=None,
                 qubo_max_pool_size=5000, qubo_seed=0,
                 qaoa_reps=1, qaoa_maxiter=100, **kwargs):

        """

        :param1 qubo_cardinality_penalty: anchor weight (relative to median |a_i|).
        :param2 qubo_cardinality_grid:    target sizes k_t; None → spread over [3, max_k].
        :param3 qubo_max_pool_size:       cap on candidates refit exactly.
        :param4 qubo_seed:                seed for the sampler and QAOA initial point.
        :param5 qaoa_reps:                QAOA depth p.
        :param6 qaoa_maxiter:             COBYLA iterations.
        :param7 kwargs:                   ModelSelector arguments.

        :return: None.

        """

        super().__init__(**kwargs)
        self.qubo_cardinality_penalty = float(qubo_cardinality_penalty)
        self.qubo_cardinality_grid = qubo_cardinality_grid
        self.qubo_max_pool_size = int(qubo_max_pool_size)
        self.qubo_seed = int(qubo_seed)
        self.qaoa_reps = int(qaoa_reps)
        self.qaoa_maxiter = int(qaoa_maxiter)
        self._sampler, self._qaoa_transpiler, self._qaoa_backend = \
            _make_qaoa_sampler(seed=self.qubo_seed)
        # Seeded QAOA initial point: an unseeded one makes identical configs diverge
        self._qaoa_initial_point = np.random.default_rng(self.qubo_seed).uniform(
            0.0, np.pi, size=2 * self.qaoa_reps,
        )

    def shared_kwargs(self):

        """Base selector kwargs plus the QUBO / QAOA knobs."""

        kwargs = super().shared_kwargs()
        kwargs.update(
            qubo_cardinality_penalty=self.qubo_cardinality_penalty,
            qubo_cardinality_grid=self.qubo_cardinality_grid,
            qubo_max_pool_size=self.qubo_max_pool_size,
            qubo_seed=self.qubo_seed,
            qaoa_reps=self.qaoa_reps,
            qaoa_maxiter=self.qaoa_maxiter,
        )
        return kwargs

    #%% 2.1 QUBO construction
    def build_qubo(self, a, B, c_t, k_t, lam):

        """

        Upper-triangular QUBO for one cardinality target (constant lam * k_t^2 dropped).

        :param1 a:   length-p surrogate linear terms.
        :param2 B:   (p, p) upper-triangular surrogate interactions.
        :param3 c_t: per-bit mBIC complexity cost at size k_t.
        :param4 k_t: cardinality target.
        :param5 lam: anchor weight (absolute).

        :return: (p, p) QUBO matrix Q.

        """

        p = len(a)
        Q = np.zeros((p, p), dtype=np.float64)
        Q[np.triu_indices(p, k=1)] = B[np.triu_indices(p, k=1)] + 2.0 * lam
        np.fill_diagonal(Q, a + c_t + lam * (1.0 - 2.0 * k_t))
        return Q

    #%% 2.2 QAOA solver
    def solve_qubo(self, Q, max_k):

        """

        Minimise one QUBO with MinimumEigenOptimizer(QAOA); every sampled bitstring
        becomes a candidate. Retries aer, then falls back to StatevectorSampler.

        :param1 Q:     upper-triangular QUBO matrix.
        :param2 max_k: maximum subset size kept.

        :return: set of sorted index tuples (1 <= len <= max_k).

        """

        p = Q.shape[0]
        qp = QuadraticProgram()
        for i in range(p):
            qp.binary_var(name=f"z{i}")
        linear = {f"z{i}": float(Q[i, i]) for i in range(p) if Q[i, i] != 0.0}
        quadratic = {
            (f"z{i}", f"z{j}"): float(Q[i, j])
            for i in range(p) for j in range(i + 1, p)
            if Q[i, j] != 0.0
        }
        qp.minimize(linear=linear, quadratic=quadratic)

        result = self._solve_with_sampler(
            qp, self._sampler, self._qaoa_transpiler, attempts=2,
        )
        if result is None:
            log("Warning", "Aer sampler failed repeatedly; retrying with StatevectorSampler")
            sv_sampler, sv_transpiler, _ = _make_statevector_sampler()
            result = self._solve_with_sampler(
                qp, sv_sampler, sv_transpiler, attempts=1,
            )
        if result is None:
            log("Warning", "QAOA solve failed on every backend; target skipped")
            return set()

        solutions = set()
        for sample in getattr(result, "samples", []) or []:
            indices = tuple(int(i) for i, bit in enumerate(sample.x) if bit > 0.5)
            if 0 < len(indices) <= max_k:
                solutions.add(indices)
        best = tuple(int(i) for i, bit in enumerate(result.x) if bit > 0.5)
        if 0 < len(best) <= max_k:
            solutions.add(best)
        return solutions

    def _solve_with_sampler(self, qp, sampler, transpiler, attempts):

        """

        Run MinimumEigenOptimizer(QAOA), retrying the transient aer "broadcast" error.

        :param1 qp:         QuadraticProgram to minimise.
        :param2 sampler:    qiskit sampler primitive.
        :param3 transpiler: pass manager (aer) or None.
        :param4 attempts:   maximum number of tries.

        :return: OptimizationResult, or None if every attempt failed.

        """

        for attempt in range(attempts):
            qaoa_kwargs = dict(
                sampler=sampler,
                optimizer=COBYLA(maxiter=self.qaoa_maxiter),
                reps=self.qaoa_reps,
                initial_point=self._qaoa_initial_point,
            )
            if transpiler is not None:
                qaoa_kwargs['transpiler'] = transpiler
            qaoa = QAOA(**qaoa_kwargs)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SparseEfficiencyWarning)
                    return MinimumEigenOptimizer(qaoa).solve(qp)
            except ValueError as exc:
                if "broadcast" not in str(exc):
                    raise
                if attempt + 1 < attempts:
                    log("Warning", f"Aer empty-shots error ({exc}); retry "
                                   f"{attempt + 2}/{attempts}")
        return None

    #%% 2.3 Selection driver
    def feature_selection(self, X, y):

        """

        Stage 1 fits all singles and pairs (surrogate data and BMA candidates);
        stage 2 solves one QUBO per cardinality target; stage 3 refits the proposals.

        :param1 X: (n, p) reduced feature matrix.
        :param2 y: length-n target.

        :return: dict {all_models, best_model, best_features, best_mbic, sigma_0_squared}.

        """

        n_features = X.shape[1]
        self.p = n_features

        X, y = self.sanitise(X, y)
        max_k = self.subset_cap(n_features)
        all_models = {}

        # Stage 1: singles + pairs
        n_pairs = comb(n_features, 2) if max_k >= 2 else 0
        n_exhaustive = sum(comb(n_features, k) for k in range(1, max_k + 1))
        log("Selection", f"Stage 1: fitting {n_features} singles + {n_pairs} pairs "
                         f"(exhaustive: {n_exhaustive})")
        singles = [(i,) for i in range(n_features)]
        self.register(all_models, self.fit_subsets(X, y, singles))
        if max_k >= 2:
            pairs = combinations(range(n_features), 2)
            self.register(all_models, self.fit_subsets(X, y, pairs, "Pairs", total=n_pairs))

        if not all_models:
            log("Warning", "No single/pair model survived; selection aborted")
            return {
                'all_models': {}, 'best_model': None, 'best_features': None,
                'best_mbic': None, 'sigma_0_squared': 1.0,
            }

        # Stage 2: one QUBO per cardinality target proposes subsets with k >= 3
        n = len(y)
        candidate_pool = set()
        if max_k >= 3:
            # Log-variance surrogate relative to the intercept-only model h0
            h0 = float(np.log(max(np.var(y), 1e-300)))
            B = np.zeros((n_features, n_features))
            h_single = np.full(n_features, np.nan)
            for i in range(n_features):
                info = all_models.get(str(i))
                if info is not None:
                    h_single[i] = np.log(max(info['sigma_ml_sq'], 1e-300))
            h_single = np.where(np.isnan(h_single), h0, h_single)   # unfittable → no gain
            a = h_single - h0
            for i, j in combinations(range(n_features), 2):
                info = all_models.get(f"{i}_{j}")
                if info is not None:
                    h_ij = np.log(max(info['sigma_ml_sq'], 1e-300))
                    B[i, j] = h_ij - h_single[i] - h_single[j] + h0

            grid = self.qubo_cardinality_grid
            if grid is None:
                n_targets = min(8, max_k - 2)
                grid = sorted(set(
                    int(k) for k in np.linspace(3, max_k, num=max(n_targets, 1))
                ))
            lam = self.qubo_cardinality_penalty * max(
                float(np.median(np.abs(a))), 1e-8,
            )

            log("QAOA", f"Stage 2: QUBO solves for cardinality targets {grid}")
            log("QAOA", f"Backend: {self._qaoa_backend} | {n_features} qubits | "
                        f"reps {self.qaoa_reps} | maxiter {self.qaoa_maxiter}")
            for k_t in progress(grid, "QAOA targets"):
                c_t = (np.log(n) / n) \
                    + (2.0 * self.gamma / n) * np.log(max(n_features / k_t, 1.0))
                Q = self.build_qubo(a, B, c_t, k_t, lam)
                candidate_pool.update(s for s in self.solve_qubo(Q, max_k) if len(s) >= 3)
            log("QAOA", f"Proposed {len(candidate_pool)} unique subsets (k >= 3)")

            # Keep the lowest-surrogate-energy candidates when the pool is too large
            if len(candidate_pool) > self.qubo_max_pool_size:
                Bsym = B + B.T

                def surrogate_energy(subset):

                    """Return the QUBO surrogate energy of one subset."""

                    idx = np.array(subset)
                    return float(a[idx].sum() + 0.5 * Bsym[np.ix_(idx, idx)].sum())

                candidate_pool = set(sorted(
                    candidate_pool, key=surrogate_energy,
                )[:self.qubo_max_pool_size])
                log("QAOA", f"Pool capped at {self.qubo_max_pool_size} by surrogate energy")

        # Stage 3: exact refit of the proposals
        if candidate_pool:
            log("Selection", f"Stage 3: exact refit of {len(candidate_pool)} QAOA candidates")
            self.register(all_models, self.fit_subsets(
                X, y, sorted(candidate_pool), "QAOA candidates",
            ))

        return self.finalise(all_models, n)
