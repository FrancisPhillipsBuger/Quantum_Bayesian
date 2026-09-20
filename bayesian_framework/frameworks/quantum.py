#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

BayesianFramework with QAOA subset selection (QuantumModelSelector) in place
of the exhaustive search.

Created on: Thu Jun 11 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from bayesian_framework.frameworks.base import BayesianFramework
from bayesian_framework.selectors.quantum import QuantumModelSelector


#%% 1. Framework (quantum variant)
class BayesianFrameworkQuantum(BayesianFramework):

    """

    Training and inference façade with QAOA subset selection.


    """

    def __init__(self, qubo_cardinality_penalty=0.5, qubo_cardinality_grid=None,
                 qubo_max_pool_size=5000, qubo_seed=0,
                 qaoa_reps=1, qaoa_maxiter=100, **kwargs):

        """

        :param1 qubo_cardinality_penalty: cardinality anchor weight.
        :param2 qubo_cardinality_grid:    target sizes; None → auto.
        :param3 qubo_max_pool_size:       cap on candidates refit exactly.
        :param4 qubo_seed:                QUBO / QAOA seed.
        :param5 qaoa_reps:                QAOA depth p.
        :param6 qaoa_maxiter:             COBYLA iterations.
        :param7 kwargs:                   BayesianFramework arguments.

        :return: None.

        """

        super().__init__(**kwargs)
        self.qubo_seed = qubo_seed
        self.model_selector = QuantumModelSelector(
            **self.model_selector.shared_kwargs(),
            qubo_cardinality_penalty=qubo_cardinality_penalty,
            qubo_cardinality_grid=qubo_cardinality_grid,
            qubo_max_pool_size=qubo_max_pool_size,
            qubo_seed=qubo_seed,
            qaoa_reps=qaoa_reps,
            qaoa_maxiter=qaoa_maxiter,
        )
