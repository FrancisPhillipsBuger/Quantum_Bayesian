#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the bayesian_framework.predictors package.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Public re-exports
from bayesian_framework.predictors.composition_baseline import CompositionBaseline
from bayesian_framework.predictors.bayesian_averaging import BayesianModelAveraging
from bayesian_framework.predictors.residual_learning import (
    ClassicalGaussianProcess,
    KernelNystroem,
    LinearRidgeGCV,
    ResidualLearning,
)
from bayesian_framework.predictors.quantum_residual_learning import QuantumResidualLearning
from bayesian_framework.predictors.quantum_reuploading_residual import (
    ReuploadingResidualLearning,
)

__all__ = [
    "CompositionBaseline",
    "BayesianModelAveraging",
    "ClassicalGaussianProcess",
    "KernelNystroem",
    "LinearRidgeGCV",
    "ResidualLearning",
    "QuantumResidualLearning",
    "ReuploadingResidualLearning",
]
