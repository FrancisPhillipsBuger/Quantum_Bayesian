#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the bayesian_framework package.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Public re-exports
from bayesian_framework.frameworks import (
    BayesianFramework,
    BayesianFrameworkQuantum,
    BayesianFrameworkQuantumKernel,
    BayesianFrameworkQuantumReupload,
)
from bayesian_framework.descriptors import FeatureExtractor, AtomicNystroemPooler
from bayesian_framework.selectors import (
    ModelSelector,
    QuantumModelSelector,
    QuantumKernelModelSelector,
    ReuploadingModelSelector,
)
from bayesian_framework.predictors import (
    BayesianModelAveraging,
    ClassicalGaussianProcess,
    ResidualLearning,
    QuantumResidualLearning,
    ReuploadingResidualLearning,
    CompositionBaseline,
)

__all__ = [
    "BayesianFramework",
    "BayesianFrameworkQuantum",
    "BayesianFrameworkQuantumKernel",
    "BayesianFrameworkQuantumReupload",
    "FeatureExtractor",
    "AtomicNystroemPooler",
    "ModelSelector",
    "QuantumModelSelector",
    "QuantumKernelModelSelector",
    "ReuploadingModelSelector",
    "BayesianModelAveraging",
    "ClassicalGaussianProcess",
    "ResidualLearning",
    "QuantumResidualLearning",
    "ReuploadingResidualLearning",
    "CompositionBaseline",
]
