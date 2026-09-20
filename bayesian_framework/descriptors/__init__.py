#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the bayesian_framework.descriptors package.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Public re-exports
from bayesian_framework.descriptors.feature_extractor import FeatureExtractor
from bayesian_framework.descriptors.atomic_pooling import AtomicNystroemPooler

__all__ = ["FeatureExtractor", "AtomicNystroemPooler"]
