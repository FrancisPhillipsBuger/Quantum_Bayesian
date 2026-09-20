#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Array-conversion helper shared across the framework.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import numpy as np


#%% 1. Helpers
def to_numpy(x):

    """

    Convert a torch tensor or array-like object to a numpy array.

    :param1 x: torch tensor, numpy array, or any array-like.

    :return: numpy.ndarray on CPU.

    """

    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)
