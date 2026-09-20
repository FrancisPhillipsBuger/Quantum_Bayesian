#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PyG DataLoader factory for training and validation splits.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.lmdb_dataset import LmdbDataset


#%% 1. Loaders
def build_loaders(data_cfg):

    """

    Build (train, val_id, val_ood) loaders from the data config section.

    :param1 data_cfg: data section of the variant config.

    :return: tuple of three PyG DataLoaders.

    """

    paths = data_cfg["paths"]
    loader_kwargs = dict(data_cfg["loader"])
    base = paths["base_dir"]

    def make(split, shuffle):

        """Return a loader over base/<split>."""

        return PyGDataLoader(LmdbDataset({"src": f"{base}/{paths[split]}"}),
                             shuffle=shuffle, **loader_kwargs)

    return (make("train_split", True), make("val_id_split", False),
            make("val_ood_split", False))
