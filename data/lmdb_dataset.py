#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Lazy LMDB dataset reader for OC22 IS2RE-Total shards.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import glob
import os
import pickle
from typing import List

import lmdb
import numpy as np
from torch_geometric.data import Data as PyGData


#%% 1. Legacy PyG sample migration
def migrate_old_pyg_data(obj):

    """

    Rebuild a legacy PyG Data sample (fields in __dict__, no _store) for modern PyG.

    :param1 obj: pickle-loaded sample.

    :return: PyG Data object compatible with modern PyG.

    """

    if isinstance(obj, PyGData):
        if "_store" in object.__getattribute__(obj, "__dict__"):
            return obj
        fields = dict(object.__getattribute__(obj, "__dict__"))
        return PyGData(**fields)
    return obj


#%% 2. Dataset
class LmdbDataset:

    """

    Concatenated reader for all data.*.lmdb shards under a directory.


    """

    def __init__(self, config: dict):

        """

        :param1 config: dict with key "src" pointing to a shard directory.

        :return: None.

        """

        src = config["src"]
        self.paths = sorted(glob.glob(os.path.join(src, "data.*.lmdb")))
        if not self.paths:
            raise FileNotFoundError(f"No LMDB files found at: {src}/data.*.lmdb")

        sizes: List[int] = []
        for path in self.paths:
            env = self.open_env(path)
            with env.begin() as txn:
                raw = txn.get(b"length")
                if raw is not None:
                    n = pickle.loads(raw)
                else:
                    n = sum(1 for k, _ in txn.cursor() if k != b"length")
            env.close()
            sizes.append(int(n))

        self.cumulative = np.cumsum([0] + sizes)

        # Per-process env cache avoids an mmap setup per sample
        self._envs: dict = {}
        self._env_pid = os.getpid()

    @staticmethod
    def open_env(path):

        """Open one LMDB shard read-only, without locking or readahead."""

        return lmdb.open(path, subdir=False, readonly=True, lock=False,
                         readahead=False, meminit=False)

    def env(self, shard_idx: int):

        """

        Return the cached env for a shard, reopened after a DataLoader fork.

        :param1 shard_idx: index into self.paths.

        :return: open lmdb.Environment.

        """

        pid = os.getpid()
        if pid != self._env_pid:
            self._envs = {}
            self._env_pid = pid
        env = self._envs.get(shard_idx)
        if env is None:
            env = self.open_env(self.paths[shard_idx])
            self._envs[shard_idx] = env
        return env

    def __getstate__(self):

        """Drop open environments so the dataset can be pickled to workers."""

        state = self.__dict__.copy()
        state["_envs"] = {}
        return state

    def __len__(self) -> int:

        """Return the total sample count across all shards."""

        return int(self.cumulative[-1])

    def __getitem__(self, idx: int):

        """

        Load one sample by global index across all shards.

        :param1 idx: global sample index.

        :return: PyG Data object.

        """

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx = int(np.searchsorted(self.cumulative[1:], idx, side="right"))
        local_idx = idx - int(self.cumulative[shard_idx])
        with self.env(shard_idx).begin() as txn:
            raw = txn.get(f"{local_idx}".encode("ascii"))
        if raw is None:
            raise KeyError(f"Shard {self.paths[shard_idx]} missing key {local_idx}")
        return migrate_old_pyg_data(pickle.loads(raw))
