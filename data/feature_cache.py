#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

On-disk cache of per-atom descriptors, keyed on the data shards and every
FeatureExtractor setting; pooling and later stages are always recomputed.

Created on: Wed Aug 20 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import hashlib
import json
import os

import numpy as np

from common.report import log


#%% 1. Cache
class AtomFeatureCache:

    """

    Ragged per-molecule atom descriptors persisted as flat arrays + offsets.


    """

    VERSION = 1

    def __init__(self, cache_dir, key):

        """

        :param1 cache_dir: directory holding the cache files.
        :param2 key:       content hash of the (data, descriptor) pair.

        :return: None.

        """

        self.cache_dir = cache_dir
        self.key = key
        self.path = os.path.join(cache_dir, f"atom_features_{key}.npz")

    #%% 1.1 Key construction
    @classmethod
    def build_key(cls, dataset, extractor, max_samples=None):

        """

        Hash everything that can change the per-atom descriptor.

        :param1 dataset:     LmdbDataset (or any object with paths / cumulative).
        :param2 extractor:   FeatureExtractor instance.
        :param3 max_samples: sample cap applied while collecting, if any.

        :return: (cache_dir, key), or (None, None) when the dataset has no files.

        """

        paths = getattr(dataset, "paths", None)
        if not paths:
            return None, None
        cumulative = getattr(dataset, "cumulative", None)
        sizes = np.diff(cumulative).astype(np.int64).tolist() if cumulative is not None else []

        signature = {
            "version": cls.VERSION,
            "shards": [os.path.basename(p) for p in paths],
            "sizes": sizes,
            "n_used": int(len(dataset)),
            "max_samples": None if max_samples is None else int(max_samples),
            "rdf_n_basis": int(extractor.rdf_n_basis),
            "adf_n_basis": int(extractor.adf_n_basis),
            "tdf_n_basis": int(extractor.tdf_n_basis),
            "cutoff_radius": float(extractor.cutoff_radius),
            "use_pbc": bool(extractor.use_pbc),
            "pbc": np.asarray(extractor.pbc).astype(bool).tolist(),
            "species": (None if extractor.species is None
                        else [int(z) for z in extractor.species]),
        }
        blob = json.dumps(signature, sort_keys=True).encode("utf-8")
        key = hashlib.sha256(blob).hexdigest()[:16]
        return os.path.join(os.path.dirname(paths[0]), ".feature_cache"), key

    #%% 1.2 Load / save
    def load(self):

        """

        Read the cached descriptors back into per-molecule arrays.

        :return: (mol_atoms, mol_z, mol_ads, y) or None on any miss/error.

        """

        if not os.path.exists(self.path):
            return None
        try:
            with np.load(self.path) as data:
                offsets = data["offsets"]
                feats = data["feats"]
                z = data["z"]
                ads = data["ads"]
                has_ads = data["has_ads"]
                y = data["y"]
        except (OSError, ValueError, KeyError):
            return None

        mol_atoms, mol_z, mol_ads = [], [], []
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            mol_atoms.append(feats[a:b].astype(np.float64))
            mol_z.append(z[a:b].astype(np.int64))
            mol_ads.append(ads[a:b].astype(np.float64) if has_ads[i] else None)
        return mol_atoms, mol_z, mol_ads, y.astype(np.float64)

    def save(self, mol_atoms, mol_z, mol_ads, y):

        """

        Persist descriptors as float32 flat arrays + offsets (atomic tmp-file rename).

        :param1 mol_atoms: list of (n_atoms_i, F) arrays.
        :param2 mol_z:     list of (n_atoms_i,) atomic-number arrays.
        :param3 mol_ads:   list of (n_atoms_i,) adsorbate distances or None.
        :param4 y:         length-n target vector.

        :return: True when written, False otherwise.

        """

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            counts = np.array([len(m) for m in mol_atoms], dtype=np.int64)
            offsets = np.concatenate([[0], np.cumsum(counts)])
            has_ads = np.array([d is not None for d in mol_ads], dtype=bool)
            ads = np.concatenate([
                (np.asarray(d, dtype=np.float32).ravel() if d is not None
                 else np.zeros(counts[i], dtype=np.float32))
                for i, d in enumerate(mol_ads)
            ]) if len(mol_ads) else np.zeros(0, dtype=np.float32)

            tmp = self.path + ".tmp"
            np.savez(
                tmp,
                feats=np.vstack(mol_atoms).astype(np.float32),
                z=np.concatenate([np.asarray(v).ravel() for v in mol_z]).astype(np.int32),
                ads=ads,
                has_ads=has_ads,
                offsets=offsets,
                y=np.asarray(y, dtype=np.float64),
            )
            os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, self.path)
            return True
        except (OSError, ValueError, MemoryError) as exc:
            log("Warning", f"Could not write the descriptor cache ({exc})")
            return False
