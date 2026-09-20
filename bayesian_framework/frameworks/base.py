#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

BayesianFramework: descriptor extraction, atom-level Nystrom pooling,
composition baseline, mBIC subset selection, Tier-1 model averaging and
Tier-2 residual learning.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import gc
import json
import os
import time

import numpy as np
import psutil
import torch
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common.arrays import to_numpy
from common.reducers import PLSReducer
from common.report import log, progress
from data.feature_cache import AtomFeatureCache
from bayesian_framework.descriptors.atomic_pooling import AtomicNystroemPooler
from bayesian_framework.predictors.bayesian_averaging import BayesianModelAveraging
from bayesian_framework.predictors.composition_baseline import CompositionBaseline
from bayesian_framework.descriptors.feature_extractor import FeatureExtractor
from bayesian_framework.selectors.baseline import ModelSelector
from bayesian_framework.predictors.residual_learning import ResidualLearning


#%% 1. Framework
class BayesianFramework:

    """

    Top-level training and inference façade (baseline: exhaustive mBIC selection).


    """

    def __init__(
        self,
        device='cuda',
        alpha=0.05,
        sigma_noise=0.1,
        gamma_krr=None,
        rdf_n_basis=11,
        adf_n_basis=11,
        tdf_n_basis=11,
        cutoff_radius=6.0,
        species=None,
        gamma=1.0,
        atom_nystroem_components=2048,
        pooling_ads_decay_length=None,
        n_jobs=None,
        residual_kernel_type="matern",
        residual_use_nystroem=False,
        residual_nystroem_components=512,
        residual_linear=False,
        residual_nu=2.5,
        residual_exact_gp_max_samples=2000,
        use_composition_baseline=False,
        composition_baseline_alphas=None,
        selection_max_features=None,
        residual_oof_folds=5,
        use_feature_cache=True,
        reduction="pls",
        residual_features="pooled",
        residual_projection_components=None,
        tier1_weighting="stacking",
        selection_alpha_grid=None,
        selection_use_nystroem=True,
        selection_nystroem_components=200,
        selection_nystroem_random_state=0,
        selection_linear=False,
    ):

        """

        :param1 device:                    torch device label (sklearn runs on CPU).
        :param2 alpha:                     selector KRR ridge strength.
        :param3 sigma_noise:               residual GP observation noise.
        :param4 gamma_krr:                 selector RBF bandwidth; None → auto.
        :param5 rdf_n_basis:               RDF basis count.
        :param6 adf_n_basis:               ADF basis count.
        :param7 tdf_n_basis:               TDF basis count (0 disables).
        :param8 cutoff_radius:             descriptor cut-off (Å).
        :param9 species:                   atomic numbers; None → detected from data.
        :param10 gamma:                    mBIC sparsity penalty.
        :param11 atom_nystroem_components: atom-pool landmark count.
        :param12 n_jobs:                   joblib workers.
        :param13 residual_*:               Tier-2 kernel / solver settings (residual_learning).
        :param14 use_composition_baseline: subtract a RidgeCV composition baseline.
        :param15 selection_*:              selector settings (max_features, Nystrom, linear, alpha grid).
        :param16 residual_oof_folds:       K for out-of-fold Tier-1 predictions (< 2 disables).
        :param17 reduction:                'pls' (supervised) or 'pca'.
        :param18 residual_features:        Tier-2 input: 'pooled' descriptor or 'reduced' scores.
        :param19 tier1_weighting:          'stacking' (OOF NNLS) or 'bic' (posterior).

        :return: None.

        """

        self.device = device
        self.alpha = alpha
        self.gamma_krr = gamma_krr
        self.gamma = gamma
        self.atom_nystroem_components = atom_nystroem_components
        self.n_jobs = n_jobs
        self.residual_oof_folds = int(residual_oof_folds or 0)
        self.use_feature_cache = bool(use_feature_cache)
        self.reduction = str(reduction).lower()
        self.residual_feature_source = str(residual_features).lower()
        self.residual_projection_components = residual_projection_components   # None → learner default
        self.tier1_weighting = str(tier1_weighting).lower()
        for name, value, allowed in (
            ("reduction", self.reduction, {"pls", "pca"}),
            ("residual_features", self.residual_feature_source, {"pooled", "reduced"}),
            ("tier1_weighting", self.tier1_weighting, {"stacking", "bic"}),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")

        self.feature_extractor = FeatureExtractor(
            rdf_n_basis=rdf_n_basis,
            adf_n_basis=adf_n_basis,
            tdf_n_basis=tdf_n_basis,
            cutoff_radius=cutoff_radius,
            species=species,
            n_jobs=n_jobs if n_jobs is not None else 1,
        )
        self.model_selector = ModelSelector(
            alpha=alpha, gamma_krr=gamma_krr, gamma=gamma,
            use_nystroem=selection_use_nystroem,
            nystroem_components=selection_nystroem_components,
            nystroem_random_state=selection_nystroem_random_state,
            linear=selection_linear,
            n_jobs=n_jobs, max_features=selection_max_features,
            alpha_grid=selection_alpha_grid,
        )
        self.bma = BayesianModelAveraging()
        self.residual_learner = ResidualLearning(
            sigma_noise=sigma_noise,
            linear=residual_linear,
            use_nystroem=residual_use_nystroem,
            nystroem_components=residual_nystroem_components,
            kernel_type=residual_kernel_type,
            nu=residual_nu,
            exact_gp_max_samples=residual_exact_gp_max_samples,
        )

        self.atom_pooler = AtomicNystroemPooler(
            n_components=atom_nystroem_components, random_state=0,
            ads_decay_length=pooling_ads_decay_length,
        )
        self.composition = CompositionBaseline(alphas=composition_baseline_alphas)
        self.use_composition_baseline = bool(use_composition_baseline)

        self.models = {}
        self.best_model = None
        self.model_weights = None
        self.features = None
        self.target = None
        self.residual_model = None
        self.pca = None                     # fitted reducer (PCA or PLSReducer)
        self.scaler = None
        self.composition_baseline_train_pred = None
        self.train_n_atoms = None
        self.top_models = None
        self.tier1_weight_report = None
        self.residual_feature_source_used_ = self.residual_feature_source
        self.residual_projection_ = None    # (scaler, reducer) when projecting

    #%% 1.1 Species cache
    @staticmethod
    def _species_cache_path(train_loader):

        """

        Species-cache path and shard signature of a loader's LMDB dataset.

        :param1 train_loader: PyG DataLoader.

        :return: (path, signature), or (None, None) without LMDB files.

        """

        dataset = getattr(train_loader, "dataset", None)
        paths = getattr(dataset, "paths", None)
        if not paths:
            return None, None
        signature = {
            "shards": [os.path.basename(p) for p in paths],
            "sizes": np.diff(dataset.cumulative).astype(np.int64).tolist()
            if hasattr(dataset, "cumulative") else [],
        }
        cache_path = os.path.join(os.path.dirname(paths[0]), ".species_cache.json")
        return cache_path, signature

    def _load_cached_species(self, train_loader):

        """Return the cached species array if the signature matches, else None."""

        cache_path, signature = self._species_cache_path(train_loader)
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path) as f:
                cached = json.load(f)
        except (OSError, ValueError):
            return None
        if cached.get("signature") != signature:
            return None
        return np.array(cached["species"], dtype=np.int64)

    def _save_cached_species(self, train_loader, species):

        """Write detected species next to the dataset shards (best effort)."""

        cache_path, signature = self._species_cache_path(train_loader)
        if not cache_path:
            return
        try:
            with open(cache_path, "w") as f:
                json.dump({"signature": signature, "species": species.tolist()}, f)
        except OSError:
            pass

    #%% 1.2 Descriptor collection (training and evaluation)
    def collect_atom_features(self, dataloader, max_samples=None,
                              desc="Descriptors", label="Descriptors"):

        """

        Stream a loader into per-molecule atom descriptors, using the on-disk cache.

        :param1 dataloader:  PyG DataLoader.
        :param2 max_samples: stop after this many molecules (None → all).
        :param3 desc:        progress-bar label.
        :param4 label:       name used in the log line.

        :return: (mol_atoms, mol_atomic_numbers, mol_ads_dist, y).

        """

        cache = None
        if self.use_feature_cache:
            cache_dir, key = AtomFeatureCache.build_key(
                getattr(dataloader, "dataset", None), self.feature_extractor,
                max_samples=max_samples,
            )
            if cache_dir is not None:
                cache = AtomFeatureCache(cache_dir, key)
                hit = cache.load()
                if hit is not None:
                    log("Data", f"{label}: {len(hit[0])} structures (cached)")
                    return hit

        mol_atoms, mol_z, mol_ads, targets = [], [], [], []
        for batch in progress(dataloader, desc):
            features, batch_z, batch_ads = self.feature_extractor.extract_features(batch)
            if isinstance(features, list):
                for i, feat in enumerate(features):
                    mol_atoms.append(np.asarray(feat, dtype=np.float64))
                    mol_z.append(np.asarray(batch_z[i], dtype=np.int64))
                    mol_ads.append(batch_ads[i])
            else:
                mol_atoms.append(np.asarray(features, dtype=np.float64))
                mol_z.append(np.asarray(batch_z[0], dtype=np.int64))
                mol_ads.append(batch_ads[0])
            targets.append(to_numpy(batch.y_relaxed))

            if max_samples is not None and len(mol_atoms) >= max_samples:
                break

        y = np.concatenate(targets) if targets else np.zeros(0)
        y = y[:len(mol_atoms)]      # a truncated last batch leaves extra targets

        log("Data", f"{label}: {len(mol_atoms)} structures (extracted)")
        if cache is not None:
            cache.save(mol_atoms, mol_z, mol_ads, y)
        return mol_atoms, mol_z, mol_ads, y

    #%% 1.3 Prediction helpers
    def predict_residual(self, X, X_pooled=None):

        """

        Tier-2 residual prediction on the features Tier 2 was trained on.

        :param1 X:        (n, p) reduced feature matrix.
        :param2 X_pooled: (n, d) pooled descriptor (needed for 'pooled' / projection inputs).

        :return: length-n residual prediction, or None when Tier 2 is off.

        """

        if self.residual_model is None:
            return None
        used = getattr(self, "residual_feature_source_used_",
                       self.residual_feature_source)
        if used.startswith("projection"):
            if X_pooled is None or self.residual_projection_ is None:
                raise ValueError("Tier 2 uses a projection of the pooled descriptor; "
                                 "X_pooled is required.")
            scaler, reducer = self.residual_projection_
            feats = reducer.transform(scaler.transform(X_pooled))
        elif used == "pooled":
            if X_pooled is None:
                raise ValueError("Tier 2 uses the pooled descriptor; X_pooled is required.")
            feats = X_pooled
        else:
            feats = X
        return self.residual_model['model'].predict(feats, return_std=False)

    def predict_composition_baseline_from_atomic_numbers(self, mol_atomic_numbers):

        """

        Composition-baseline energies for a list of molecules.

        :param1 mol_atomic_numbers: list of (n_atoms_i,) atomic-number arrays.

        :return: length-n baseline vector (zeros when the baseline is off).

        """

        if not self.use_composition_baseline:
            return np.zeros(len(mol_atomic_numbers), dtype=np.float64)
        return self.composition.predict(mol_atomic_numbers)

    def reduce(self, X_pooled):

        """

        Apply the fitted scaler + reducer (identity when reduction is off).

        :param1 X_pooled: (n, d) pooled descriptor.

        :return: (n, p) reduced features.

        """

        if self.pca is None or self.scaler is None:
            return X_pooled
        return self.pca.transform(self.scaler.transform(X_pooled))

    #%% 1.4 Dimensionality reduction
    def apply_reduction(self, X, y, n_components):

        """

        Standardise X, then reduce it with PLS (supervised on y) or PCA.

        :param1 X:            (n, d) pooled features.
        :param2 y:            length-n target tracked by PLS.
        :param3 n_components: component count.

        :return: (n, n_components) scores.

        """

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_components = int(n_components)

        if self.reduction == "pca":
            max_components = min(X.shape[0], X.shape[1])
            n_final = min(n_components, max_components)
            svd_solver = 'randomized' if n_final < max_components * 0.8 else 'auto'
            reducer = PCA(n_components=n_final, svd_solver=svd_solver)
            scores = reducer.fit_transform(X_scaled)
        else:
            reducer, scores = PLSReducer.fit(X_scaled, y, n_components)

        self.pca = reducer
        self.scaler = scaler
        log("Reduction", f"{self.reduction.upper()}: {X.shape[1]} -> "
                         f"{reducer.n_components_} components "
                         f"({np.sum(reducer.explained_variance_ratio_) * 100:.2f}% variance)")
        return scores

    #%% 1.5 Tier 1: out-of-fold predictions and weighting
    def out_of_fold_tier1(self, X, y, top_models, n_splits):

        """

        Refit the Tier-1 members per fold and predict held-out samples (removes
        Tier-1 fitting bias; subset selection itself stays in-sample).

        :param1 X:          (n, p) feature matrix.
        :param2 y:          length-n target.
        :param3 top_models: model_info dicts to refit.
        :param4 n_splits:   number of folds.

        :return: (n, k) per-model out-of-fold predictions, or None on failure.

        """

        subsets = [tuple(m['feature_indices']) for m in top_models]
        if not subsets:
            return None

        log("Tier-1", f"Out-of-fold predictions: {n_splits} folds x {len(subsets)} models")

        oof = np.full((len(y), len(subsets)), np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for fold, (tr, te) in enumerate(kf.split(X)):
            fits = self.model_selector.fit_subsets(X[tr], y[tr], subsets)
            for col, (fi, info) in enumerate(zip(subsets, fits)):
                if info is None:
                    log("Warning", f"Subset {fi} failed on fold {fold + 1}; "
                                   f"out-of-fold predictions disabled")
                    return None
                oof[te, col] = info['model_result']['model'].predict(
                    X[np.ix_(te, list(fi))]
                )

        if np.isnan(oof).any():
            return None
        return oof

    def predict_tier1(self, X):

        """

        Combine the stored top models with the weights chosen at fit time.

        :param1 X: (n, p) reduced feature matrix.

        :return: length-n Tier-1 prediction.

        """

        if not self.top_models:
            raise ValueError("Tier-1 models not available; call fit() first.")
        return self.bma.combine(X, self.top_models, self.model_weights)

    @staticmethod
    def stacking_weights(oof_matrix, y):

        """

        Non-negative, sum-to-one stacking weights by NNLS with a heavily weighted
        row enforcing sum(w) = 1.

        :param1 oof_matrix: (n, k) out-of-fold predictions.
        :param2 y:          length-n target.

        :return: length-k weight vector.

        """

        P = np.asarray(oof_matrix, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        rho = float(np.sqrt(np.mean(P ** 2)) * P.shape[0]) or 1.0
        A = np.vstack([P, np.full((1, P.shape[1]), rho)])
        rhs = np.concatenate([y, [rho]])
        try:
            w, _ = nnls(A, rhs)
        except Exception as exc:
            log("Warning", f"NNLS stacking failed ({exc}); using equal weights")
            return np.full(P.shape[1], 1.0 / P.shape[1])
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(P.shape[1], 1.0 / P.shape[1])
        return w / total

    def choose_tier1_weights(self, top_models, oof_matrix, y):

        """

        Pick the Tier-1 weights and report BIC / stacking / equal weighting out of fold.

        :param1 top_models: model_info dicts.
        :param2 oof_matrix: (n, k) out-of-fold predictions, or None.
        :param3 y:          length-n target.

        :return: length-k weight vector.

        """

        bic_w = self.bma.bic_weights(top_models)
        if oof_matrix is None:
            if self.tier1_weighting == "stacking":
                log("Warning", "Stacking requires out-of-fold predictions; using BIC weights")
            return bic_w

        stack_w = self.stacking_weights(oof_matrix, y)
        y = np.asarray(y).ravel()
        report = {}
        for name, w in (("bic", bic_w), ("stacking", stack_w),
                        ("equal", np.full(len(top_models), 1.0 / len(top_models)))):
            err = oof_matrix @ w - y
            report[name] = (float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))))
        self.tier1_weight_report = report

        summary = ", ".join(
            f"{name} {mae:.4f} / {rmse:.4f}" + (" (selected)" if name == self.tier1_weighting else "")
            for name, (mae, rmse) in report.items()
        )
        log("Tier-1", f"Weighting, OOF MAE / RMSE (eV): {summary}")

        return stack_w if self.tier1_weighting == "stacking" else bic_w

    #%% 1.6 Tier 2 input
    def build_residual_features(self, X_pooled, X_reduced, target_resid):

        """

        Tier-2 input: a residual-targeted PLS projection of the pooled descriptor when
        the learner has a preferred width, else the pooled descriptor or the reduced
        scores (per residual_features).

        :param1 X_pooled:     (n, D) pooled descriptor.
        :param2 X_reduced:    (n, p) Tier-1 reduced scores.
        :param3 target_resid: length-n Tier-1 residual.

        :return: (feature matrix, source label).

        """

        k = self.residual_projection_components
        if k is None:
            k = getattr(self.residual_learner, "PREFERRED_RESIDUAL_DIM", None)

        if k:
            k = int(min(k, X_pooled.shape[1]))
            scaler = StandardScaler()
            reducer, scores = PLSReducer.fit(
                scaler.fit_transform(X_pooled), target_resid, k,
            )
            self.residual_projection_ = (scaler, reducer)
            return scores, f"projection({reducer.n_components_})"

        self.residual_projection_ = None
        if self.residual_feature_source == "pooled":
            return X_pooled, "pooled"
        return X_reduced, "reduced"

    #%% 1.7 Training
    def fit(self, train_loader, use_bma=True, use_residual=True,
            top_k=16, use_pca=True, pca_components=16,
            memory_efficient=False, max_samples_in_memory=None):

        """

        Train the framework end to end.

        :param1 train_loader:          PyG DataLoader.
        :param2 use_bma:               Tier-1 ensemble of the top_k subsets (else best only).
        :param3 use_residual:          train the Tier-2 residual learner.
        :param4 top_k:                 Tier-1 ensemble size.
        :param5 use_pca:               apply the dimensionality reduction.
        :param6 pca_components:        reduced dimension.
        :param7 memory_efficient:      free per-atom buffers after pooling.
        :param8 max_samples_in_memory: training-sample cap (None → all).

        :return: self.

        """

        # Species from the whole training set (cached on disk after the first scan)
        if not self.feature_extractor.initialized:
            species = self._load_cached_species(train_loader)
            source = "cached"
            if species is None:
                species_set = set()
                for batch in progress(train_loader, "Species scan"):
                    atomic_numbers = np.round(to_numpy(batch.atomic_numbers)).astype(np.int64)
                    species_set.update(np.unique(atomic_numbers).tolist())
                species = np.array(sorted(species_set), dtype=np.int64)
                self._save_cached_species(train_loader, species)
                source = "scanned"
            self.feature_extractor.initialize_from_batch(species)
            log("Data", f"Species: {len(self.feature_extractor.species)} elements ({source})")

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        all_mol_atoms, all_mol_atomic_numbers, all_mol_ads_dist, y = \
            self.collect_atom_features(train_loader, max_samples=max_samples_in_memory,
                                       label="Train descriptors")

        if memory_efficient:
            mem_current = process.memory_info().rss / 1024 / 1024
            log("Data", f"Memory: {mem_before:.1f} MB -> {mem_current:.1f} MB")

        if len(all_mol_atomic_numbers) != len(y):
            raise ValueError(
                f"Composition baseline length mismatch: {len(all_mol_atomic_numbers)} vs {len(y)}"
            )

        baseline_pred = np.zeros_like(y, dtype=np.float64)
        y_model = y
        if self.use_composition_baseline:
            baseline_pred = self.composition.fit(all_mol_atomic_numbers, y)
            y_model = y - baseline_pred
            log("Baseline", f"Residual target std {float(np.std(y_model)):.4f} eV "
                            f"(raw {float(np.std(y)):.4f} eV)")

        X = self.atom_pooler.fit(all_mol_atoms, ads_dist_list=all_mol_ads_dist)

        if memory_efficient:
            del all_mol_atoms
            gc.collect()

        self.features = X
        self.target = y
        self.train_n_atoms = np.array(
            [len(z) for z in all_mol_atomic_numbers], dtype=np.float64,
        )
        self.composition_baseline_train_pred = baseline_pred

        X_pooled = X
        if use_pca:
            X = self.apply_reduction(X, y_model, pca_components)

        selection_result = self.model_selector.feature_selection(X, y_model)
        self.models = selection_result['all_models']
        self.best_model = selection_result['best_model']

        best_mbic = selection_result['best_mbic']
        best_txt = "n/a" if best_mbic is None else f"{best_mbic:.4e}"
        log("Selection", f"Best subset: mBIC {best_txt}, "
                         f"features {selection_result['best_features']}")

        # Tier-1 members, best mBIC first
        if use_bma:
            self.top_models = sorted(
                self.models.values(), key=lambda m: m['mbic'],
            )[:top_k]
            log("Tier-1", f"Ensemble: top {len(self.top_models)} of {len(self.models)} subsets")
        else:
            self.top_models = [self.best_model]
            log("Tier-1", "Best subset only")

        # One OOF matrix serves both the stacking weights and the residual target
        need_oof = (
            self.tier1_weighting == "stacking"
            or (use_residual and self.residual_oof_folds >= 2)
        )
        oof_matrix = None
        if need_oof and self.residual_oof_folds >= 2:
            oof_matrix = self.out_of_fold_tier1(
                X, y_model, self.top_models, self.residual_oof_folds,
            )
        elif need_oof:
            log("Warning", "residual_oof_folds < 2: no out-of-fold predictions")

        self.model_weights = self.choose_tier1_weights(
            self.top_models, oof_matrix, y_model,
        )
        y_pred = self.predict_tier1(X)
        y_pred_oof = None if oof_matrix is None else oof_matrix @ self.model_weights

        if use_residual:
            tier1_for_resid = y_pred if y_pred_oof is None else y_pred_oof
            residual_X, source = self.build_residual_features(
                X_pooled, X, y_model - tier1_for_resid,
            )
            self.residual_feature_source_used_ = source
            log("Tier-2", f"{type(self.residual_learner).__name__} on {source} features "
                          f"({residual_X.shape[1]} dims)")
            if y_pred_oof is None:
                log("Warning", "Tier 2 is trained on in-sample Tier-1 residuals (biased low)")

            y_pred, self.residual_model = self.residual_learner.fit(
                residual_X, y_model, y_pred, self.best_model,
                feature_indices=list(range(residual_X.shape[1])),
                y_pred_oof=y_pred_oof,
            )

        y_pred_total = baseline_pred + y_pred
        mae = float(np.mean(np.abs(y - y_pred_total)))
        r2 = float(1 - np.sum((y - y_pred_total) ** 2) / np.sum((y - np.mean(y)) ** 2))
        log("Training", f"Training fit: MAE {mae:.4f} eV | R2 {r2:.4f}")

        return self

    #%% 1.8 Persistence
    def save_model(self, filepath, save_all_models=False, compress=True):

        """

        Save a torch checkpoint plus a JSON summary.

        :param1 filepath:        target path (.pt appended if missing).
        :param2 save_all_models: persist every candidate instead of the ensemble.
        :param3 compress:        zip serialisation.

        :return: None.

        """

        start_time = time.time()

        # Ordered keys: model_weights is positional
        top_model_keys = [ModelSelector.key(m['feature_indices']) for m in (self.top_models or [])]

        if save_all_models:
            models_to_save = self.models
            scope = f"all {len(self.models)} models"
        elif self.top_models:
            models_to_save = {ModelSelector.key(m['feature_indices']): m for m in self.top_models}
            scope = f"top {len(self.top_models)} of {len(self.models)} models"
        else:
            models_to_save = {'best': self.best_model}
            top_model_keys = ['best']
            scope = "best model only"

        model_data = {
            'device': str(self.device),
            'alpha': self.alpha,
            'gamma_krr': self.gamma_krr,
            'gamma': self.gamma,
            'feature_extractor_config': {
                'rdf_n_basis': self.feature_extractor.rdf_n_basis,
                'adf_n_basis': self.feature_extractor.adf_n_basis,
                'tdf_n_basis': self.feature_extractor.tdf_n_basis,
                'cutoff_radius': self.feature_extractor.cutoff_radius,
                'species': self.feature_extractor.species,
            },
            'best_model': self.best_model,
            'models': models_to_save,
            'save_all_models': save_all_models,
            'top_model_keys': top_model_keys,
            'model_weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'tier1_weighting': self.tier1_weighting,
            'tier1_weight_report': self.tier1_weight_report,
            'reduction': self.reduction,
            'residual_features': self.residual_feature_source,
            'residual_features_used': self.residual_feature_source_used_,
            'residual_projection': self.residual_projection_,
            'residual_projection_components': self.residual_projection_components,
            'residual_oof_folds': self.residual_oof_folds,
            'atom_scaler': self.atom_pooler.scaler,
            'atom_nystroem': self.atom_pooler.nystroem,
            'atom_nystroem_components': self.atom_nystroem_components,
            'pooling_ads_decay_length': self.atom_pooler.ads_decay_length,
            'pca': self.pca,
            'scaler': self.scaler,
            'residual_model': self.residual_model,
            'use_composition_baseline': self.use_composition_baseline,
            'composition_model': self.composition.model,
            'composition_species': self.composition.species,
            'composition_baseline_alphas': (
                np.asarray(self.composition.alphas).tolist()
                if self.composition.alphas is not None else None
            ),
            'selection_max_features': self.model_selector.max_features,
        }

        if not filepath.endswith('.pt'):
            filepath = filepath + '.pt'

        try:
            if compress:
                torch.save(model_data, filepath, _use_new_zipfile_serialization=True)
            else:
                torch.save(model_data, filepath)
        except Exception as e:
            log("Warning", f"Compressed save failed ({e}); retrying uncompressed")
            torch.save(model_data, filepath)

        save_time = time.time() - start_time
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

        summary_path = filepath.replace('.pt', '_summary.json')
        summary = {
            'alpha': float(self.alpha) if self.alpha is not None else None,
            'gamma_krr': float(self.gamma_krr) if self.gamma_krr is not None else None,
            'best_model_mbic': float(self.best_model['mbic']) if self.best_model else None,
            'best_model_features': (
                [int(x) for x in self.best_model['feature_indices']] if self.best_model else None
            ),
            'n_models_saved': int(len(models_to_save)),
            'n_models_total': int(len(self.models)),
            'save_all_models': save_all_models,
            'n_top_k_models': int(len(self.model_weights)) if self.model_weights is not None else 0,
            'reduction': self.reduction,
            'tier1_weighting': self.tier1_weighting,
            'tier1_weight_report': self.tier1_weight_report,
            'residual_features': self.residual_feature_source,
            'use_pca': bool(self.pca is not None),
            'n_pca_components': int(self.pca.n_components_) if self.pca is not None else None,
            'n_residual_models': 1 if self.residual_model is not None else 0,
            'use_composition_baseline': bool(self.use_composition_baseline),
            'composition_baseline_alpha': (
                float(self.composition.model.alpha_)
                if self.composition.model is not None and hasattr(self.composition.model, 'alpha_')
                else None
            ),
            'composition_baseline_model': (
                type(self.composition.model).__name__ if self.composition.model is not None else None
            ),
            'selection_max_features': (
                int(self.model_selector.max_features)
                if self.model_selector.max_features is not None else None
            ),
            'file_size_mb': float(file_size_mb),
            'save_time_seconds': float(save_time),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log("Checkpoint", f"Saved {filepath} ({scope}, {file_size_mb:.2f} MB)")

    @classmethod
    def load_model(cls, filepath, device='cuda'):

        """

        Load a checkpoint written by save_model().

        :param1 filepath: path to the .pt checkpoint.
        :param2 device:   device label.

        :return: BayesianFramework instance.

        """

        # weights_only=False: the checkpoint stores sklearn objects
        model_data = torch.load(filepath, map_location='cpu', weights_only=False)

        config = model_data['feature_extractor_config']
        instance = cls(
            device=device,
            alpha=model_data['alpha'],
            gamma_krr=model_data['gamma_krr'],
            gamma=model_data['gamma'],
            rdf_n_basis=config['rdf_n_basis'],
            adf_n_basis=config['adf_n_basis'],
            tdf_n_basis=config.get('tdf_n_basis', 11),
            cutoff_radius=config['cutoff_radius'],
            species=config['species'],
            atom_nystroem_components=model_data.get('atom_nystroem_components', 2048),
            pooling_ads_decay_length=model_data.get('pooling_ads_decay_length'),
            use_composition_baseline=model_data.get('use_composition_baseline', False),
            composition_baseline_alphas=model_data.get('composition_baseline_alphas'),
            selection_max_features=model_data.get('selection_max_features'),
        )

        instance.best_model = model_data['best_model']
        instance.models = model_data['models']
        instance.model_weights = (
            np.array(model_data['model_weights']) if model_data['model_weights'] is not None else None
        )
        instance.reduction = model_data.get('reduction', 'pca')
        instance.residual_feature_source = model_data.get('residual_features', 'reduced')
        instance.residual_feature_source_used_ = model_data.get(
            'residual_features_used', instance.residual_feature_source,
        )
        instance.residual_projection_ = model_data.get('residual_projection')
        instance.residual_projection_components = model_data.get(
            'residual_projection_components',
        )
        instance.tier1_weighting = model_data.get('tier1_weighting', 'bic')
        instance.tier1_weight_report = model_data.get('tier1_weight_report')

        # Rebuild the ordered ensemble so the weights line up positionally
        keys = model_data.get('top_model_keys')
        if keys:
            instance.top_models = [instance.models[k] for k in keys if k in instance.models]
            if len(instance.top_models) != len(keys):
                log("Warning", "Checkpoint lacks some ensemble members; Tier-1 disabled")
                instance.model_weights = None
                instance.top_models = None
        elif instance.models:
            n_top = len(instance.model_weights) if instance.model_weights is not None else 1
            instance.top_models = sorted(
                instance.models.values(), key=lambda m: m['mbic'],
            )[:n_top]

        instance.use_composition_baseline = bool(model_data.get('use_composition_baseline', False))
        instance.composition.model = model_data.get('composition_model')
        instance.composition.species = model_data.get('composition_species')

        instance.atom_pooler.scaler = model_data.get('atom_scaler')
        instance.atom_pooler.nystroem = model_data.get('atom_nystroem')

        instance.pca = model_data['pca']
        instance.scaler = model_data['scaler']
        instance.residual_model = model_data['residual_model']

        log("Checkpoint", f"Loaded {filepath}: {len(instance.models)} models, "
                          f"reduced dim {instance.pca.n_components_ if instance.pca else 'n/a'}, "
                          f"Tier 2 {'on' if instance.residual_model else 'off'}")
        return instance
