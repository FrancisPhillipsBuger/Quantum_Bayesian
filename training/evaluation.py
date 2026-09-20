#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluation helpers: descriptor extraction over a loader, full prediction
(composition + Tier 1 + Tier 2) and regression metrics.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from collections import namedtuple

import numpy as np
import torch
from scipy import stats

from common.report import log


#%% 1. Feature extraction over a loader
# One evaluation split; X_pooled is needed when Tier 2 uses the pooled descriptor
EvalSplit = namedtuple("EvalSplit", "X X_pooled y baseline n_atoms")


def extract_and_transform(model, dataloader, label):

    """

    Run a loader through the fitted pooling and reduction chain.

    :param1 model:      trained BayesianFramework.
    :param2 dataloader: PyG DataLoader.
    :param3 label:      split name for the log line.

    :return: EvalSplit(X, X_pooled, y, baseline, n_atoms).

    """

    mol_atoms, all_mol_atomic_numbers, mol_ads, y = model.collect_atom_features(
        dataloader, desc=f"{label} descriptors", label=f"{label} descriptors",
    )
    X_pooled = model.atom_pooler.transform(mol_atoms, ads_dist_list=mol_ads)
    baseline_pred = model.predict_composition_baseline_from_atomic_numbers(all_mol_atomic_numbers)
    n_atoms = np.array([len(z) for z in all_mol_atomic_numbers], dtype=np.float64)
    return EvalSplit(model.reduce(X_pooled), X_pooled, y, baseline_pred, n_atoms)


#%% 2. Full prediction
def evaluate_model(model, X, baseline_pred, prediction_cfg, X_pooled=None):

    """

    Compose composition baseline + Tier-1 ensemble + Tier-2 residual.

    :param1 model:          trained BayesianFramework.
    :param2 X:              reduced features.
    :param3 baseline_pred:  composition prediction, or None.
    :param4 prediction_cfg: dict {use_bma, use_residual}.
    :param5 X_pooled:       pooled descriptor (needed when Tier 2 uses it).

    :return: length-n prediction.

    """

    if prediction_cfg["use_bma"]:
        y_pred = model.predict_tier1(X)
    else:
        best_features = model.best_model['feature_indices']
        y_pred = model.best_model['model_result']['model'].predict(X[:, list(best_features)])

    if prediction_cfg["use_residual"]:
        residual_pred = model.predict_residual(X, X_pooled=X_pooled)
        if residual_pred is not None:
            y_pred = y_pred + residual_pred

    if baseline_pred is not None:
        y_pred = baseline_pred + y_pred
    return y_pred


#%% 3. Regression metrics
def calculate_metrics(y_true, y_pred, n_atoms=None):

    """

    R², RMSE, MAE, per-atom MAE (meV/atom), MAPE (NaN unless y keeps one sign),
    Pearson r and error mean / std.

    :param1 y_true:  ground truth.
    :param2 y_pred:  predictions.
    :param3 n_atoms: optional per-sample atom counts.

    :return: dict of scalar metrics.

    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    keys = ('r2', 'mape', 'rmse', 'mae', 'mae_per_atom', 'corr', 'error_mean', 'error_std')
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        log("Warning", "Predictions contain NaN/Inf; metrics set to NaN")
        return {k: np.nan for k in keys}

    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    # Relative error is undefined when the target changes sign or touches zero
    y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
    if (y_min > 0 or y_max < 0) and float(np.min(np.abs(y_true))) > 0:
        mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
    else:
        mape = np.nan

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    if n_atoms is None:
        mae_per_atom = np.nan
    else:
        counts = np.asarray(n_atoms, dtype=np.float64)
        mae_per_atom = float(
            np.mean(np.abs(y_true - y_pred) / np.maximum(counts, 1.0)) * 1000.0
        )

    errors = y_pred - y_true
    try:
        corr = stats.pearsonr(y_true, y_pred)[0]
    except Exception as e:
        log("Warning", f"Pearson correlation failed ({e}); set to NaN")
        corr = np.nan

    return {
        'r2': float(r2),
        'mape': float(mape),
        'rmse': float(rmse),
        'mae': float(mae),
        'mae_per_atom': mae_per_atom,
        'corr': corr,
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
    }
