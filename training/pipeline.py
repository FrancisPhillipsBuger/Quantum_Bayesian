#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train, evaluate and report one framework variant from its config file.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import os
import time
import traceback

import torch

from common.config_loader import load_variant_config
from common.report import log
from training.evaluation import calculate_metrics, evaluate_model, extract_and_transform
from training.loaders import build_loaders
from training.variants import VARIANTS, build_model


#%% 1. Evaluation
def evaluate_split(model, name, X, X_pooled, y, baseline, n_atoms, prediction_cfg):

    """

    Predict one split, compute its metrics and log a one-line result.

    :param1 model:          trained framework.
    :param2 name:           split name.
    :param3 X:              reduced features.
    :param4 X_pooled:       pooled descriptor.
    :param5 y:              ground truth.
    :param6 baseline:       composition prediction, or None.
    :param7 n_atoms:        per-sample atom counts.
    :param8 prediction_cfg: dict {use_bma, use_residual}.

    :return: metrics dict.

    """

    start = time.time()
    y_pred = evaluate_model(model, X, baseline, prediction_cfg, X_pooled=X_pooled)
    m = calculate_metrics(y, y_pred, n_atoms=n_atoms)
    log("Evaluation", f"{name}: R2 {m['r2']:.4f} | RMSE {m['rmse']:.4f} eV | "
                      f"MAE {m['mae']:.4f} eV ({time.time() - start:.1f} s)")
    return m


def evaluate_all(model, val_loaders, prediction_cfg):

    """

    Evaluate the training set (in-sample, cached features) and both validation splits.

    :param1 model:          trained framework.
    :param2 val_loaders:    (val_id_loader, val_ood_loader).
    :param3 prediction_cfg: dict {use_bma, use_residual}.

    :return: dict {split name: metrics}.

    """

    # Reuse cached training features so the order matches model.target
    baseline = model.composition_baseline_train_pred if model.use_composition_baseline else None
    metrics = {"Training": evaluate_split(
        model, "Training", model.reduce(model.features), model.features, model.target,
        baseline, model.train_n_atoms, prediction_cfg,
    )}
    for name, loader in zip(("Val (ID)", "Val (OOD)"), val_loaders):
        split = extract_and_transform(model, loader, name)
        metrics[name] = evaluate_split(
            model, name, split.X, split.X_pooled, split.y, split.baseline,
            split.n_atoms, prediction_cfg,
        )
    return metrics


#%% 2. Reporting
def print_summary(metrics):

    """

    Print the per-split metrics table and a generalisation note.

    :param1 metrics: dict {split name: metrics}.

    :return: None.

    """

    rule = "-" * 72
    print(rule)
    print(f"{'Dataset':<12}{'R2':>10}{'RMSE (eV)':>12}{'MAE (eV)':>12}{'MAE (meV/atom)':>18}")
    print(rule)
    for name, m in metrics.items():
        print(f"{name:<12}{m['r2']:>10.4f}{m['rmse']:>12.4f}{m['mae']:>12.4f}"
              f"{m['mae_per_atom']:>18.2f}")
    print(rule, flush=True)

    train, val = metrics["Training"]["rmse"], metrics["Val (ID)"]["rmse"]
    if train - val < -0.1:
        log("Summary", f"Train RMSE {train:.4f} eV < Val (ID) RMSE {val:.4f} eV: "
                       f"possible overfitting")
    elif abs(train - val) < 0.05:
        log("Summary", "Train and Val (ID) RMSE agree")
    else:
        log("Summary", f"Train / Val (ID) RMSE gap: {abs(train - val):.4f} eV")


def write_metrics_log(log_dir, metrics):

    """

    Write per-split metrics as CSV to log_dir/performance_metrics.txt.

    :param1 log_dir: output directory.
    :param2 metrics: dict {split name: metrics}.

    :return: None.

    """

    path = os.path.join(log_dir, "performance_metrics.txt")
    names = {"Training": "Training", "Val (ID)": "Val_ID", "Val (OOD)": "Val_OOD"}
    with open(path, "w") as f:
        f.write("Dataset,R2,RMSE,MAE,MAE_per_atom_meV,MAPE,Correlation,"
                "Error_Mean,Error_Std\n")
        for name, m in metrics.items():
            f.write(f"{names[name]},{m['r2']:.6f},{m['rmse']:.6f},{m['mae']:.6f},"
                    f"{m['mae_per_atom']:.6f},{m['mape']:.6f},{m['corr']:.6f},"
                    f"{m['error_mean']:.6f},{m['error_std']:.6f}\n")
    log("Summary", f"Metrics: {path}")


#%% 3. Top-level pipeline
def run(variant, args):

    """

    Train, evaluate and report one framework variant.

    :param1 variant: key of training.variants.VARIANTS.
    :param2 args:    parsed CLI arguments (config_dir, base_dir, result_dir).

    :return: exit code (0 success, 1 failure).

    """

    spec = VARIANTS[variant]
    try:
        configs = load_variant_config(args.config_dir, spec["config_stem"])
        data_cfg = configs["data"]
        if args.base_dir:
            data_cfg["paths"]["base_dir"] = args.base_dir
        if args.result_dir:
            data_cfg["result_dir"] = args.result_dir

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(data_cfg["result_dir"], "models")
        log_dir = os.path.join(data_cfg["result_dir"], "logs")
        for d in (model_dir, log_dir):
            os.makedirs(d, exist_ok=True)

        fw_cfg = configs["framework"]
        prediction_cfg = fw_cfg["prediction"]
        training_cfg = fw_cfg["training"]
        log("Setup", f"Variant: {variant} | Config: {spec['config_stem']}.yaml | "
                     f"Device: {device.type}")
        log("Setup", f"Reduction: {training_cfg.get('reduction', 'pls').upper()}, "
                     f"{training_cfg['pca_components']} components | "
                     f"Tier-1 top_k: {prediction_cfg['top_k']}")

        train_loader, val_id_loader, val_ood_loader = build_loaders(data_cfg)
        model = build_model(variant, configs, device)

        start = time.time()
        model.fit(
            train_loader,
            use_bma=prediction_cfg["use_bma"],
            use_residual=prediction_cfg["use_residual"],
            top_k=prediction_cfg["top_k"],
            use_pca=training_cfg["use_pca"],
            pca_components=training_cfg["pca_components"],
            memory_efficient=training_cfg["memory_efficient"],
            max_samples_in_memory=training_cfg["max_samples_in_memory"],
        )
        train_time = time.time() - start
        with open(os.path.join(log_dir, "training_log.txt"), "w") as f:
            f.write(f"Epoch,Train_Time\n1,{train_time}\n")
        log("Training", f"Completed in {train_time:.1f} s")

        model.save_model(
            os.path.join(model_dir, spec["checkpoint"] + ".pt"),
            save_all_models=fw_cfg["save"]["save_all_models"],
            compress=fw_cfg["save"]["compress"],
        )

        metrics = evaluate_all(model, (val_id_loader, val_ood_loader), prediction_cfg)
        print_summary(metrics)
        write_metrics_log(log_dir, metrics)
        log("Summary", "Completed")
        return 0
    except FileNotFoundError as e:
        log("Error", str(e))
        return 1
    except Exception as e:
        log("Error", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
