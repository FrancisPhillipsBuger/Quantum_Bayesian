#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Variant registry: maps --variant to its config file and checkpoint name, and
builds the framework from the variant's self-contained config.

Created on: Mon Jun 15 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from bayesian_framework.frameworks.base import BayesianFramework
from bayesian_framework.frameworks.quantum import BayesianFrameworkQuantum
from bayesian_framework.frameworks.quantum_kernel import BayesianFrameworkQuantumKernel
from bayesian_framework.frameworks.quantum_reuploading import BayesianFrameworkQuantumReupload


#%% 1. Variant specifications
# variant -> config-file stem and checkpoint stem
VARIANTS = {
    "baseline": {
        "config_stem": "model_selection",
        "checkpoint": "bayesian_framework_is2re_best",
    },
    "quantum": {
        "config_stem": "model_selection_quantum",
        "checkpoint": "bayesian_framework_quantum_is2re_best",
    },
    "quantum_kernel": {
        "config_stem": "model_selection_quantum_kernel",
        "checkpoint": "bayesian_framework_quantum_kernel_is2re_best",
    },
    "quantum_reupload": {
        "config_stem": "model_selection_quantum_reupload",
        "checkpoint": "bayesian_framework_quantum_reupload_is2re_best",
    },
}


#%% 2. Kwargs builders
def base_kwargs(configs, sel_cfg, device):

    """

    Constructor kwargs shared by every framework variant.

    :param1 configs: dict from load_variant_config.
    :param2 sel_cfg: the selection section.
    :param3 device:  torch device.

    :return: kwargs dict.

    """

    fw = configs["framework"]
    fx = configs["feature_extractor"]
    res = configs["residual_learning"]
    return dict(
        device=device,
        n_jobs=fw["n_jobs"],
        atom_nystroem_components=fw["atom_nystroem_components"],
        pooling_ads_decay_length=fw.get("pooling_ads_decay_length"),
        use_composition_baseline=fw["use_composition_baseline"],
        composition_baseline_alphas=fw["composition_baseline_alphas"],
        rdf_n_basis=fx["rdf_n_basis"],
        adf_n_basis=fx["adf_n_basis"],
        tdf_n_basis=fx["tdf_n_basis"],
        cutoff_radius=fx["cutoff_radius"],
        species=fx["species"],
        alpha=sel_cfg["alpha"],
        gamma_krr=sel_cfg["gamma_krr"],
        gamma=sel_cfg["gamma"],
        selection_max_features=sel_cfg["max_features"],
        selection_use_nystroem=sel_cfg.get("use_nystroem", True),
        selection_nystroem_components=sel_cfg.get("nystroem_components", 200),
        selection_nystroem_random_state=sel_cfg.get("nystroem_random_state", 0),
        selection_linear=sel_cfg.get("linear", False),
        sigma_noise=res["sigma_noise"],
        residual_kernel_type=res["kernel_type"],
        residual_use_nystroem=res["use_nystroem"],
        residual_nystroem_components=res["nystroem_components"],
        residual_linear=res["linear"],
        residual_nu=res["nu"],
        residual_exact_gp_max_samples=res.get("exact_gp_max_samples", 2000),
        residual_oof_folds=fw["prediction"].get("residual_oof_folds", 5),
        residual_features=fw["prediction"].get("residual_features", "pooled"),
        residual_projection_components=fw["prediction"].get(
            "residual_projection_components"),
        tier1_weighting=fw["prediction"].get("tier1_weighting", "stacking"),
        reduction=fw["training"].get("reduction", "pls"),
        use_feature_cache=configs["data"].get("feature_cache", True),
        selection_alpha_grid=sel_cfg.get("alpha_grid"),
    )


def qubo_kwargs(qubo_cfg):

    """QUBO subset-search kwargs (quantum variants)."""

    return dict(
        qubo_cardinality_penalty=qubo_cfg.get("cardinality_penalty", 0.5),
        qubo_cardinality_grid=qubo_cfg.get("cardinality_grid"),
        qubo_max_pool_size=qubo_cfg.get("max_pool_size", 5000),
        qubo_seed=qubo_cfg.get("seed", 0),
    )


def quantum_kwargs(quantum_cfg):

    """QAOA solver kwargs (quantum variants)."""

    return dict(
        qaoa_reps=quantum_cfg.get("qaoa_reps", 1),
        qaoa_maxiter=quantum_cfg.get("qaoa_maxiter", 100),
    )


def qkernel_kwargs(qk_cfg):

    """Quantum-kernel kwargs (quantum_kernel variant)."""

    return dict(
        qk_feature_map=qk_cfg.get("feature_map", "zz"),
        qk_n_qubits=qk_cfg.get("n_qubits", 6),
        qk_n_layers=qk_cfg.get("n_layers", 1),
        qk_scale=qk_cfg.get("scale", 1.0),
        quantum_residual=qk_cfg.get("apply_to_residual", True),
    )


def quantum_reupload_kwargs(ru_cfg, proxy_cfg):

    """

    Re-uploading model kwargs (quantum_reupload variant).

    :param1 ru_cfg:    the reuploading config block.
    :param2 proxy_cfg: the selection_proxy block (stage-A proxy).

    :return: constructor kwargs dict.

    """

    return dict(
        ru_proxy_backend=proxy_cfg.get("backend", "quantum_kernel"),
        ru_proxy_feature_map=proxy_cfg.get("feature_map", "zz"),
        ru_proxy_n_qubits=proxy_cfg.get("n_qubits", 6),
        ru_proxy_n_layers=proxy_cfg.get("n_layers", 1),
        ru_proxy_scale=proxy_cfg.get("scale", 1.0),
        ru_residual_backend=ru_cfg.get("residual_backend", "explicit"),
        ru_laplace=ru_cfg.get("laplace", True),
        ru_laplace_prior_precision=ru_cfg.get("laplace_prior_precision", 1.0),
        ru_n_qubits=ru_cfg.get("n_qubits", 4),
        ru_n_layers=ru_cfg.get("n_layers", 3),
        ru_scale=ru_cfg.get("scale", 1.0),
        ru_learning_rate=ru_cfg.get("learning_rate", 0.05),
        ru_epochs=ru_cfg.get("epochs", 30),
        ru_batch_size=ru_cfg.get("batch_size", 256),
        ru_l2=ru_cfg.get("l2", 0.0),
        ru_selection_epochs=ru_cfg.get("selection_epochs", 20),
        ru_selection_max_samples=ru_cfg.get("selection_max_samples", 400),
        ru_patience=ru_cfg.get("patience", 5),
        ru_rescore_top_n=ru_cfg.get("rescore_top_n", 64),
        ru_readout_learning_rate=ru_cfg.get("readout_learning_rate"),
    )


#%% 3. Model construction
def build_model(variant, configs, device):

    """

    Construct the framework model for a variant from loaded configs.

    :param1 variant: one of VARIANTS.
    :param2 configs: dict from load_variant_config.
    :param3 device:  torch device.

    :return: configured framework instance.

    """

    sel_cfg = configs["selection"]
    kwargs = base_kwargs(configs, sel_cfg, device)

    if variant == "baseline":
        return BayesianFramework(**kwargs)

    # Quantum variants share the QUBO + QAOA settings
    kwargs.update(qubo_kwargs(sel_cfg.get("qubo", {})))
    kwargs.update(quantum_kwargs(sel_cfg.get("quantum", {})))

    if variant == "quantum":
        return BayesianFrameworkQuantum(**kwargs)

    if variant == "quantum_reupload":
        kwargs.update(quantum_reupload_kwargs(
            sel_cfg.get("reuploading", {}),
            sel_cfg.get("selection_proxy", {}),
        ))
        return BayesianFrameworkQuantumReupload(**kwargs)

    kwargs.update(qkernel_kwargs(sel_cfg.get("quantum_kernel", {})))
    return BayesianFrameworkQuantumKernel(**kwargs)
