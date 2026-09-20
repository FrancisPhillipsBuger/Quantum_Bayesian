#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

YAML configuration loader; each variant reads one self-contained config file.

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from pathlib import Path

import yaml


#%% 1. Defaults
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config_file"

# Sections every variant file must define
REQUIRED_SECTIONS = ("data", "feature_extractor", "framework",
                     "residual_learning", "selection")


#%% 2. Loaders
def load_yaml(path):

    """

    Load a single YAML file into a dict.

    :param1 path: path to the YAML file.

    :return: parsed dict (empty dict if the file is empty).

    """

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_variant_config(config_dir, stem):

    """

    Load and validate one variant's config file.

    :param1 config_dir: config directory; None → model/config_file.
    :param2 stem:       file stem, e.g. "model_selection_quantum".

    :return: dict with the file's top-level sections.

    """

    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"Config directory not found: {base}")

    path = base / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = load_yaml(path)
    missing = [k for k in REQUIRED_SECTIONS if k not in config]
    if missing:
        raise KeyError(f"{path} is missing required section(s): {missing}")
    return config
