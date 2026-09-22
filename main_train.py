#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Entry point for BayesianFramework training and evaluation on OC22 IS2RE.
--variant selects baseline (mBIC + RBF KRR), quantum (QAOA search),
quantum_kernel (fidelity ZZ kernel) or quantum_reupload (data re-uploading VQC).

Created on: Sun May 24 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
import argparse
import os
import sys

# Pin BLAS threads before numpy / torch load; joblib supplies the parallelism
BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for v in BLAS_VARS:
    os.environ.setdefault(v, "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from training.pipeline import run
from training.variants import VARIANTS


#%% 1. Argument parsing
def build_parser():

    """

    Build the CLI parser.

    :return: argparse.ArgumentParser.

    """

    parser = argparse.ArgumentParser(description="Train BayesianFramework on OC22 IS2RE.")
    sub = parser.add_subparsers(dest='command')

    p_tr = sub.add_parser('train', help='Run training and evaluation.')
    p_tr.add_argument('--variant', default='baseline', choices=sorted(VARIANTS),
                      help='Selector / kernel variant.')
    p_tr.add_argument('--config_dir', default=None,
                      help='Config directory (default: model/config_file).')
    p_tr.add_argument('--base_dir', default=None,
                      help='Dataset directory; overrides data.paths.base_dir.')
    p_tr.add_argument('--result_dir', default=None,
                      help='Output directory; overrides data.result_dir.')
    p_tr.set_defaults(func=lambda args: run(args.variant, args))

    return parser


def main():

    """

    Parse the CLI and dispatch the sub-command.

    :return: exit code (0 on success).

    """

    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'func', None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
