<p align="center">
    <h2 align="center">QBM: A Quantum Bayesian Machine for Relaxed Energy Prediction on Metal-Oxide Catalysts</h2>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" alt="Linux">
<img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/Qiskit-2.4-6929C4?logo=qiskit&logoColor=white" alt="Qiskit 2.4">
<img src="https://img.shields.io/badge/PyTorch%20Geometric-2.7-3C2179?logo=pyg&logoColor=white" alt="PyTorch Geometric 2.7">
</p>

## What is QBM?
QBM is a Python-based Bayesian model-selection framework, with classical and quantum variants, that predicts the relaxed total energy of a metal-oxide catalyst surface from its initial structure. The complete workflow begins with a per-atom descriptor built from pseudospin-weighted radial (RDF), angular (ADF) and optional torsional (TDF) distribution functions. The per-atom descriptors are mapped through a Nyström approximation of an RBF kernel and sum-pooled into one vector per structure. A ridge regression on per-element atom counts then removes most of the total-energy scale, and partial least squares (PLS) projects the remaining signal onto a small number of supervised components. Kernel ridge regression models are fitted on subsets of these components and scored by a per-sample modified Bayesian Information Criterion (mBIC),

$$\mathrm{mBIC} = \ln\frac{\sigma^2}{\sigma_0^2} + \frac{k\ln n}{n} + \frac{2\gamma}{n}\,k\ln\frac{p}{k},$$

where $k$ is the subset size, $p$ the number of candidate components and $n$ the number of training samples. The top-$k$ subset models are averaged (Tier 1), and a residual learner corrects what they miss.
QBM is specifically designed to compare classical and quantum machine learning on the same materials-science regression task: the subset search and the kernels can run on a classical computer or on a simulated gate-model quantum computer.

## Why QBM?
Graph neural networks predict catalyst energies by training one large model end to end, which needs large amounts of compute and gives little insight into which structural features drive the prediction. Quantum machine learning is a promising alternative, but current quantum hardware and simulators can only handle a few qubits, so a quantum model cannot take a full atomic structure as input.

### Key Features
QBM is specifically designed to overcome these limitations by reducing the problem to a compact, physically motivated feature space in which classical and quantum learners can be compared directly.
- `Pseudospin descriptor` with atom-level Nyström pooling encodes multi-element surfaces in a fixed-length vector, and is cached on disk so later runs skip the most expensive step.
- `mBIC subset selection` finds the most informative descriptor subsets, either by exhaustive enumeration or as a QUBO solved by QAOA.
- `Two-tier prediction` combines Bayesian model averaging over the top-$k$ subset models with a residual learner (exact GP, Nyström + Bayesian ridge, linear ridge, or a quantum model).
- `Quantum kernels` replace the classical kernels with a fidelity quantum kernel (ZZ feature map) or a variational data re-uploading circuit. All circuits run on classical statevector simulators, so no quantum hardware is needed.

## Requirements
Required Python packages library include:
- `joblib==1.5.3`
- `lmdb==1.7.3`
- `numba==0.67.0`
- `numpy==2.5.1`
- `psutil==7.2.2`
- `PyYAML==6.0.3`
- `qiskit==2.4.2`
- `qiskit-aer==0.17.2`
- `qiskit-algorithms==0.4.0`
- `qiskit-optimization==0.7.0`
- `scikit-learn==1.9.0`
- `scipy==1.17.1`
- `torch==2.8.0`
- `torch-geometric==2.7.0`
- `tqdm==4.68.3`

Alternatively, install the environment using the provided .txt and .YAML files at [environment](/environment/).

## How to use QBM?

### Installation
* Clone this repository
    ```
    git clone <repository-url>
    ```
* Install dependencies
    ```
    pip install -r environment/requirements.txt
    ```
    or create the conda environment
    ```
    conda env create -f environment/environment.yaml
    conda activate qbm
    ```

PyTorch is only used for data loading and runs on CPU. If you need a specific CUDA build, install it first by following the [PyTorch instructions](https://pytorch.org/get-started/locally/).

### Dataset and Experiment
QBM reads the OC22 **IS2RE-Total** LMDB files, which store PyTorch Geometric `Data` objects with the initial positions, cell, tags and the relaxed DFT energy `y_relaxed`. Download `is2res_total_train_val_test_lmdbs.tar.gz` from the [OC22 dataset page](https://fair-chem.github.io/catalysts/datasets/oc22.html) and extract it into `dataset/`.

#### Experiment 1: Adsorbate-resolved subsets
Training on surfaces with and without adsorbates separately, to test how the framework handles adsorbate–surface interactions. Split each of `train`, `val_id` and `val_ood` by the `nads` field:
- Surfaces with adsorbates (`nads > 0`): `adsorbed` (31,137 / 1,688 / 1,857 structures)
- Clean slabs (`nads == 0`): `clean` (14,753 / 936 / 923 structures)

#### Experiment 2: Full IS2RE-Total
Training every classical and quantum variant on the full OC22 IS2RE-Total set, with in-distribution (`val_id`) and out-of-distribution (`val_ood`) validation.
- Full dataset: `is2re-total` (45,890 / 2,624 / 2,780 structures)

> [!CAUTION]
> To begin the algorithm process, ensure that the abovementioned dataset files are placed in the appropriate directories:
> - The full dataset is at `dataset/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/{train, val_id, val_ood}`.
> - The adsorbate-resolved subsets are at `dataset/is2res_by_adsorbate/adsorbed/{train, val_id, val_ood}` and `dataset/is2res_by_adsorbate/clean/{train, val_id, val_ood}`.
> - To keep the data somewhere else, set `DATA_ROOT` for the shell scripts or pass `--base_dir` to `main_train.py`.

> [!TIP]
> Per-atom descriptors are cached in a `.feature_cache/` folder inside each split directory. The cache key covers the data shards and every descriptor setting, so the cache is reused only when both match. Set `data.feature_cache: false` to turn caching off.

### Execution for program
`main_train.py` is the single entry point for training and evaluation. `--variant` selects which subset search and kernels are used, and each variant reads one self-contained YAML file from `config_file/`. Run it from the repository root:

```
python -m main_train train --variant <variant> --base_dir <dataset_split_folder> --result_dir <output_folder>
```

| Argument       | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| `--variant`    | `baseline`, `quantum`, `quantum_kernel` or `reupload`                           |
| `--config_dir` | Folder containing the variant YAML files (default: `config_file/`)              |
| `--base_dir`   | Dataset folder with `train/`, `val_id/` and `val_ood/` (overrides the config)   |
| `--result_dir` | Output folder (overrides the config)                                            |

#### 01 Baseline
`baseline` is the classical reference. It exhaustively enumerates every descriptor subset, fits an RBF kernel ridge model on each and scores it by mBIC. Tier 2 uses a classical GP or ridge residual learner. Configured by `model_selection.yaml`.

```
python -m main_train train --variant baseline --base_dir <dataset_split_folder> --result_dir <output_folder>
```

#### 02 Quantum
`quantum` formulates the subset search as a QUBO, solves it with QAOA on Qiskit Aer and rescores the candidate subsets by mBIC. The subset and residual models stay classical (RBF kernel ridge, and GP or ridge). Configured by `model_selection_quantum.yaml`.

```
python -m main_train train --variant quantum --base_dir <dataset_split_folder> --result_dir <output_folder>
```

#### 03 Quantum kernel
`quantum_kernel` uses the QAOA subset search and replaces the Tier-1 kernel with a fidelity quantum kernel built from a ZZ feature map. The Tier-2 GP can optionally use the same quantum kernel. Configured by `model_selection_quantum_kernel.yaml`.

```
python -m main_train train --variant quantum_kernel --base_dir <dataset_split_folder> --result_dir <output_folder>
```

#### 04 Data re-uploading
`reupload` uses the QAOA subset search with two-stage rescoring, a variational data re-uploading circuit as the Tier-1 model, and a data re-uploading regressor with Laplace uncertainty as the Tier-2 residual learner. Configured by `model_selection_quantum_reupload.yaml`.

```
python -m main_train train --variant reupload --base_dir <dataset_split_folder> --result_dir <output_folder>
```

> [!NOTE]
> In the quantum variants, `framework.training.pca_components` is also the number of QAOA qubits. A statevector simulation needs memory proportional to $2^p$, so keep this value small (the default is 16).

#### Outputs
Each run writes to its result folder:

```
<result_dir>/
├── models/<checkpoint>.pt        # Trained framework (compressed)
└── logs/
    ├── performance_metrics.txt   # R², RMSE, MAE, MAE/atom, MAPE, correlation for Training / Val_ID / Val_OOD
    └── training_log.txt          # Wall-clock training time
```

A summary table of R², RMSE (eV), MAE (eV) and MAE (meV/atom) for the training set and both validation splits is also printed to the console.

#### Configuration
The most important settings in the YAML files are:

| Section                          | Key                                                  | Meaning                                                      |
| -------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| `feature_extractor`              | `rdf_n_basis`, `adf_n_basis`, `tdf_n_basis`          | Basis size of each distribution function                     |
|                                  | `cutoff_radius`                                      | Neighbour cutoff in Å                                        |
| `framework`                      | `n_jobs`                                             | Number of parallel workers (set this to your CPU core count) |
|                                  | `atom_nystroem_components`                           | Rank of the atom-level Nyström map                           |
| `framework.training`             | `reduction`, `pca_components`                        | Reduction method (`pls`) and number of components or qubits  |
| `framework.prediction`           | `top_k`, `tier1_weighting`                           | Tier-1 ensemble size and weighting (`stacking` or `bic`)     |
| `selection`                      | `gamma`, `alpha_grid`                                | mBIC sparsity penalty and ridge regularisation grid          |
| `selection.qubo` / `.quantum`    | `cardinality_penalty`, `qaoa_reps`, `qaoa_maxiter`   | QUBO formulation and QAOA depth and optimiser budget         |
| `selection.quantum_kernel`       | `feature_map`, `n_qubits`, `n_layers`, `scale`       | Fidelity-kernel feature map                                  |
| `selection.reuploading`          | `n_qubits`, `n_layers`, `epochs`, `residual_backend` | Re-uploading circuit and training settings                   |
| `residual_learning`              | `kernel_type`, `sigma_noise`                         | Tier-2 kernel (`linear`, `rbf` or `matern`) and noise level  |


### Execution for full work flow
The following automated shell scripts run the complete QBM pipeline, from the raw LMDB files to the evaluation metrics, for each experiment.

#### Experiment 1: Adsorbate-resolved subsets
`run_train.sh` trains one variant on `adsorbed`, `clean` and `is2re-total`, in that order, and stops at the first failure.

* Classical baseline
    ```bash
    bash run_train.sh baseline
    ```

* Any other variant (`quantum`, `quantum_kernel` or `reupload`)
    ```bash
    bash run_train.sh quantum
    ```

#### Experiment 2: Full IS2RE-Total
`run_train_all_variants.sh` trains every variant in turn on `is2re-total`. A failed variant does not stop the others.

* All four variants
    ```bash
    bash run_train_all_variants.sh
    ```

* A subset of variants
    ```bash
    VARIANTS="baseline quantum" bash run_train_all_variants.sh
    ```

> [!IMPORTANT]
> - Validates the config and dataset directories before execution
> - Pins BLAS libraries to one thread; parallelism comes from joblib (`framework.n_jobs`)
> - Writes a timestamped log file under `RESULT_BASE`
> - `run_train.sh` exits immediately if any step fails (`set -e`); `run_train_all_variants.sh` reports the number of completed and failed variants at the end

## Acknowledgements
> QBM is developed by Chien-Chai Chang (Francis).\
> ©Copyright 2026. All rights reserved.
