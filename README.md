# PertSpectra: Guided triplet factor analysis of perturb-seq data with a prior
Factor analysis model for perturb-seq data, guided by perturbation labels and prior graph regularization

Code accompanying [TODO: paper link here]()

## Abstract
Computational modeling of interventional data is a rapidly expanding area of machine learning. In drug discovery, measuring the effects of genetic
interventions on cells is important for characterizing unknown disease mechanisms, but interpreting the resulting measurements remains a challenging task. Reliable latent space interpretability and modeling interactions between interventions are key areas of improvement for current models
in literature. Therefore, we propose PertSpectra, an extension of previously described factor analysis method Spectra (Kunes et al., 2024) to
explicitly support intervensional data. PertSpectra leverages Spectra’s differentiable graph regularization to incorporate prior biological knowledge
to generate sparse, biologically relevant latent factors that capture perturbational effects. We assess PertSpectra on three single cell sequencing
datasets with genetic perturbations, measuring latent space interpretability, predictive ability on unseen combinations of perturbations, and identification of perturbations of similar biological function. We show that PertSpectra provides an integrated modeling approach to understanding
combinatorial interventional data in the context of drug discovery.

## Directory Overview
Outline of the organization of the codebase.
* `.`: Contains notebooks and helper functions for downstream analyses
* `./src`: Contains the PertSpectra code, edited from the Spectra codebase
* `./data`: Contains instructions for downloading datasets used in analysis
* `./data_preprocessing`: Contains notebooks for preprocessing the input data
* `./PertSpectra_training_scripts`: Contains training scripts for PertSpectra
* `./PertSpectra_load_checkpoints`: Contains notebooks for loading trained PertSpectra models from checkpoints
* `./scETM`: Contains notebooks for training scETM
* `./GSFA`: Contains notebooks for preprocessing and training GSFA
* `./figures`: Contains notebooks for figures

## Environment Setup
PertSpectra has been tested on Linux systems.

Please execute:

```
make install
```

This will generate a conda environment called `pertspectra` and an associated jupyter kernel that
can be used to execute the notebooks in this code repository.


## Data / Setup
Data preprocessing and setup.
### Gene expression normalization
The model expects log-normalized count data. Ensure that the log-normalized expression is either in the `.X` field or `.layers` field of the anndata.

### Stringdb graph pruning
The model accepts an adjacency matrix as a prior to regularize against during training. Currently, the model regularizes against a stringdb prior graph - the notebooks to subset the stringdb graph to the input genes measured in the perturb-seq experiment are located in the `prior_graph_preprocessing/` subdirectory. Otherwise, create any prior graph as desired as a sparse adjacency matrix and store under `.uns["sparse_gene_network"]` in the anndata.

## Training
Overview of training and saving the model.
### Launching training runs
Training scripts for PertSpectra are located in `./PertSpectra_training_scripts`. Run
``` python3 [training script] [args]```
to launch a training run.

### Model loading
The model can be loaded with a helper function in `utils.py` (refer to `utils.py` for details):
```
wrapper, adata = load_model(
                   adata=adata,
                   s3_dir='s3_directory_where_training_runs_are_stored',
                   experiment_name='folder_in_s3_directory_where_training_run_is_located',
                   model_name='name_of_saved_model_pickle',
                   markers_top_n=50,
                   use_cell_types=False,
                )
```
From the loaded model, we also need to reconstruct the binarized perturbation matrix (cell x perturbation), as the binarization may differ across different models. For details on reconstructing the binarized perturbation matrix, refer to the analysis notebooks.

The returned anndata from `load_model` also saves the following outputs from the model:
* `adata.uns['SPECTRA_pert_scores']` stores the learned perturbation-level factors
* `adata.uns['SPECTRA_factors']` stores the learned gene-level factors

For full details on loading saved PertSpectra models, see `./PertSpectra_load_checkpoints`.

## Downstream analysis
Please reference the Jupyter notebooks in the main directory for all code relating to downstream analysis.

## Figure Generation
Code to generate figures based on the downstream analyses located in `./figures`.
