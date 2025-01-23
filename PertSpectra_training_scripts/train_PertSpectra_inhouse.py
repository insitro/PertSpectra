import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
from src.Spectra import Spectra_Pert as spc_linear
from src.Spectra.Spectra_Pert import vectorize_perts, vectorize_perts_combinations
from utils import (
    filter_noisy_genes,
    generate_k_fold,
    generate_loss_weights,
    read_aws_h5ad,
    set_seed,
    write_model_pickle_to_s3,
)


def train(args):
    # read in anndata containing prior graph
    unfilterd_adata = read_aws_h5ad(args.anndata_s3_url)
    adata = filter_noisy_genes(unfilterd_adata)
    adata.layers["logcounts"] = adata.X.copy()
    adata.X = adata.X.todense()
    device = torch.device(args.device)
    gene_network = adata.uns["sparse_gene_network"].todense()

    # filter adata to perturbations with at least 50 samples for each treatment
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    adata.obs["Treatment"] = adata.obs["Treatment"].astype(str)
    adata.obs["pert_treat"] = adata.obs["condition"] + "+" + adata.obs["Treatment"]
    obs_df = pd.DataFrame(adata.obs["pert_treat"])
    category_counts = obs_df["pert_treat"].value_counts()
    filtered_categories = category_counts[category_counts >= 50].index
    adata = adata[adata.obs["pert_treat"].isin(filtered_categories)]

    if args.cell_type_key == "Treatment":
        L = {"global": args.global_latent_dim}
        adj_matrices = {"global": gene_network.copy()}
        for key in adata.obs[args.cell_type_key].unique():
            L[key] = args.celltype_latent_dim
            adj_matrices[key] = gene_network.copy()

    # init model parameters
    vocab = adata.var_names
    labels = adata.obs[args.cell_type_key].values
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    X = adata.X
    loss_weights = generate_loss_weights(adata, args.perturbation_key)

    # perturbation labels
    if args.perturbation_key in adata.obs:
        if not args.encode_combos_as_unique:
            D, pert_labels = vectorize_perts(
                adata, args.perturbation_key, args.control_key
            )
            pert_idx = np.array(
                [
                    adata.var_names.get_loc(i.split("_")[1])
                    if i.split("_")[1] in adata.var_names
                    else -1
                    for i in pert_labels
                ]
            )
        else:
            D, pert_labels = vectorize_perts_combinations(
                adata, args.perturbation_key, args.control_key
            )
            pert_idx = np.array(
                [
                    adata.var_names.get_loc(i) if i in adata.var_names else -1
                    for i in pert_labels
                ]
            )

        # add ctrl one-hot-encoding
        ctrl_vector = np.array(
            [1.0 if i == "ctrl" else 0.0 for i in adata.obs[args.perturbation_key]]
        )
        basal_vector = np.array([1.0 for i in adata.obs[args.perturbation_key]])
        D = np.concatenate(
            [D, ctrl_vector.reshape(len(ctrl_vector), 1)], axis=1
        ).astype(np.float32)
        D = np.concatenate(
            [D, basal_vector.reshape(len(basal_vector), 1)], axis=1
        ).astype(np.float32)
        pert_idx = np.append(pert_idx, [-1, -1])
        pert_labels = pert_labels + ["ctrl", "basal"]
    else:
        print("Perturbation key not found or not given!")
        D = np.array([])
        pert_idx = np.array([])

    # init Spectra wrapper
    wrapper = spc_linear.SPECTRA_Model(
        X=X,
        labels=labels,
        pert_idx=pert_idx,
        pert_labels=pert_labels,
        L=L,
        vocab=vocab,
        adj_matrix=adj_matrices,
        use_weights=args.use_weights,
        lam=args.lam,
        psi=args.psi,
        delta=args.delta,
        kappa=None,
        rho=args.rho,
        use_cell_types=args.use_cell_types,
    )
    wrapper.initialize(None, word2id, adata.X, 0)
    wrapper.internal_model.to(device)

    # train-val-test split
    train_idx, val_idx, test_idx = generate_k_fold(
        adata, X, adata.obs[args.perturbation_key], fold_idx=args.kfold_idx
    )
    X_train = X[train_idx]
    D_train = D[train_idx]
    loss_weights_train = loss_weights[train_idx]
    labels_train = labels[train_idx]
    X_val = X[val_idx]
    D_val = D[val_idx]
    loss_weights_val = loss_weights[val_idx]
    labels_val = labels[val_idx]

    # train model
    train_loss, val_loss = wrapper.train(
        X_train,
        D_train,
        loss_weights_train,
        X_val,
        D_val,
        loss_weights_val,
        labels=labels_train,
        labels_val=labels_val,
    )

    write_model_pickle_to_s3(
        s3_url=args.model_save_s3_url + args.experiment_name + "/",
        model_name=f"kfold_{args.kfold_idx}",
        model=wrapper,
    )


if __name__ == "__main__":
    # set seed for reproducibility
    set_seed(0)
    parser = argparse.ArgumentParser(description="Run pertspectra experiment.")
    # Passing in hyperparameters as arguments
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--psi", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--kappa", type=float, default=0.00001)
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--global_latent_dim", type=int, default=20)
    parser.add_argument("--celltype_latent_dim", type=int, default=5)
    parser.add_argument("--use_highly_variable", type=bool, default=False)
    parser.add_argument("--use_weights", type=bool, default=True)
    parser.add_argument("--kfold_idx", type=int, default=0)

    # cell type
    parser.add_argument(
        "--use_cell_types",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    # encode combos as unique
    parser.add_argument(
        "--encode_combos_as_unique",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    # cell type key: Treatment, treatment_and_pert
    parser.add_argument("--cell_type_key", type=str, default="Treatment")
    # perturbation key if using perturbations
    parser.add_argument("--perturbation_key", type=str, default="condition")
    # control key
    parser.add_argument("--control_key", type=str, default=["ctrl", "nan"])
    # prior to use: None, stringdb
    parser.add_argument("--prior", type=str, default="stringdb")
    # device to use
    parser.add_argument("--device", type=str, default="cuda:0")
    # name of wandb run
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pertspectra_inhouse",
    )

    parser.add_argument(
        "--anndata_s3_url",
        type=str,
        default="s3://pert-spectra/inhouse_adata_spectra.h5ad",
    )
    parser.add_argument(
        "--model_save_s3_url",
        type=str,
        default="s3://pert-spectra/PertSpectra_checkpoints/",
    )
    args = parser.parse_args()

    train(args)
