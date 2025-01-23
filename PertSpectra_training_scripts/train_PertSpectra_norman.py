import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
from src.Spectra import Spectra_Pert as spc_linear
from src.Spectra.Spectra_Pert import (
    get_guide_one_hot_cols,
    vectorize_perts_combinations,
)
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
    if args.prior == "None":
        gene_network = np.zeros(gene_network.shape)

    # filter adata to perturbations with at least 50 samples
    obs_df = pd.DataFrame(adata.obs[args.perturbation_key])
    category_counts = obs_df[args.perturbation_key].value_counts()
    filtered_categories = category_counts[category_counts >= 50].index
    adata = adata[adata.obs[args.perturbation_key].isin(filtered_categories)]

    # init model parameters
    L = args.global_latent_dim
    adj_matrices = gene_network.copy()
    vocab = adata.var_names
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    loss_weights = generate_loss_weights(adata, args.perturbation_key)
    X = adata.X

    # create binary perturbation label matrix from data
    if args.perturbation_key in adata.obs:
        if not args.encode_combos_as_unique:
            pert_labels = get_guide_one_hot_cols(adata.obs)
            adata.obs["num_guides"] = adata.obs[pert_labels].sum(1)
            # combinations encoded as application of two individual guides
            D = adata.obs[pert_labels].to_numpy().astype(np.float32)
            pert_id = []
            for i in pert_labels:
                guide = i.split("_")[1]
                if guide in adata.var_names:
                    pert_id.append(adata.var_names.get_loc(guide))
                else:
                    pert_id.append(-1)
            pert_idx = np.array(pert_id)
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
            [1.0 if i == "control" else 0.0 for i in adata.obs[args.perturbation_key]]
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
        labels=None,
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
    train_idx, val_idx, _ = generate_k_fold(
        adata, X, D, fold_idx=args.kfold_idx, perturbation_key=args.perturbation_key
    )
    X_train = X[train_idx]
    D_train = D[train_idx]
    loss_weights_train = loss_weights[train_idx]
    X_val = X[val_idx]
    D_val = D[val_idx]
    loss_weights_val = loss_weights[val_idx]

    # train model
    train_loss, val_loss = wrapper.train(
        X_train, D_train, loss_weights_train, X_val, D_val, loss_weights_val
    )
    # save to s3
    write_model_pickle_to_s3(
        s3_url=args.model_save_s3_url + args.experiment_name + "/",
        model_name=f"kfold_{args.kfold_idx}",
        model=wrapper,
    )


if __name__ == "__main__":
    # set seed for reproducibility
    set_seed(0)
    parser = argparse.ArgumentParser(description="Run pertspectra experiment.")
    # Passing in hyperparameters as arguments.
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lam", type=float, default=1)
    parser.add_argument("--psi", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--kappa", type=float, default=0.00001)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--use_highly_variable", type=bool, default=False)
    parser.add_argument("--use_weights", type=bool, default=True)
    parser.add_argument("--global_latent_dim", type=int, default=200)
    parser.add_argument("--kfold_idx", type=int, default=2)

    # cell type
    parser.add_argument("--use_cell_types", type=bool, default=False)
    # encode combos as unique
    parser.add_argument(
        "--encode_combos_as_unique",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    # perturbation key if using perturbations
    parser.add_argument("--perturbation_key", type=str, default="perturbation_name")
    # control key
    parser.add_argument("--control_key", type=str, default=["control"])
    # prior to use: None, stringdb
    parser.add_argument("--prior", type=str, default="stringdb")
    # device to use
    parser.add_argument("--device", type=str, default="cuda:0")
    # name of training run
    parser.add_argument("--experiment_name", type=str, default="pertspectra_norman")

    parser.add_argument(
        "--anndata_s3_url",
        type=str,
        default="s3://pert-spectra/norman_adata_spectra.h5ad",
    )
    parser.add_argument(
        "--model_save_s3_url",
        type=str,
        default="s3://pert-spectra/PertSpectra_checkpoints/",
    )
    args = parser.parse_known_args()[0]

    train(args)
