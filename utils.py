import itertools
import os
import pickle
import random
import re
from typing import Any, List

import anndata as ad
import boto3
import botocore
import numpy as np
import pandas as pd
import torch
from gprofiler import GProfiler
from scipy import sparse
from scipy.stats import false_discovery_control, hypergeom
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split

SPECTRA_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "pertspectra_cache")
GSEA_COLUMN_DATATYPES_FOR_SERIALIZATION = {
    "Name": "category",
    "Term": "category",
    "ES": "float32",
    "NES": "float32",
    "NOM p-val": "float32",
    "FDR q-val": "float32",
    "FWER p-val": "float32",
    "Tag %": "str",
    "Gene %": "str",
    "Lead_genes": "str",
    "gsea_weight": "category",
    "pval": "float32",
    "fdr_bh": "float32",
    "fwer_bf": "float32",
    "GO_ID": "category",
    "Lead_g_id": "str",
}


def read_aws_h5ad(s3_url):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    s3_object.download_file(save_path)

    adata = ad.read_h5ad(save_path)
    return adata


def read_aws_csv(s3_url, sep=",", zipped=False, header="infer"):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        if not zipped:
            df = pd.read_csv(save_path, sep=sep, header=header)
        else:
            df = pd.read_csv(
                save_path, compression="gzip", delimiter="\t", header=header
            )
        return df
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None


def read_aws_npz(s3_url, sep=","):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        mtx = np.load(save_path)
        return mtx
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None


def read_aws_pickle(s3_url, sep=","):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        with open(save_path, "rb") as f:
            pickle_obj = pickle.load(f)
        return pickle_obj
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None


def read_aws_json(s3_url):
    import json

    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        f = open(save_path)
        data = json.load(f)
        return data
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None


def write_model_pickle_to_s3(s3_url, model_name, model):
    model_path = f"{model_name}.pickle"
    temp_path = os.path.join(SPECTRA_DEFAULT_DIR, model_path)

    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    s3 = boto3.resource("s3")

    with open(temp_path, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)
    # Specify model version in key
    key = key + model_path
    # Write model to s3
    s3.Object(bucket_name=bucket_name, key=key).put(Body=open(temp_path, "rb"))


def write_adata_to_s3(s3_url, adata_name, adata):
    adata_path = f"{adata_name}.h5ad"
    temp_path = os.path.join(SPECTRA_DEFAULT_DIR, adata_path)

    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    s3 = boto3.resource("s3")

    adata.write_h5ad(temp_path)
    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)
    # Specify model version in key
    key = key + adata_path
    # Write model to s3
    s3.Object(bucket_name=bucket_name, key=key).put(Body=open(temp_path, "rb"))


def read_model_pickle_from_s3(s3_url, model_name):
    model_path = f"{model_name}.pickle"
    temp_path = os.path.join(SPECTRA_DEFAULT_DIR, model_path)

    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)
    # Specify model version in key
    key = key + model_path

    # Read model from s3
    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(temp_path)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("Model could not be found")
            return None

    with open(temp_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_model(
    adata,
    s3_dir,
    experiment_name,
    model_name,
    use_cell_types=False,
    cell_type_key="",
):
    """
    Loads trained model

    adata: AnnData, data to store model results
    s3_dir: str, directory where the model is stored
    experiment_name: str, name of experiment (same as wandb name)
    model_name: str, name of model
    use_cell_types: bool, boolean if model used cell types
    cell_type_key: str, cell type key is use_cell_types==True

    returns:
    trained model: SPECTRA_Model instance
    anndata with saved parameters: contains the following fields:
     -

    """
    # load model from checkpoint
    wrapper = read_model_pickle_from_s3(s3_dir + experiment_name, model_name)

    # initialize Spectra wrapper
    if use_cell_types:
        labels = np.array(adata.obs[cell_type_key])
        wrapper._SPECTRA_Model__store_parameters(labels)
    else:
        wrapper._SPECTRA_Model__store_parameters_no_celltypes()

    # save parameters
    # vocab = adata.var_names
    # id2word = dict((idx, v) for idx, v in enumerate(vocab))
    # word2id = dict((v, idx) for idx, v in enumerate(vocab))
    adata.uns["SPECTRA_factors"] = wrapper.factors
    adata.uns["SPECTRA_L"] = wrapper.internal_model.L
    adata.uns["SPECTRA_pert_scores"] = wrapper.cell_scores

    return wrapper, adata


# +
# preprocess perturbations for rna565 - leave intergenics as separate
# reformat guide gene naming conventions: ctrl for control, + delimiting the guides
def replace_ctrl_words(s):
    # Define the regex pattern
    pattern = r"\bCTRL00\w*\b"

    # Check if the string is 'nan'
    if s == "nan":
        return s

    else:
        rep = re.sub(pattern, "ctrl", s)
        return rep


def replace_intergenic_words(s):
    # Define the regex pattern
    pattern = r"\bINTERGENIC\w*\b"

    # Check if the string is 'nan'
    if s == "nan":
        return s

    else:
        rep = re.sub(pattern, "intergenic", s)
        return rep


def inhouse_preprocess(adata):
    guides = np.array(adata.obs["target_gene_name"])
    # leave nans, reformat | as +, and replace CTRL as controls
    # filtered_nan_guides = np.where(guides == "nan", "ctrl", guides)
    filtered_delimiter_guides = np.array([x.replace("|", "+") for x in guides])
    v_replace_intergenic_words = np.vectorize(replace_intergenic_words)
    filtered_delimiter_guides = v_replace_intergenic_words(filtered_delimiter_guides)
    v_replace_ctrl_words = np.vectorize(replace_ctrl_words)
    filtered_delimiter_guides = v_replace_ctrl_words(filtered_delimiter_guides)
    adata.obs["condition"] = filtered_delimiter_guides
    adata.obs["condition"] = np.where(
        adata.obs["condition"] == "ctrl+ctrl", "ctrl", adata.obs["condition"]
    )

    # add control column
    condition = np.array(adata.obs["condition"])
    controls = np.where(condition == "ctrl", 1, 0)
    adata.obs["control"] = controls

    # # reformat singletons
    for i, guide in enumerate(adata.obs["condition"]):
        if ("ctrl" in guide) and (guide.count("+") == 1):
            pert = guide.split("+")
            if pert[0] == "ctrl":
                adata.obs["condition"][i] = pert[1]
            else:
                adata.obs["condition"][i] = pert[0]
    return adata


def filter_noisy_genes(adata):
    """
    Filter noisy genes from anndata - both the expression and graph
    """
    # filter noise genes
    noise_prefixes = set(["RPL", "RPS", "MT-", "NEAT1", "MALAT1", "NDUF"])

    def prefix_match(name, taglist):
        taglist = tuple(taglist)
        if name.startswith(taglist):
            return name
        return None

    relevant_gene_idx = []
    for i, x in enumerate(adata.var_names):
        match = prefix_match(x, noise_prefixes)
        if not match:
            relevant_gene_idx.append(i)

    adata = adata[:, relevant_gene_idx]
    adata.uns["sparse_gene_network"] = sparse.csr_matrix(
        adata.uns["sparse_gene_network"].todense()[relevant_gene_idx][
            :, relevant_gene_idx
        ]
    )

    return adata


def split_data_by_cell(X, D, test_size=0.2, val_size=0.2):
    """
    Split data into train/val/test by cells (seeing all perturbations in training)
    """
    data_idx = [i for i in range(X.shape[0])]
    train_val_idx, test_idx, D_train_val, D_test = train_test_split(
        data_idx, D, test_size=test_size, random_state=1, stratify=D
    )
    train_idx, val_idx, D_train, D_val = train_test_split(
        train_val_idx,
        D_train_val,
        test_size=val_size,
        random_state=1,
        stratify=D_train_val,
    )

    return train_idx, val_idx, test_idx


def split_data_by_combinations(
    adata,
    X,
    D,
    perturbation_key="condition",
    intergenic="intergenic",
    test_size=0.3,
    val_size=0.2,
):
    """
    Split data into train/val/test by perturbation (holdout some combinations)
    """
    pert_list = list(adata.obs[perturbation_key].unique())
    combo_perts = [i for i in pert_list if ("+" in i) and (intergenic not in i)]
    single_perts = [i for i in pert_list if ("+" not in i) or (intergenic in i)]

    single_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in single_perts
    ]
    D_single = D[single_idx]

    train_idx_single, val_idx_single, D_train_single, D_val_single = train_test_split(
        single_idx, D_single, test_size=test_size, random_state=1, stratify=D_single
    )
    train_val_combos, test_combos = train_test_split(
        combo_perts, test_size=val_size, random_state=1
    )
    train_val_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in train_val_combos
    ]
    test_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in test_combos
    ]
    D_train_val = D[train_val_idx]
    train_idx_c, val_idx_c, _, _ = train_test_split(
        train_val_idx, D_train_val, test_size=0.2, random_state=1, stratify=D_train_val
    )
    train_idx = train_idx_single + train_idx_c
    val_idx = val_idx_single + val_idx_c

    return train_idx, val_idx, test_idx


def generate_k_fold(
    adata,
    X,
    D,
    perturbation_key="condition",
    intergenic="intergenic",
    folds=5,
    fold_idx=0,
    test_size=0.2,
    val_size=0.2,
):
    """
    Split data into folds
    """
    pert_list = list(adata.obs[perturbation_key].unique())
    combo_perts = [i for i in pert_list if ("+" in i)]
    single_perts = [i for i in pert_list if ("+" not in i)]

    # singletons
    single_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in single_perts
    ]
    D_single = D[single_idx]
    train_val_single_idx, test_single_idx, D_train_val_single, D_test_single = (
        train_test_split(
            single_idx, D_single, test_size=test_size, random_state=1, stratify=D_single
        )
    )
    train_single_idx, val_single_idx, _, _ = train_test_split(
        train_val_single_idx,
        D_train_val_single,
        test_size=val_size,
        random_state=1,
        stratify=D_train_val_single,
    )

    # combos - kfold
    kf = KFold(n_splits=folds, random_state=1, shuffle=True)
    kf_splits = kf.split(combo_perts)
    train_val_combos_idx, test_combos_idx = next(
        itertools.islice(kf_splits, fold_idx, None)
    )
    train_val_combos = np.array(combo_perts)[train_val_combos_idx]
    test_combos = np.array(combo_perts)[test_combos_idx]

    train_val_combos_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in train_val_combos
    ]
    test_combos_idx = [
        i for i, x in enumerate(adata.obs[perturbation_key]) if x in test_combos
    ]
    D_train_val = D[train_val_combos_idx]
    train_idx_c, val_idx_c, _, _ = train_test_split(
        train_val_combos_idx,
        D_train_val,
        test_size=0.2,
        random_state=1,
        stratify=D_train_val,
    )

    train_idx = train_single_idx + train_idx_c
    val_idx = val_single_idx + val_idx_c
    test_idx = test_single_idx + test_combos_idx

    return train_idx, val_idx, test_idx


def set_seed(seed: int) -> None:
    """Sets the random seed to seed.

    Args:
        seed: the random seed.
    """

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def generate_loss_weights(adata, perturbation_key):
    """
    Generate loss weights for weighting based on fraction of perturbations

    Parameters:
    - adata: full anndata
    - perturbation_key: key for perturbation annotations
    """
    # generate weights for losses - inverse of number of cells
    loss_weights = {}
    for x in adata.obs[perturbation_key].unique():
        weight = 1 / adata.obs[perturbation_key].value_counts()[x]
        loss_weights[x] = weight
    sample_weights = np.array([loss_weights[i] for i in adata.obs[perturbation_key]])
    return sample_weights


#### Interpretability Analyses ####
GPROFILER_SOURCES = [
    "GO:MF",
    "GO:CC",
    "GO:BP",
    "REAC",
    "WP",
    "TF",
    "MIRNA",
    "HPA",
    "CORUM",
    "HP",
]


def get_gprofiler(
    de_genes: pd.DataFrame,
    organism: str = "hsapiens",
    sources: List[str] = GPROFILER_SOURCES,
    no_evidences: bool = False,
) -> pd.DataFrame:
    """
    Perform gene set enrichment using gprofiler.
    de_genes should have a column "gene_symbol" containing gene symbols.
    """
    gp = GProfiler(return_dataframe=True)
    result = gp.profile(
        query=list(de_genes["gene_symbol"]),
        organism=organism,
        sources=sources,
        no_evidences=no_evidences,
    )
    go_to_gene = read_aws_pickle("s3://pert-spectra/references/GO_to_Gene.pickle")
    filtered_goterms = list(go_to_gene.keys())
    result = result[result.native.isin(filtered_goterms)]
    return result


def retrieve_stringdb_neighbors(genes: List = []):
    """
    Use StringDB to retrieve functional neighbors for perturbations
    """
    # retrieve stringdb
    PERTSPECTRA_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "pertspectra_cache")
    stringdb_s3_url = "s3://pert-spectra/references/StringDB.HQ.txt"
    save_path = os.path.join(PERTSPECTRA_DEFAULT_DIR, stringdb_s3_url.split("/")[-1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")
    # Get the bucket name and key from the s3 url
    bucket_name, key = stringdb_s3_url.removeprefix("s3://").split("/", 1)
    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    s3_object.download_file(save_path)
    stringdb_hq = pd.read_csv(save_path, sep="\t")

    perts = [x for x in genes if x not in ["ctrl", "intergenic", "basal"]]
    pert_neighbors: dict[Any, Any] = {key: set() for key in perts}
    for index, row in stringdb_hq.iterrows():
        if row["x"] > 0.8:
            if row["i_genes"] in pert_neighbors.keys():
                pert_neighbors[row["i_genes"]].add(row["j_genes"])
            elif row["j_genes"] in pert_neighbors.keys():
                pert_neighbors[row["j_genes"]].add(row["i_genes"])
    return pert_neighbors


def run_gsea(
    de_genes: pd.DataFrame,
    min_size: int = 10,
    max_size: int = 500,
    weighted_score_type: int = 0,
    permutation_num: int = 1000,
    ascending: bool = False,
    no_plot: bool = True,
    processes: int = -1,
    verbose: bool = True,
    seed: int = 0,
    gsea_name_col: str = "gene_symbol",
):
    import gseapy as gp
    from statsmodels.stats.multitest import multipletests

    gsea_inputs = de_genes[[gsea_name_col, "z"]].sort_values("z", ascending=False)
    gene_sets = read_aws_pickle("s3://pert-spectra/references/GO_to_Gene.pickle")

    # Run prerank
    pre_res = gp.prerank(
        gsea_inputs,
        gene_sets=gene_sets,
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        weighted_score_type=weighted_score_type,
        permutation_num=permutation_num,
        ascending=ascending,
        no_plot=no_plot,
        verbose=verbose,
        threads=processes,
        seed=seed,
    )
    result = pre_res.res2d

    # Adjust p-value
    result["gsea_weight"] = weighted_score_type

    result["pval"] = (result["NOM p-val"] * permutation_num + 1) / (permutation_num + 1)
    result["fdr_bh"] = multipletests(result["pval"], method="fdr_bh")[1]
    result["fwer_bf"] = np.clip(result["pval"] * result.shape[0], 0, 1)
    result["GO_ID"] = result["Term"]
    # adjust data types to enable serialization to h5ad if gsea output is added to .uns
    # attribute of anndata
    columns_to_format = {
        k: v
        for (k, v) in GSEA_COLUMN_DATATYPES_FOR_SERIALIZATION.items()
        if k in result.columns
    }
    result = result.astype(columns_to_format)

    return result, pre_res


def factor_enrichment_gsea(adata, latent, max_size=300, fdr=5e-3):
    factor_to_go = {}
    latent = latent
    for i in range(len(latent)):
        # rank by latent factor loadings
        lvec = latent[i]
        gsea_input = pd.DataFrame(
            [adata.var_names, lvec],
            index=["gene_symbol", "z"],
        ).T
        # run gsea
        gsea_input["z"] = gsea_input["z"].astype("float")
        res = run_gsea(gsea_input, max_size=max_size)[0]
        # filter to BP
        go_reference = read_aws_csv(
            "s3://pert-spectra/references/GO_terms.txt.gz", zipped=True
        )
        go_bp = go_reference[go_reference["go_category"] == "biological_process"]
        go_bp_ids = set(go_bp["go_id"].values)
        # filter on fdr and nes
        res = res[res["fdr_bh"] <= fdr]
        res = res[np.abs(res["NES"]) > 1]
        res = res[res["GO_ID"].isin(go_bp_ids)]
        factor_to_go[i] = res
    return factor_to_go


def perturbation_signal_recovery(
    pert_to_go, model_pert_to_go, filtered_go_terms, perturbations
):
    """
    Returns p-value of bootstrapped hypergeoemtric test of the overlap between known processes
    vs model identified processes

    Args:
        pert_to_go (dict): dict of perturbations to GO terms from a prior
        model_pert_to_go (dict): dict of perturbations to GO terms from the model
        filtered_go_terms (list): list of all GO terms in the prior
        perturbations (list): list of perturbations

    Return:
        pvals (dict): dict of perturbation to corrected p-value
    """
    set_seed(0)
    pvals = {}
    for pert in perturbations:
        if pert in ["ctrl", "intergenic", "basal"]:
            continue
        groupA = pert_to_go[pert]
        groupB = model_pert_to_go[pert]
        # Total number of unique items
        M = len(filtered_go_terms)
        # Number of items in set1
        n = len(groupA)
        # Number of items in set2
        k = len(groupB)
        # Number of overlapping items (intersection of both sets)
        N = len(groupA.intersection(groupB))

        # only consider if there are >5 processes from researchdb
        if n < 5:
            continue
        rv = hypergeom(M, n, k)
        observed_p_value = rv.sf(N - 1)

        # Output the p-value
        print(f"Overlap for {pert}: {N} out of {n} in researchDB")
        print(f"P-value for {pert}: {observed_p_value}")
        if N == 0:
            pvals[pert] = 1
        else:
            pvals[pert] = observed_p_value

    # pvalue correction
    pval_list = list(pvals.values())
    pvals_corrected = false_discovery_control(pval_list)
    for i, key in enumerate(pvals):
        pvals[key] = pvals_corrected[i]
    return pvals


### Precision-recall analysis ###
def auprc(
    dist_matrix: pd.DataFrame,
    benchmark_sources: list = ["StringDB", "CORUM"],
    benchmark_data_dir: str = "s3://pert-spectra/references/recall_datasets/",
    log_stats: bool = False,
):
    """
    Return AUC and best F1 score+threshold of precision-recall curve
    """
    # convert distance matrix to sim matrix
    d_norm = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())
    sim_matrix = 1 - d_norm

    # calculate pr metric
    auc_metrics = {}
    f1_metrics = {}
    pr_metrics = {}
    # inputs = {}
    for s in benchmark_sources:
        rels = get_benchmark_relationships(benchmark_data_dir, s)
        rels = rels[
            rels.node_1.isin(sim_matrix.index) & rels.node_2.isin(sim_matrix.index)
        ]
        adj_true = np.array(
            [
                [0 for _ in range(len(sim_matrix.index))]
                for _ in range(len(sim_matrix.index))
            ]
        )
        adj_labels = {x: i for i, x in enumerate(sim_matrix.index)}
        for i in range(adj_true.shape[0]):
            adj_true[i][i] = 1
        for i, row in rels.iterrows():
            n1, n2 = row["node_1"], row["node_2"]
            adj_true[adj_labels[n1]][adj_labels[n2]] = 1
            adj_true[adj_labels[n2]][adj_labels[n1]] = 1
        fpr, tpr, thresholds = precision_recall_curve(
            np.reshape(adj_true.flatten(), (-1, 1)),
            np.reshape(sim_matrix.values.flatten(), (-1, 1)),
        )
        # calculate auc
        auc_metrics[s] = {auc(tpr, fpr)}
        # calculate best f1
        f1_scores = 2 * tpr * fpr / (tpr + fpr)
        f1_metrics[s] = [np.max(f1_scores), thresholds[np.argmax(f1_scores)]]
        # record pr metrics
        pr_metrics[s] = {"precision": fpr, "recall": tpr, "thresholds": thresholds}
        # record inputs
        # inputs[s] = {'adj_true':np.reshape(adj_true.flatten(), (-1, 1)),
    #'sim_matrix':np.reshape(sim_matrix.values.flatten(), (-1, 1))}

    return (
        pd.DataFrame.from_dict(auc_metrics, orient="index", columns=["AUC"]),
        pd.DataFrame.from_dict(f1_metrics, orient="index", columns=["F1", "Threshold"]),
        pr_metrics,
    )


### Recall Analyses borrowed from EFAAR###
def get_benchmark_relationships(benchmark_data_dir: str, src: str, filter=True):
    """
    Reads a CSV file containing benchmark data and returns a filtered DataFrame.

    Args:
        benchmark_data_dir (str): The directory containing the benchmark data files.
        src (str): The name of the source containing the benchmark data.
        filter (bool, optional): Whether to filter the DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark relationships.
    """
    df = read_aws_csv(benchmark_data_dir + src + ".txt")
    return filter_relationships(df) if filter else df


def convert_metrics_to_df(metrics: dict, source: str) -> pd.DataFrame:
    """
    Convert metrics dictionary to dataframe to be used in summary.

    Args:
        metrics (dict): metrics dictionary
        source (str): benchmark source name

    Returns:
        pd.DataFrame: a dataframe with metrics
    """
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def filter_relationships(df: pd.DataFrame):
    """
    Filters a DataFrame of relationships between entities, removing any rows with
    self-relationships
    , ie. where the same entity appears in both columns, and also removing any duplicate
    relationships (A-B and B-A).

    Args:
        df (pd.DataFrame): DataFrame containing columns 'entity1' and 'entity2', representing the
        entities involved in
        each relationship.

    Returns:
        pd.DataFrame: DataFrame containing columns 'entity1' and 'entity2', representing
        the entities
        involved in
        each relationship after removing any rows where the same entity appears in both columns.
    """
    df["sorted_entities"] = df.apply(
        lambda row: tuple(sorted([row.node_1, row.node_2])), axis=1
    )
    df["node_1"] = df.sorted_entities.apply(lambda x: x[0])
    df["node_2"] = df.sorted_entities.apply(lambda x: x[1])
    return df[["node_1", "node_2"]].query("node_1!=node_2").drop_duplicates()
