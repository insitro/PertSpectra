### graph functions ####
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy
import torch

# from adjustText import adjust_text


def _compute_geary(L: torch.Tensor, Z: torch.Tensor):
    """Compute Geary's C multivariate autocorrelation.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian.
    Z : np.ndarray
        Feature matrix.
    Returns
    -------
    np.ndarray
        Geary's C values.
    """
    c = torch.linalg.multi_dot([Z.T, L, Z]).diagonal()

    return c


def geary_autocorrelation_multivariate(
    G: torch.Tensor,
    X: torch.Tensor,
    max_z: int = 100,
):
    """Compute Geary's C multivariate autocorrelation.

    Parameters
    ----------
    G : torch.Tensor
        Graph or adjacency matrix.
    X : torch.Tensor
        Feature matrix.
    max_z : int
        Maximum value for z score, by default 100

    Returns
    -------
    pd.Series
        Geary's C values.
    """

    n, p = X.shape
    d = torch.sum(G, dim=0)
    D = torch.diag(d)
    L = D - G

    Z = (X - X.mean(dim=0)) / X.std(dim=0)
    Z = Z.nan_to_num()

    c_unnormalized = _compute_geary(L, Z)
    c_normalized = c_unnormalized / d.sum()  # per annotation

    out_dict = {
        "stat": c_normalized.sum(),
        "pval": 1,
        "z": 0,
        "stat_annotation": c_normalized,
    }

    return out_dict


"""
methods
_______

amatrix(gene_set_list, gene2id)

amatrix_weighted(gene_set_list, gene2id)

unravel_dict(dict_)

process_gene_sets()

process_gene_sets_no_celltypes()

overlap_coefficient()

label_marker_genes()


"""


def overlap_coefficient(list1, list2):
    """
    Computes overlap coefficient between two lists
    """
    intersection = len(list(set(list1).intersection(set(list2))))
    union = min(len(list1), len(list2))  # + len(list2)) - intersection
    return float(intersection) / union


def check_gene_set_dictionary(
    adata,
    annotations,
    obs_key="cell_type_annotations",
    global_key="global",
    return_dict=True,
    min_len=3,
    use_cell_types=True,
):
    """
    Filters annotations dictionary to contain only genes contained in the
    adata.
    Checks that annotations dictionary cell type keys and adata cell types
    are identical.
    Checks that all gene sets in annotations dictionary contain >2 genes
    after filtering.


    adata: AnnData , data to use with Spectra
    annotations: dict , gene set annotations dictionary to use with Spectra
    obs_key: str , column name for cell type annotations in adata.obs
    global_key: str , key for global gene sests in gene set annotation
    dictionary
    return_dict: bool , return filtered gene set annotation dictionary
    min_len: int, minimum length of gene sets

    returns: dict , filtered gene set annotation dictionary

    """
    # test if keys match
    if use_cell_types:
        adata_labels = list(set(adata.obs[obs_key])) + [
            global_key
        ]  # cell type labels in adata object
    else:
        annotations = {global_key: annotations}
        adata_labels = [global_key]
    annotation_labels = list(annotations.keys())
    #     matching_celltype_labels = list(set(adata_labels).intersection(annotation_labels))
    if set(annotation_labels) != set(adata_labels):
        missing_adata = set(adata_labels) - set(annotation_labels)
        missing_dict = set(annotation_labels) - set(adata_labels)
        raise ValueError(
            "The following adata labels are missing in the gene set \
            annotation dictionary:",
            missing_dict,
            "The following gene set annotation dictionary keys are \
            missing in the adata labels:",
            missing_adata,
        )
        dict_keys_OK = False
    else:
        print(
            "Cell type labels in gene set annotation dictionary and \
            AnnData object are identical"
        )
        dict_keys_OK = True

    # check that gene sets in dictionary have len >2
    annotations_new = {}
    for k, v in annotations.items():
        annotations_new[k] = {}
        for k2, v2 in v.items():
            gs = [x for x in v2 if x in adata.var_names]
            if len(gs) < min_len:
                print(
                    "removing gene set",
                    k2,
                    "for cell type",
                    k,
                    "which is of length",
                    len(v2),
                    len(gs),
                    "genes are found in the data.",
                    "minimum length is",
                    min_len,
                )
            else:
                annotations_new[k][k2] = gs

    # raise error if no gene sets remain
    if not use_cell_types and len(annotations_new[global_key]) == 0:
        raise ValueError(
            "No gene sets remain in the gene set annotation dictionary. \
            Please make sure that gene names correspond to names found \
            in `adata.var_names`. See: https://github.com/dpeerlab/spectra/issues/34."
        )

    if dict_keys_OK:
        print("Your gene set annotation dictionary is now correctly formatted.")
    if return_dict:
        if not use_cell_types:
            return annotations_new[global_key]
        else:
            return annotations_new


def label_marker_genes(marker_genes, gs_dict, threshold=0.4):
    """
    label an array of marker genes using the gene_set_dictionary in est_spectra
    returns a dataframe of overlap coefficients for each gene set annotation
    and marker gene
    marker_genes: array factors x marker genes or a KnowledgeBase object
    label an array containing marker genes by its overlap with a dictionary of
    gene sets from the knowledge base:
    KnowledgeBase.celltype_process_dict
    """

    overlap_df = pd.DataFrame()
    marker_set_len_dict = {}  # len of gene sets to resolve ties
    for i, v in pd.DataFrame(marker_genes).T.items():
        overlap_temp = []
        gs_names_temp = []

        for gs_name, gs in gs_dict.items():
            marker_set_len_dict[gs_name] = len(gs)
            overlap_temp.append(overlap_coefficient(set(gs), set(v)))
            gs_names_temp.append(gs_name)
        overlap_df_temp = pd.DataFrame(overlap_temp, columns=[i], index=gs_names_temp).T
        overlap_df = pd.concat([overlap_df, overlap_df_temp])
    overlap_df.loc["gene_set_length"] = list(
        overlap_df.columns.map(marker_set_len_dict)
    )

    # find maximum overlap coefficient gene set label for each factor, resolve
    # ties by gene set length
    marker_gene_labels = []  # gene sets

    marker_gene_list = list(overlap_df.index)
    marker_gene_list.remove("gene_set_length")
    for marker_set in marker_gene_list:
        # resolve ties in overlap_coefficient by selecting the bigger gene set
        max_overlap = (
            overlap_df.loc[[marker_set, "gene_set_length"]]
            .T.sort_values(by="gene_set_length", ascending=True)
            .sort_values(by=marker_set, ascending=True)[marker_set]
            .index[-1]
        )

        if overlap_df.loc[marker_set].sort_values().values[-1] > threshold:
            marker_gene_labels.append(max_overlap)
        else:
            marker_gene_labels.append(marker_set)
    overlap_df = overlap_df.drop(index="gene_set_length")
    overlap_df.index = marker_gene_labels
    return overlap_df


def amatrix(gene_set_list, gene2id):
    """
    creates adjacency matrix from gene set list
    """
    n = len(gene2id)
    adj_matrix = np.zeros((n, n))
    for gene_set in gene_set_list:
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                if (g1 in gene2id) & (g2 in gene2id):
                    adj_matrix[gene2id[g1], gene2id[g2]] = 1
    return adj_matrix


def amatrix_weighted(gene_set_list, gene2id):
    """
    Creates weighted adjacency matrix from gene sets
    uses 1/{n choose 2} as edge weights - edge weights accumulate additively
    """
    n = len(gene2id)
    adj_matrix = np.zeros((n, n))
    ws = []
    for gene_set in gene_set_list:
        if len(gene_set) > 1:
            w = 1.0 / (len(gene_set) * (len(gene_set) - 1) / 2.0)
        else:
            w = 1.0
        ws.append(w)
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                if (g1 in gene2id) & (g2 in gene2id):
                    adj_matrix[gene2id[g1], gene2id[g2]] += w
    med = np.median(np.array(ws))
    return adj_matrix / float(med)


def unravel_dict(dict_):
    lst = []
    for key in dict_.keys():
        lst.append(dict_[key])
    return lst


def process_gene_sets(gs_dict, gene2id, weighted=True):
    """
    { "global": {"<gene set name>": [<gene 1>, <gene 2>, ...]}
    }
    """
    adict = OrderedDict()
    adict["global"] = amatrix(unravel_dict(gs_dict["global"]), gene2id)
    weights = None

    if weighted:
        weights = OrderedDict()
        weights["global"] = amatrix_weighted(unravel_dict(gs_dict["global"]), gene2id)

    for key in gs_dict.keys():
        if len(gs_dict[key]) > 0:
            adict[key] = amatrix(unravel_dict(gs_dict[key]), gene2id)
            if weighted:
                weights[key] = amatrix_weighted(unravel_dict(gs_dict[key]), gene2id)
        else:
            adict[key] = []
            if weighted:
                weights[key] = []

    return adict, weights


def process_gene_sets_no_celltypes(gs_dict, gene2id, weighted=True):
    """
    input: {"<gene set name>": [<gene 1>, <gene 2>, ...]}
    }
    gene2id {gene name: index in vocab}

    weighted: whether to return NoneType or weighted adjacency matrix
    """
    adict = amatrix(unravel_dict(gs_dict), gene2id)
    weights = None
    if weighted:
        weights = amatrix_weighted(unravel_dict(gs_dict), gene2id)
    return adict, weights


def get_factor_celltypes(adata, obs_key, cellscore_obsm_key="SPECTRA_cell_scores"):
    """
    Assigns Spectra factors to cell types by analyzing the factor cell scores.
    Cell type specific factors will have zero cell scores except in their respective cell type

    adata: AnnData , object containing the Spectra output
    obs_key: str , column name in adata.obs containing the cell type annotations
    cellscore_obsm_key: str , key for adata.obsm containing the Spectra cell scores

    returns: dict , dictionary of {factor index : 'cell type'}
    """

    # get cellscores
    import pandas as pd

    cell_scores_df = pd.DataFrame(adata.obsm[cellscore_obsm_key])
    cell_scores_df["celltype"] = list(adata.obs[obs_key])

    # find global and cell type specific fators
    global_factors_series = (cell_scores_df.groupby("celltype").mean() != 0).all()
    global_factors = [
        factor
        for factor in global_factors_series.index
        if global_factors_series[factor]
    ]
    specific_cell_scores = (
        (cell_scores_df.groupby("celltype").mean()).T[~global_factors_series].T
    )
    specific_factors = {}

    for i in set(cell_scores_df["celltype"]):
        specific_factors[i] = [
            factor
            for factor in specific_cell_scores.loc[i].index
            if specific_cell_scores.loc[i, factor]
        ]

    # inverse dict factor:celltype
    factors_inv = {}
    for i, v in specific_factors.items():
        for factor in v:
            factors_inv[factor] = i

    # add global

    for factor in global_factors:
        factors_inv[factor] = "global"

    return factors_inv


## importance and information score functions


def mimno_coherence_single(w1, w2, W):
    #     eps = 0.01
    dw1 = W[:, w1] > 0
    dw2 = W[:, w2] > 0
    #     N = W.shape[0]

    dw1w2 = (dw1 & dw2).float().sum()
    dw1 = dw1.float().sum()
    dw2 = dw2.float().sum()
    if (dw1 == 0) | (dw2 == 0):
        return -0.1 * np.inf
    return ((dw1w2 + 1) / (dw2)).log()


def mimno_coherence_2011(words, W):
    score = 0
    V = len(words)

    for j1 in range(1, V):
        for j2 in range(j1):
            score += mimno_coherence_single(words[j1], words[j2], W)
    denom = V * (V - 1) / 2
    return score / denom


def get_information_score(adata, idxs, cell_type):
    if "spectra_vocab" not in adata.var.columns:
        print("adata requires spectra_vocab attribute.")
        return None

    idx_to_use = adata.var["spectra_vocab"]
    X = adata.X[:, idx_to_use]
    if X is scipy.sparse.csr.csr_matrix:
        X = np.array(X.todense())
    X = torch.Tensor(X)
    lst = []
    # for j in range(idxs.shape[0]):
    # lst.append(mimno_coherence_2011(list(idxs[j,:]), X[labels == cell_type]))
    return lst
