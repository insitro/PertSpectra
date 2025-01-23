from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opt_einsum import contract
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import Spectra_util  # noqa
from .initialization import compute_init_scores, compute_init_scores_noct


class SPECTRA(nn.Module):
    """

    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used
            for initialization of
            self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types ==
            False, then should
            be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key,
            indicating the number
            of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > Note that L contains the number of factors that describe the graph.
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...

            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then
        ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is non-zero
            when adj_matrix[i,j] = 0
            this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence of the
            graph vs expression
            loss functions. This term multiplies the expression loss, so smaller values
            of lambda upweight the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene scaling
            factors. If delta is small then the maximum ratio between gene scaling factors
            is larger and lowly expressed genes can be put on the same scale as highly
            expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa is a float,
            kappa is fixed to the given float value. If kappa == None, then kappa is a
            parameter that is estimated from the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho is a float,
            rho is fixed to
            the given float value. If rho == None, then rho is a parameter that is estimated
            from the data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell type labels
            are used to fit
            the model. If False, then parameters are initialized as nn.Parameter rather
            than as
            nn.ParameterDict with cell type keys that index nn.Parameter values
        determinant_penalty : float
            determinant penalty affects the selection parameters that are fit when
            L[cell_type] >
            K[cell_type]. A determinantally regularized selection parameter is fit
            with determinant
            penalty that encourages sparsity and diversity.
    Attributes
        ----------
        model.delta : delta parameter of the model

        model.lam : lambda parameter of the model

        model.determinant_penalty : determinant penalty of the model

        model.L : L parameter, either int, dict or OrderedDict()

        model.p : number of genes

        model.n : number of cells

        model.use_cell_types : if True then cell types are considered, else cell types
        ignored. Affects
        the dimensions of the initialized parameters.

        model.kappa : if not kappa, nn.ParameterDict() if use_cell_types, else
        nn.Parameter(). If
        kappa is a float, it is fixed throughout training

        model.rho : if not rho, nn.ParamterDict() if use_cell_types, else
        nn.Parameter. If rho is a
        float it is fixed throughout training

        model.adj_matrix : adjacency matrix with diagonal removed. dict containing
        torch.Tensors

        model.adj_matrix_1m : 1 - adjacency matrix with diagonal removed. dict
        containing torch.Tensors

        model.weights : contains edge weights. format matches adj_matrix

        model.cell_types : np.ndarray containing array of unique cell types

        model.cell_type_counts : dict {key = cell type, values = number of cells}

        model.theta : nn.ParameterDict() or nn.Parameter() containing the factor
        weights

        model.alpha : nn.ParameterDict() or nn.Parameter() containing the cell
        loadings

        model.eta : nn.ParameterDict() or nn.Parameter() containing the
        interaction matrix between
        factors

        model.gene_scaling : nn.ParameterDict() or nn.Parameter() containing
        the gene scale factors

        model.selection :  nn.ParameterDict() or nn.Parameter() containing
        the attention weights.
        Only initialized when L[cell_type] > K[cell_type] for some cell type
        or when L > K and
        use_cell_types == False

        model.kgeql_flag : dict or bool. dictionary of boolean values indicating
        whether K >= L.
        When use_cell_types == False, it is a boolean value

    Methods
        ----------

        model.loss(self, X, labels) : called by fit if use_cell_types = True.
        Evalutes the loss of
        the model

        model.loss_no_cell_types(self,X) : called by fit if use_cell_types =
        False. Evalutes the loss
        of the model

        model.initialize(self, gene_sets,val) : initialize the model based
        on given dictionary of gene
        sets. val is a float that determines the strength of the initialization.

        model.initialize_no_celltypes(self, gs_list, val) : initialize the
        model based on given list
        of gene sets. val is a float that determines the strength of the
        initialization.


    To do:
        __________

        > Alternative initialization functions

        > comment SPECTRA-EM code

        > test lower bound constraint [see pyspade_global.py implementation]

        > Overlap threshold test statistic


    """

    def __init__(  # noqa
        self,
        X,
        labels,
        adj_matrix,
        L,
        pert_idx,
        pert_labels=None,
        weights=None,
        lam=0.01,
        psi=0.01,
        delta=0.001,
        kappa=None,
        rho=0.001,
        use_cell_types=True,
        device=torch.device("cuda:0"),
    ):
        super(SPECTRA, self).__init__()

        # hyperparameters
        self.delta = delta
        self.lam = lam
        self.psi = psi
        self.L = L
        # for memory efficiency we don't store X in the object attributes, but require X
        # dimensions to
        # be known at initialization
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.use_cell_types = use_cell_types
        self.device = device

        self.pert_idx = pert_idx
        self.pert_labels = pert_labels
        # add one dim for ctrl one-hot
        self.n_p = len(pert_idx)

        if not use_cell_types:
            # check that L is an int
            assert isinstance(self.L, int)

            # trust the user to input a np.ndarray for adj_matrix
            self.adj_matrix = torch.Tensor(adj_matrix).to(self.device) - torch.Tensor(
                np.diag(np.diag(adj_matrix))
            ).to(self.device)
            adj_matrix_1m = 1.0 - adj_matrix
            self.adj_matrix_1m = torch.Tensor(
                adj_matrix_1m - np.diag(np.diag(adj_matrix_1m))
            ).to(self.device)
            if weights is not None:
                self.weights = torch.Tensor(weights).to(self.device) - torch.Tensor(
                    np.diag(np.diag(adj_matrix))
                ).to(self.device)
            else:
                self.weights = self.adj_matrix

            self.theta = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.L]))
            self.alpha = nn.Parameter(Normal(0.0, 1.0).sample([self.n_p, self.L]))
            self.eta = nn.Parameter(Normal(0.0, 1.0).sample([self.L, self.L]))
            self.gene_scaling = nn.Parameter(torch.zeros(self.p))

            if kappa is None:
                self.kappa = nn.Parameter(Normal(0.0, 1.0).sample())
            else:
                self.kappa = torch.tensor(np.log(kappa / (1 - kappa))).to(self.device)
            if rho is None:
                self.rho = nn.Parameter(Normal(0.0, 1.0).sample())
            else:
                self.rho = torch.tensor(np.log(rho / (1 - rho))).to(self.device)

        if use_cell_types:
            # convert adjacency matrices to pytorch tensors to make optimization easier later
            self.adj_matrix = {
                cell_type: (
                    torch.Tensor(adj_matrix[cell_type]).to(self.device)
                    - torch.Tensor(np.diag(np.diag(adj_matrix[cell_type]))).to(
                        self.device
                    )
                    if len(adj_matrix[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix.keys()
            }
            # for convenience store 1 - adjacency matrix elements [except on diagonal,
            # where we store 0]
            adj_matrix_1m = {
                cell_type: (
                    1.0 - adj_matrix[cell_type]
                    if len(adj_matrix[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix.keys()
            }  # one adj_matrix per cell type
            self.adj_matrix_1m = {
                cell_type: (
                    torch.Tensor(
                        adj_matrix_1m[cell_type]
                        - np.diag(np.diag(adj_matrix_1m[cell_type]))
                    ).to(self.device)
                    if len(adj_matrix_1m[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix_1m.keys()
            }  # one adj_matrix per cell type

            # if weights are provided, convert these to tensors, else set weights = to
            # adjacency matrices
            if weights:
                self.weights = {
                    cell_type: (
                        torch.Tensor(weights[cell_type]).to(self.device)
                        - torch.Tensor(np.diag(np.diag(weights[cell_type]))).to(
                            self.device
                        )
                        if len(weights[cell_type]) > 0
                        else []
                    )
                    for cell_type in weights.keys()
                }
            else:
                self.weights = self.adj_matrix

            self.cell_types = np.unique(
                labels
            )  # cell types are the unique labels, again require knowledge of labels at
            # initialization
            # but do not store them

            # store a dictionary containing the counts of each cell type
            self.cell_type_counts = {}
            for cell_type in self.cell_types:
                n_c = sum(labels == cell_type)
                self.cell_type_counts[cell_type] = n_c

            # initialize parameters randomly, we use torch's ParameterDict() for storage
            # for intuitive
            # accessing cell type specific parameters
            self.theta = nn.ParameterDict()
            self.alpha = nn.ParameterDict()
            self.eta = nn.ParameterDict()
            self.gene_scaling = nn.ParameterDict()

            if kappa is None:
                self.kappa = nn.ParameterDict()
            if rho is None:
                self.rho = nn.ParameterDict()
            # initialize global params
            self.theta["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.p, self.L["global"]])
            )
            self.eta["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.L["global"], self.L["global"]])
            )
            self.gene_scaling["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.p])
            )
            if kappa is None:
                self.kappa["global"] = nn.Parameter(Normal(0.0, 1.0).sample())
            if rho is None:
                self.rho["global"] = nn.Parameter(Normal(0.0, 1.0).sample())

            # initialize all cell type specific params
            for cell_type in self.cell_types:
                self.theta[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.p, self.L[cell_type]])
                )
                self.eta[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.L[cell_type], self.L[cell_type]])
                )
                n_c = sum(labels == cell_type)
                self.alpha[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample(
                        [self.n_p, self.L["global"] + self.L[cell_type]]
                    )
                )
                self.gene_scaling[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.p])
                )

                if kappa is None:
                    self.kappa[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())

                if rho is None:
                    self.rho[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())

            # if kappa and rho are provided, hold these fixed during training, else fit as
            # free parameters
            # to unify the cases, we put this in the same format
            if kappa is not None:
                self.kappa = {}
                self.kappa["global"] = torch.tensor(np.log(kappa / (1 - kappa)))
                for cell_type in self.cell_types:
                    self.kappa[cell_type] = torch.tensor(np.log(kappa / (1 - kappa)))
                # self.kappa = nn.ParameterDict(self.kappa)
            if rho is not None:
                self.rho = {}
                self.rho["global"] = torch.tensor(np.log(rho / (1 - rho)))
                for cell_type in self.cell_types:
                    self.rho[cell_type] = torch.tensor(np.log(rho / (1 - rho)))
                # self.rho = nn.ParameterDict(self.rho)

    def loss(self, X, labels, loss_weights, D=None, forward=False):
        assert self.use_cell_types  # if this is False, fail because model has not been initialized to use cell types
        X = X.to(self.device)
        D = D.to(self.device)
        loss_weights = loss_weights.to(self.device)

        # initialize loss and fetch global parameters
        loss = 0.0
        theta_global = torch.softmax(self.theta["global"], dim=1)
        eta_global = (self.eta["global"]).exp() / (1.0 + (self.eta["global"]).exp())
        eta_global = 0.5 * (eta_global + eta_global.T)
        gene_scaling_global = self.gene_scaling["global"].exp() / (
            1.0 + self.gene_scaling["global"].exp()
        )
        kappa_global = self.kappa["global"].exp() / (1 + self.kappa["global"].exp())
        rho_global = self.rho["global"].exp() / (1 + self.rho["global"].exp())

        recon_dict = {}
        term1_dict = {}
        term2_dict = {}
        term3_dict = {}
        term4_dict = {}

        # loop through cell types and evaluate loss at every cell type
        for cell_type in self.cell_types:
            kappa = self.kappa[cell_type].exp() / (1 + self.kappa[cell_type].exp())
            rho = self.rho[cell_type].exp() / (1 + self.rho[cell_type].exp())
            gene_scaling_ct = self.gene_scaling[cell_type].exp() / (
                1.0 + self.gene_scaling[cell_type].exp()
            )
            X_c = X[labels == cell_type]
            loss_weights_c = loss_weights[labels == cell_type]
            adj_matrix = self.adj_matrix[cell_type]
            weights = self.weights[cell_type]
            adj_matrix_1m = self.adj_matrix_1m[cell_type]
            theta_ct = torch.softmax(self.theta[cell_type], dim=1)
            eta_ct = (self.eta[cell_type]).exp() / (1.0 + (self.eta[cell_type]).exp())
            eta_ct = 0.5 * (eta_ct + eta_ct.T)
            theta_global_ = contract(
                "jk,j->jk", theta_global, gene_scaling_global + self.delta
            )
            theta_ct_ = contract("jk,j->jk", theta_ct, gene_scaling_ct + self.delta)
            theta = torch.cat((theta_global_, theta_ct_), 1)
            alpha = torch.exp(self.alpha[cell_type])

            # terms for perturbation
            if D.shape[0] > 0:
                D_c = D[labels == cell_type]

            p_alpha = D_c @ alpha
            recon = contract("ik,jk->ij", p_alpha, theta)
            term1 = (
                -1.0
                * ((torch.xlogy(X_c, recon) - recon) * loss_weights_c[:, None]).sum()
            )

            if len(adj_matrix) > 0:
                mat = contract("il,lj,kj->ik", theta_ct, eta_ct, theta_ct)
                term2 = (
                    -1.0
                    * (
                        torch.xlogy(
                            adj_matrix * weights,
                            (1.0 - rho) * (1.0 - kappa) * mat + (1.0 - rho) * kappa,
                        )
                    ).sum()
                )
                term3 = (
                    -1.0
                    * (
                        torch.xlogy(
                            adj_matrix_1m,
                            (1.0 - kappa) * (1.0 - rho) * (1.0 - mat) + rho,
                        )
                    ).sum()
                )
                # perturbation autocorrelation term
                # include only perturbations that are in the graph
                if D.shape[0] > 0:
                    ctrl_indices = [i for i, x in enumerate(self.pert_idx) if x == -1]
                    pert_subset_idx = [
                        i for i, x in enumerate(self.pert_idx) if x != -1
                    ]
                    alpha_subset_idx = [
                        i for i in range(alpha.shape[0]) if i not in ctrl_indices
                    ]
                    term4 = (
                        -1
                        * Spectra_util.geary_autocorrelation_multivariate(
                            mat[pert_subset_idx][:, pert_subset_idx],
                            alpha[alpha_subset_idx],
                        )["stat"]
                    )
                else:
                    term4 = 0.0
            else:
                term2 = 0.0
                term3 = 0.0
                term4 = 0.0
            recon_dict[cell_type] = recon.clone()
            term1_dict[cell_type] = term1
            term2_dict[cell_type] = term2
            term3_dict[cell_type] = term3
            term4_dict[cell_type] = term4
            loss = (
                loss
                + self.lam * term1
                + (self.cell_type_counts[cell_type] / float(self.n)) * (term2 + term3)
            )

        # compute loss associated with global graph
        adj_matrix = self.adj_matrix["global"]
        adj_matrix_1m = self.adj_matrix_1m["global"]
        weights = self.weights["global"]

        if len(adj_matrix) > 0:
            mat = contract(
                "il,lj,kj->ik",
                theta_global,
                eta_global,
                theta_global,
            )
            term2 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix * weights,
                        (1.0 - rho_global) * (1.0 - kappa_global) * mat
                        + (1.0 - rho_global) * kappa_global,
                    )
                ).sum()
            )
            term3 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix_1m,
                        (1.0 - kappa_global) * (1.0 - rho_global) * (1.0 - mat)
                        + rho_global,
                    )
                ).sum()
            )
            term2_dict["global"] = term2
            term3_dict["global"] = term3

            loss = loss + term2 + term3

        if forward:
            return term1_dict, recon_dict
        # returns loss, recon, graph likelihoods, and autocorr
        return loss, term1_dict, term2_dict, term3_dict, term4_dict

    def loss_no_cell_types(self, X, loss_weights, D=None, forward=False):
        assert not self.use_cell_types  # if this is True, just fail
        X = X.to(self.device)
        D = D.to(self.device)
        loss_weights = loss_weights.to(self.device)

        theta = torch.softmax(self.theta, dim=1)
        eta = self.eta.exp() / (1.0 + (self.eta).exp())
        eta = 0.5 * (eta + eta.T)
        gene_scaling = self.gene_scaling.exp() / (1.0 + self.gene_scaling.exp())
        kappa = self.kappa.exp() / (1 + self.kappa.exp())
        rho = (self.rho.exp() / (1 + self.rho.exp())).to(self.device)
        alpha = torch.exp(self.alpha)

        adj_matrix = self.adj_matrix.to(self.device)
        weights = self.weights.to(self.device)
        adj_matrix_1m = self.adj_matrix_1m.to(self.device)
        theta_ = contract("jk,j->jk", theta, gene_scaling + self.delta)

        p_alpha = D @ alpha
        recon = contract("ik,jk->ij", p_alpha, theta_)
        term1 = -1.0 * ((torch.xlogy(X, recon) - recon) * loss_weights[:, None]).sum()

        if len(adj_matrix) > 0:
            mat = contract("il,lj,kj->ik", theta, eta, theta)
            term2 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix * weights,
                        (1.0 - rho) * (1.0 - kappa) * mat + (1.0 - rho) * kappa,
                    )
                ).sum()
            )
            term3 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix_1m, (1.0 - kappa) * (1.0 - rho) * (1.0 - mat) + rho
                    )
                ).sum()
            )
            if D.shape[0] > 0:
                # remove ctrl or pert not present
                rm_indices = []
                for i, x in enumerate(self.pert_idx):
                    if x == -1:
                        rm_indices.append(i)
                alpha_subset_idx = [
                    i
                    for i in range(alpha.shape[0])
                    if i not in rm_indices + [self.n_p - 1]
                ]
                pert_adj_idx = [i for i in self.pert_idx if i != -1]
                term4 = (
                    -1
                    * Spectra_util.geary_autocorrelation_multivariate(
                        mat[pert_adj_idx][:, pert_adj_idx], alpha[alpha_subset_idx]
                    )["stat"]
                )

        else:
            term2 = 0.0
            term3 = 0.0
            term4 = 0.0

        loss = self.lam * term1 + term2 + term3
        if forward:
            return term1, recon
        return loss, term1, term2, term3, term4

    def initialize(self, gene_sets, val):
        """
        form of gene_sets:

        cell_type (inc. global) : set of sets of idxs
        """

        for ct in self.cell_types:
            assert self.L[ct] >= len(gene_sets[ct])
            count = 0
            if self.L[ct] > 0:
                if len(self.adj_matrix[ct]) > 0:
                    for gene_set in gene_sets[ct]:
                        self.theta[ct].data[:, count][gene_set] = val
                        count = count + 1
                    for i in range(self.L[ct]):
                        self.eta[ct].data[i, -1] = -val
                        self.eta[ct].data[-1, i] = -val
                    self.theta[ct].data[:, -1][self.adj_matrix[ct].sum(axis=1) == 0] = (
                        val
                    )
                    self.theta[ct].data[:, -1][
                        self.adj_matrix[ct].sum(axis=1) != 0
                    ] = -val

        assert self.L["global"] >= len(gene_sets["global"])
        count = 0
        for gene_set in gene_sets["global"]:
            self.theta["global"].data[:, count][gene_set] = val
            count = count + 1
        for i in range(self.L["global"]):
            self.eta["global"].data[i, -1] = -val
            self.eta["global"].data[-1, i] = -val
        self.theta["global"].data[:, -1][self.adj_matrix["global"].sum(axis=1) == 0] = (
            val
        )
        self.theta["global"].data[:, -1][
            self.adj_matrix["global"].sum(axis=1) != 0
        ] = -val

    def initialize_no_geneset(self, gene_sets, val):
        for ct in self.cell_types:
            torch.nn.init.xavier_uniform_(self.theta[ct])
            torch.nn.init.xavier_uniform_(self.eta[ct])
        torch.nn.init.xavier_uniform_(self.theta["global"])
        torch.nn.init.xavier_uniform_(self.eta["global"])

    def initialize_no_celltypes(self, gs_list, val):
        assert self.L >= len(gs_list)
        count = 0
        for gene_set in gs_list:
            self.theta.data[:, count][gene_set] = val
            count = count + 1
        for i in range(self.L):
            self.eta.data[i, -1] = -val
            self.eta.data[-1, i] = -val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) == 0] = val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) != 0] = -val

    def initialize_no_celltypes_no_geneset(self, gs_list, val):
        torch.nn.init.xavier_uniform_(self.theta)
        torch.nn.init.xavier_uniform_(self.eta)


class SPECTRA_Model:
    """

    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used
            for initialization
            of self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types
            == False, then
            should be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key
            , indicating the
            number of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > L contains the number of factors that describe the graph.
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...

            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then
        ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is
            non-zero when adj_matrix[i,j]
            = 0 this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence
            of the graph vs expression
            loss functions. This term multiplies the expression loss, so smaller
            values of lambda upweight
            the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene
            scaling factors. If delta
            is small then the maximum ratio between gene scaling factors is
            larger and lowly expressed
            genes can be put on the same scale as highly expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa
            is a float, kappa is fixed
            to the given float value. If kappa == None, then kappa is a parameter
            that is estimated from
            the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho
            is a float, rho is fixed
            to the given float value. If rho == None, then rho is a parameter
            that is estimated from the
            data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell
            type labels are used to
            fit the model. If False, then parameters are initialized as
            nn.Parameter rather than as
            nn.ParameterDict with cell type keys that index nn.Parameter values
        determinant_penalty : float
            determinant penalty affects the selection parameters that are fit
            when L[cell_type] >
            K[cell_type]. A determinantally regularized selection parameter
            is fit with determinant
            penalty that encourages sparsity and diversity.
    Attributes
        ----------
        model.delta : delta parameter of the model

        model.lam : lambda parameter of the model

        model.determinant_penalty : determinant penalty of the model

        model.L : L parameter, either int, dict or OrderedDict()

        model.p : number of genes

        model.n : number of cells

        model.use_cell_types : if True then cell types are considered, else cell
        types ignored. Affects
        the dimensions of the initialized parameters.

        model.kappa : if not kappa, nn.ParameterDict() if use_cell_types, else
        nn.Parameter(). If kappa
        is a float, it is fixed throughout training

        model.rho : if not rho, nn.ParamterDict() if use_cell_types, else
        nn.Parameter. If rho is a float
        it is fixed throughout training

        model.adj_matrix : adjacency matrix with diagonal removed. dict
        containing torch.Tensors

        model.adj_matrix_1m : 1 - adjacency matrix with diagonal removed.
        dict containing torch.Tensors

        model.weights : contains edge weights. format matches adj_matrix

        model.cell_types : np.ndarray containing array of unique cell types

        model.cell_type_counts : dict {key = cell type, values = number of
        cells}

        model.factors : nn.ParameterDict() or nn.Parameter() containing the
        factor weights

        model.cell_scores : nn.ParameterDict() or nn.Parameter() containing
        the cell loadings

        model.eta : nn.ParameterDict() or nn.Parameter() containing the
        interaction matrix between
        factors

        model.gene_scaling : nn.ParameterDict() or nn.Parameter() containing
        the gene scale factors

        model.selection :  nn.ParameterDict() or nn.Parameter() containing
        the attention weights.
        Only initialized when L[cell_type] > K[cell_type] for some cell
        type or when L > K and
        use_cell_types == False



    Methods
        ----------

        model.train(self, X, labels, lr_schedule,num_epochs, verbose) :
        model.save()
        model.load()
        model.initialize
        model.return_selection()
        model.return_eta_diag()
        model.return_cell_scores()
        model.return_factors()
        model.return_eta()
        model.return_rho()
        model.return_kappa()
        model.return_gene_scalings()
        model.return_graph(ct = "global") :
        model.matching(markers, gene_names_dict, threshold = 0.4):

    """

    def __init__(
        self,
        X,
        labels,
        L,
        pert_idx,
        pert_labels,
        vocab=None,
        gs_dict=None,
        use_weights=True,
        adj_matrix=None,
        weights=None,
        lam=0.01,
        psi=0.01,
        delta=0.001,
        kappa=None,
        rho=0.001,
        use_cell_types=True,
    ):
        self.L = L
        self.lam = lam
        self.delta = delta
        self.kappa = kappa
        self.rho = rho
        self.use_cell_types = use_cell_types

        # if gs_dict is provided instead of adj_matrix, convert to adj_matrix, overrides
        # adj_matrix and weights
        if gs_dict is not None:
            gene2id = dict((v, idx) for idx, v in enumerate(vocab))

            if use_cell_types:
                adj_matrix, weights = Spectra_util.process_gene_sets(
                    gs_dict=gs_dict, gene2id=gene2id, weighted=use_weights
                )
            else:
                adj_matrix, weights = Spectra_util.process_gene_sets_no_celltypes(
                    gs_dict=gs_dict, gene2id=gene2id, weighted=use_weights
                )

        self.internal_model = SPECTRA(
            X=X,
            labels=labels,
            pert_idx=pert_idx,
            pert_labels=pert_labels,
            adj_matrix=adj_matrix,
            L=L,
            weights=weights,
            lam=lam,
            psi=psi,
            delta=delta,
            kappa=kappa,
            rho=rho,
            use_cell_types=use_cell_types,
        )

        self.cell_scores = None
        self.factors = None
        self.B_diag = None
        self.eta_matrices = None
        self.gene_scalings = None
        self.rho = None
        self.kappa = None

    def train(
        self,
        X,
        D,
        loss_weights,
        X_val,
        D_val,
        loss_weights_val,
        labels=None,
        labels_val=None,
        lr_schedule=[1.0, 0.5, 0.1, 0.01, 0.001, 0.0001],
        num_epochs=10000,
        verbose=True,
    ):
        opt = torch.optim.AdamW(
            self.internal_model.parameters(), lr=lr_schedule[0], weight_decay=0.001
        )
        counter = 0
        last = np.inf
        train_losses = []
        val_losses = []

        # batch if too large
        X = torch.from_numpy(X)
        D = torch.from_numpy(D)
        X_val = torch.from_numpy(X_val)
        D_val = torch.from_numpy(D_val)
        loss_weights = torch.from_numpy(loss_weights)
        loss_weights_val = torch.from_numpy(loss_weights_val)
        if X.shape[0] > 2e5:
            train_dataset = TensorDataset(X, D, loss_weights)
            val_dataset = TensorDataset(X_val, D_val, loss_weights_val)
            train_dataloader = DataLoader(
                train_dataset, batch_size=int(5e4), num_workers=4, shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=int(5e4), num_workers=4, shuffle=False
            )
        else:
            train_dataloader = None
            val_dataloader = None

        # train loop
        for i in tqdm(range(num_epochs)):
            train_epoch_loss = 0
            val_epoch_loss = 0
            # batch if data too large
            if train_dataloader:
                # train
                for batch in train_dataloader:
                    X_batch = batch[0]
                    D_batch = batch[1]
                    loss_weights_batch = batch[2]
                    opt.zero_grad()
                    if self.internal_model.use_cell_types:
                        assert len(labels) == X.shape[0]
                        loss, term1_dict, term2_dict, term3_dict, term4_dict = (
                            self.internal_model.loss(
                                X=X_batch,
                                D=D_batch,
                                loss_weights=loss_weights_batch,
                                labels=labels,
                            )
                        )
                    elif not self.internal_model.use_cell_types:
                        loss, term1, term2, term3, term4 = (
                            self.internal_model.loss_no_cell_types(
                                X=X_batch, D=D_batch, loss_weights=loss_weights_batch
                            )
                        )

                    loss.backward()
                    opt.step()
                    train_epoch_loss += loss.item()
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                # val
                with torch.no_grad():
                    for batch in val_dataloader:
                        X_batch = batch[0]
                        D_batch = batch[1]
                        loss_weights_batch = batch[2]
                        if self.internal_model.use_cell_types:
                            assert len(labels) == X.shape[0]
                            loss, term1_dict, term2_dict, term3_dict, term4_dict = (
                                self.internal_model.loss(
                                    X=X_batch,
                                    D=D_batch,
                                    loss_weights=loss_weights_batch,
                                    labels=labels,
                                )
                            )
                        elif not self.internal_model.use_cell_types:
                            loss, term1, term2, term3, term4 = (
                                self.internal_model.loss_no_cell_types(
                                    X=X_batch,
                                    D=D_batch,
                                    loss_weights=loss_weights_batch,
                                )
                            )
                        val_epoch_loss += loss.item()
                val_epoch_loss = val_epoch_loss / len(val_dataloader)
            else:
                # train
                opt.zero_grad()
                if self.internal_model.use_cell_types:
                    assert len(labels) == X.shape[0]
                    loss, term1_dict, term2_dict, term3_dict, term4_dict = (
                        self.internal_model.loss(
                            X=X, D=D, loss_weights=loss_weights, labels=labels
                        )
                    )
                elif not self.internal_model.use_cell_types:
                    loss, term1, term2, term3, term4 = (
                        self.internal_model.loss_no_cell_types(
                            X=X, D=D, loss_weights=loss_weights
                        )
                    )
                loss.backward()
                opt.step()
                train_epoch_loss = loss.item()

                # val
                with torch.no_grad():
                    if self.internal_model.use_cell_types:
                        assert len(labels_val) == X_val.shape[0]
                        loss, term1_dict, term2_dict, term3_dict, term4_dict = (
                            self.internal_model.loss(
                                X=X_val,
                                D=D_val,
                                loss_weights=loss_weights_val,
                                labels=labels_val,
                            )
                        )
                    elif not self.internal_model.use_cell_types:
                        loss, term1, term2, term3, term4 = (
                            self.internal_model.loss_no_cell_types(
                                X=X_val, D=D_val, loss_weights=loss_weights_val
                            )
                        )
                val_epoch_loss = loss.item()

            # lr adjustment
            if train_epoch_loss >= last:
                counter += 1
                if int(counter / 10) >= len(lr_schedule):
                    print("EARLY STOPPING")
                    break
                if counter % 10 == 0:
                    opt = torch.optim.AdamW(
                        self.internal_model.parameters(),
                        lr=lr_schedule[int(counter / 10)],
                    )
                    if verbose:
                        print("UPDATING LR TO " + str(lr_schedule[int(counter / 10)]))

            last = train_epoch_loss
            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

        # add all model parameters as attributes

        if self.use_cell_types:
            self.__store_parameters(labels)
        else:
            self.__store_parameters_no_celltypes()
        return train_losses, val_losses

    def save(self, fp):
        torch.save(self.internal_model.state_dict(), fp)

    def load(self, fp, labels=None):
        self.internal_model.load_state_dict(torch.load(fp))
        if self.use_cell_types:
            assert labels is not None
            self.__store_parameters(labels)
        else:
            self.__store_parameters_no_celltypes()

    def __store_parameters(self, labels):
        """
        Replaces __cell_scores() and __compute factors() and __compute_theta()
        store parameters after fitting the model:
            cell scores
            factors
            eta
            scalings
            gene scalings
            kappa
            rho
        """

        model = self.internal_model

        # compute the loading matrix

        k = sum(list(model.L.values()))
        out = np.zeros((model.n_p, k))

        global_idx = model.L["global"]

        tot = global_idx
        f = ["global"] * model.L["global"]
        for i, cell_type in enumerate(model.cell_types):
            alpha = torch.exp(model.alpha[cell_type]).detach().cpu().numpy()
            out[:, :global_idx] = alpha[:, :global_idx]
            out[:, tot : tot + model.L[cell_type]] = alpha[:, global_idx:]

            tot += model.L[cell_type]

            f = f + [cell_type] * model.L[cell_type]

        out2 = np.zeros((k, model.p))

        theta_ct = torch.softmax(model.theta["global"], dim=1)
        theta = theta_ct.detach().cpu().numpy().T
        tot = 0
        out2[0 : theta.shape[0], :] = theta
        tot += theta.shape[0]

        for cell_type in model.cell_types:
            theta_ct = torch.softmax(model.theta[cell_type], dim=1)
            theta = theta_ct.detach().cpu().numpy().T
            out2[tot : tot + theta.shape[0], :] = theta
            tot += theta.shape[0]

        factors = out2
        lst = []
        for i in range(len(f)):
            ct = f[i]
            scaled = (
                factors[i, :]
                * (
                    model.gene_scaling[ct].exp().detach()
                    / (1.0 + model.gene_scaling[ct].exp().detach())
                    + model.delta
                )
                .cpu()
                .numpy()
            )

            lst.append(scaled)
        scaled = np.array(lst)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)
        cell_scores = out * scaled.mean(axis=1).reshape(1, -1)
        self.cell_scores = cell_scores
        self.factors = new_factors
        self.B_diag = self.__B_diag()
        self.eta_matrices = self.__eta_matrices()
        self.gene_scalings = {
            ct: model.gene_scaling[ct].exp().detach().cpu().numpy()
            / (1.0 + model.gene_scaling[ct].exp().cpu().detach().numpy())
            for ct in model.gene_scaling.keys()
        }
        self.rho = {
            ct: model.rho[ct].exp().detach().cpu().numpy()
            / (1.0 + model.rho[ct].exp().detach().cpu().numpy())
            for ct in model.rho.keys()
        }
        self.kappa = {
            ct: model.kappa[ct].exp().detach().cpu().numpy()
            / (1.0 + model.kappa[ct].exp().detach().cpu().numpy())
            for ct in model.kappa.keys()
        }

    def __B_diag(self):
        model = self.internal_model
        k = sum(list(model.L.values()))
        out = np.zeros(k)

        Bg = model.eta["global"].exp() / (1.0 + model.eta["global"].exp())
        Bg = 0.5 * (Bg + Bg.T)
        B = torch.diag(Bg).detach().cpu().numpy()
        tot = 0
        out[0 : B.shape[0]] = B
        tot += B.shape[0]

        for cell_type in model.cell_types:
            Bg = model.eta[cell_type].exp() / (1.0 + model.eta[cell_type].exp())
            Bg = 0.5 * (Bg + Bg.T)
            B = torch.diag(Bg).detach().cpu().numpy()
            out[tot : tot + B.shape[0]] = B

            tot += B.shape[0]

        return out

    def __eta_matrices(self):
        model = self.internal_model
        eta = OrderedDict()
        Bg = model.eta["global"].exp() / (1.0 + model.eta["global"].exp())
        Bg = 0.5 * (Bg + Bg.T)
        eta["global"] = Bg.detach().cpu().numpy()

        for cell_type in model.cell_types:
            Bg = model.eta[cell_type].exp() / (1.0 + model.eta[cell_type].exp())
            Bg = 0.5 * (Bg + Bg.T)
            eta[cell_type] = Bg.detach().cpu().numpy()
        return eta

    def __store_parameters_no_celltypes(self):
        """
        store parameters after fitting the model:
            cell scores
            factors
            eta
            scalings
            gene scalings
            kappa
            rho
        """
        model = self.internal_model
        theta_ct = torch.softmax(model.theta, dim=1)
        theta = theta_ct.detach().cpu().numpy().T
        alpha = torch.exp(model.alpha).detach().cpu().numpy()
        out = alpha
        factors = theta

        scaled = factors * (
            model.gene_scaling.exp().detach().cpu()
            / (1.0 + model.gene_scaling.exp().detach().cpu())
            + model.delta
        ).numpy().reshape(1, -1)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)

        self.factors = new_factors
        self.cell_scores = out * scaled.mean(axis=1).reshape(1, -1)
        Bg = model.eta.exp() / (1.0 + model.eta.exp())
        Bg = 0.5 * (Bg + Bg.T)
        self.B_diag = torch.diag(Bg).detach().cpu().numpy()
        self.eta_matrices = Bg.detach().cpu().numpy()
        self.gene_scalings = (
            model.gene_scaling.exp().detach().cpu()
            / (1.0 + model.gene_scaling.exp().detach().cpu())
        ).numpy()
        self.rho = (
            (model.rho.exp().detach() / (1.0 + model.rho.exp().detach())).cpu().numpy()
        )
        self.kappa = (
            (model.kappa.exp().detach() / (1.0 + model.kappa.exp().detach()))
            .cpu()
            .numpy()
        )

    def initialize(self, annotations, word2id, W, init_scores, val=25):
        """
        self.use_cell_types must be True
        create form of gene_sets:

        cell_type (inc. global) : set of sets of idxs

        filter based on L_ct
        """
        if self.use_cell_types:
            if annotations:
                if init_scores is None:
                    init_scores = compute_init_scores(
                        annotations, word2id, torch.Tensor(W)
                    )  # noqa
                gs_dict = OrderedDict()
                for ct in annotations.keys():
                    mval = max(self.L[ct] - 1, 0)
                    sorted_init_scores = sorted(
                        init_scores[ct].items(), key=lambda x: x[1]
                    )
                    sorted_init_scores = sorted_init_scores[-1 * mval :]
                    names = set([k[0] for k in sorted_init_scores])
                    lst_ct = []
                    for key in annotations[ct].keys():
                        if key in names:
                            words = annotations[ct][key]
                            idxs = []
                            for word in words:
                                if word in word2id:
                                    idxs.append(word2id[word])
                            lst_ct.append(idxs)
                    gs_dict[ct] = lst_ct
                self.internal_model.initialize(gene_sets=gs_dict, val=val)
            else:
                self.internal_model.initialize_no_geneset(gene_sets=[], val=val)
        else:
            if annotations:
                if init_scores is None:
                    init_scores = compute_init_scores_noct(
                        annotations, word2id, torch.Tensor(W)
                    )  # noqa
                lst = []
                mval = max(self.L - 1, 0)
                sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
                sorted_init_scores = sorted_init_scores[-1 * mval :]
                names = set([k[0] for k in sorted_init_scores])
                for key in annotations.keys():
                    if key in names:
                        words = annotations[key]
                        idxs = []
                        for word in words:
                            if word in word2id:
                                idxs.append(word2id[word])
                        lst.append(idxs)
                self.internal_model.initialize_no_celltypes(gs_list=lst, val=val)
            else:
                self.internal_model.initialize_no_celltypes_no_geneset(
                    gs_list=[], val=val
                )

    def return_eta_diag(self):
        return self.B_diag

    def return_cell_scores(self):
        return self.cell_scores

    def return_factors(self):
        return self.factors

    def return_eta(self):
        return self.eta_matrices

    def return_rho(self):
        return self.rho

    def return_kappa(self):
        return self.kappa

    def return_gene_scalings(self):
        return self.gene_scalings

    def return_graph(self, ct="global"):
        model = self.internal_model
        if self.use_cell_types:
            eta = (model.eta[ct]).exp() / (1.0 + (model.eta[ct]).exp())
            eta = 0.5 * (eta + eta.T)
            theta = torch.softmax(model.theta[ct].data, dim=1)
            mat = contract("il,lj,kj->ik", theta, eta, theta).detach().cpu().numpy()
        else:
            eta = model.eta.exp() / (1.0 + model.eta.exp())
            eta = 0.5 * (eta + eta.T)
            theta = torch.softmax(model.theta, dim=1)
            mat = contract("il,lj,kj->ik", theta, eta, theta).detach().cpu().numpy()

        return mat

    def matching(self, markers, gene_names_dict, threshold=0.4):
        """
        best match based on overlap coefficient
        """
        markers = pd.DataFrame(markers)
        if self.use_cell_types:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0
                best = ""
                for key in gene_names_dict.keys():
                    for gs in gene_names_dict[key].keys():
                        t = gene_names_dict[key][gs]

                        jacc = Spectra_util.overlap_coefficient(
                            list(markers.iloc[i, :]), t
                        )
                        if jacc > max_jacc:
                            max_jacc = jacc
                            best = gs
                matches.append(best)
                jaccards.append(max_jacc)

        else:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0
                best = ""
                for key in gene_names_dict.keys():
                    t = gene_names_dict[key]

                    jacc = Spectra_util.overlap_coefficient(list(markers.iloc[i, :]), t)
                    if jacc > max_jacc:
                        max_jacc = jacc
                        best = key
                matches.append(best)
                jaccards.append(max_jacc)
        output = []
        for j in range(markers.shape[0]):
            if jaccards[j] > threshold:
                output.append(matches[j])
            else:
                output.append("0")
        return np.array(output)


def return_markers(factor_matrix, id2word, n_top_vals=100):
    idx_matrix = np.argsort(factor_matrix, axis=1)[:, ::-1][:, :n_top_vals]
    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i, j] = id2word[idx_matrix[i, j]]
    return df.values


def vectorize_perts(adata, key, control_key):
    """
    Vectorizing perturbation labels

    Returns: matrix of perturbation labels as vectors, labels for each column
    """
    # create guide one hots (for encoding combos as superpositions)
    perts = set()
    for t in adata.obs[key]:
        if t not in control_key:
            guides = t.split("+")
            guide1 = guides[0]
            guide2 = None
            if len(guides) == 2:
                guide2 = guides[1]
            perts.add(guide1)
            if guide2:
                perts.add(guide2)
    for p in perts:
        guides_p = []
        for t in adata.obs[key]:
            if t == control_key:
                guides_p.append(0)
            else:
                if p in t:
                    guides_p.append(1)
                else:
                    guides_p.append(0)
        adata.obs[f"guide_{p}"] = np.array(guides_p)

    guide_one_hot_cols = get_guide_one_hot_cols(adata.obs)
    adata.obs["num_guides"] = adata.obs[guide_one_hot_cols].sum(1)
    # combinations encoded as application of two individual guides
    D = adata.obs[guide_one_hot_cols].to_numpy().astype(np.float32)

    return D, guide_one_hot_cols


def vectorize_perts_combinations(adata, key, control_key):
    """
    Vectorizing perturbation labels, with combinations considered a unique perturbation

    Returns: matrix of perturbation labels as vectors, labels for each column
    """
    # encode combinations as unique
    D_df = pd.get_dummies(adata.obs[key])
    D_df = D_df.drop(columns=control_key)
    # encode non-targeting as no perturbation for consistency with other encoding
    d_var_info = np.array(D_df.T[[]].index)

    # get singletons-only binarization
    for index, row in D_df.iterrows():
        pert_idx = np.where(row)[0]
        if len(pert_idx) > 0:
            pert_label = d_var_info[pert_idx][0].split("+")
            single_idx = [list(d_var_info).index(i) for i in pert_label]
            for i in single_idx:
                row.iloc[i] = 1

    return D_df.to_numpy().astype(np.float32), list(d_var_info)


def get_guide_one_hot_cols(obs: pd.DataFrame):
    guide_one_hot_cols = [
        col
        for col in obs.columns
        if "guide_" in col and col not in ("guide_identity", "guide_ids")
    ]
    return guide_one_hot_cols
