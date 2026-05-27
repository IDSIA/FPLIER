"""Manages data loading and preprocessing for the Flower ClientApp."""

import random
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from phe import paillier
from typing import Union, Literal, Tuple
import os
import warnings
import sys
import traceback

import pyreadr
from datasets import Dataset

from logging import DEBUG, INFO, CRITICAL
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import log


partitioner = None


def get_dummy_start():
    return np.ones((1, 1))


def get_random_vector(N=5, shape=None):
    if shape is not None:
        return np.random.rand(shape[0], shape[1])
    else:
        return np.random.rand(N, N)


def load_data_simulation(
    client_id: int,
):

    if client_id == 0:
        dataset = pyreadr.read_r(f"data/client_{client_id}/gene_matrix_test.RDS")[
            None
        ].iloc[:, :301]
    elif client_id == 1:
        dataset = pyreadr.read_r(f"data/client_{client_id}/gene_matrix_test.RDS")[
            None
        ].iloc[:, 301:601]
    elif client_id == 2:
        dataset = pyreadr.read_r(f"data/client_{client_id}/gene_matrix_test.RDS")[
            None
        ].iloc[:, 601:]

    # if client_id == 0:
    #     dataset = pyreadr.read_r(
    #         f"data/client_{client_id}/centralized_dataset_demo.RDS"
    #     )[None].iloc[:, :501]
    # elif client_id == 1:
    #     dataset = pyreadr.read_r(
    #         f"data/client_{client_id}/centralized_dataset_demo.RDS"
    #     )[None].iloc[:, 501:]

    signatures = pyreadr.read_r(f"data/client_{client_id}/signatures_matrix_demo.RDS")[
        None
    ]

    return dataset, signatures


# def load_data(
#     partition_id: int,
#     num_partitions: int,
#     load_Y_or_C: str,
#     seed_value: int,
# ):
#     """
#     Loads a partition of the synthetic gene expression dataset.

#     Parameters:
#         partition_id (int): The ID of the partition to load.
#         num_partitions (int): Total number of partitions.
#         num_individuals (int): Number of individuals in the dataset.
#         num_genes (int): Number of genes in the dataset.
#         seed_value (int): Random seed for reproducibility.

#     Returns:
#         pd.DataFrame: The selected partition of the dataset.
#     """
#     global partitioner

#     load_Y_or_C = load_Y_or_C.lower().strip()
#     if load_Y_or_C not in ["y", "c"]:
#         raise ValueError("load_Y_or_C must be 'Y' or 'C'.")

#     if load_Y_or_C == "y":

#         if partitioner is None:  # Create partitioner only once
#             dataset = pyreadr.read_r("data/gene_matrix_test.RDS")[None]

#             # Convert the dataset to a pandas DataFrame
#             dataset = Dataset.from_pandas(
#                 dataset.T.reset_index()
#             )  # transposing so that the partitioning works on the columns
#             # resetting index so that I can restore the signatures names later

#             partitioner = IidPartitioner(num_partitions)
#             partitioner.dataset = dataset

#         partition = (
#             partitioner.load_partition(partition_id).with_format("pandas").to_pandas()
#         ).T  # transposing back to original shape (genes on the rows and samples on columns (partitioned))

#         partition.columns = partition.iloc[0]  # Set the first column as the header
#         partition = partition[1:]  # Remove the first row

#         # remove the index name
#         partition.columns.names = [""]
#         partition.columns.names = [""]

#     elif load_Y_or_C == "c":
#         partition = pyreadr.read_r("data/signatures_matrix_demo.RDS")[None]

#     return partition


class FplierUtils:
    """
    This is a utility class designed specifically to support the PLIER (Pathway-Level Information ExtractoR) process.
    It consists of a set of static methods that handle various auxiliary tasks required for the efficient functioning
    of the main PLIER process.

    Each method in this class serves a unique purpose.
    """

    @staticmethod
    def common_names(
        genes1: list | np.ndarray, genes2: list | np.ndarray
    ) -> list:  # tested
        """
        The common names of the rows between 2 dataframes

        Parameters
        ----------
        data1 : np.ndarray or list
            the first genes to compare
        data2 : np.ndarray or list
            the second set of genes

        Returns
        -------
        list
            the common names of the rows (index)
        """

        row_names1 = set(genes1)
        row_names2 = set(genes2)

        cm = sorted(list(row_names1.intersection(row_names2)))

        return cm

    def getAUC(
        self, plierRes: dict, data: np.ndarray, prior_mat: np.ndarray
    ) -> dict:  # tested
        """
        Calculate AUC (Area Under the Curve) values for PLIER results.

        Parameters
        ----------
        plierRes : dict
            Current PLIER result containing 'B', 'Z', 'L1', 'L2', and 'U' matrices,
            which are np.ndarrays.

        data : np.ndarray
            The input data.

        prior_mat : np.ndarray
            The prior information matrix.

        Returns
        -------
        dict
            A dictionary containing 'Uauc', 'Upval', and 'summary' matrices.
        """

        from statsmodels.stats.multitest import multipletests

        Y = data
        Z = plierRes["Z"]
        Zcv = np.copy(Z)
        k = Z.shape[1]
        L1 = plierRes["L1"]
        L2 = plierRes["L2"]

        # Cross-validation loop
        for i in range(5):
            ii = np.arange(0, np.floor(data.shape[0] / 5) * 5, 5) + i
            ii = ii[ii <= Z.shape[0]].astype(int)

            all_indices = np.arange(Z.shape[0])  # all indices for Z
            # indices not in ii, similar to R's -ii
            not_ii = np.setdiff1d(all_indices, ii)

            Bcv = (
                np.linalg.inv(Z[not_ii].T @ Z[not_ii] + L2 * np.eye(k))
                @ Z[not_ii].T
                @ Y[not_ii]
            )

            Zcv[ii, :] = Y[ii, :] @ Bcv.T @ np.linalg.inv(Bcv @ Bcv.T + L1 * np.eye(k))

        Uauc = plierRes["U"].copy()
        Uauc[:] = 0
        Up = plierRes["U"].copy()
        Up[:] = 1
        out = np.empty((0, 4))
        ii = np.where(np.sum(plierRes["U"], axis=0) > 0)[0]

        # Calculate AUC values
        for i in ii:
            iipath = np.where(plierRes["U"][:, i] > 0)[0]
            for j in iipath:
                true = prior_mat[:, j]
                value = plierRes["Z"][:, i]
                aucres = self.AUC(true, value)
                out = np.vstack([out, [j, i, aucres["auc"], aucres["pval"]]])
                Uauc[j, i] = aucres["auc"]
                Up[j, i] = aucres["pval"]

        out = pd.DataFrame(out, columns=["pathway", "LV index", "AUC", "p-value"])
        out[["AUC", "p-value"]] = out[["AUC", "p-value"]].astype(float)
        out["FDR"] = multipletests(out["p-value"], method="fdr_bh")[1]  # BH function

        return {"Uauc": Uauc, "Upval": Up, "summary": out}

    def cross_val(
        self, plierRes: dict, prior_mat: np.ndarray, prior_mat_cv: np.ndarray
    ) -> dict:  # tested
        """
        Performs cross-validation on PLIER results using prior information matrix.

        Parameters
        ----------
        plierRes : dict
            PLIER results containing 'U' and 'Z' matrices.

        prior_mat : np.ndarray
            The real prior information matrix.

        prior_mat_cv : np.ndarray
            The zeroed-out prior information matrix used for PLIER computations.

        Returns
        -------
        dict
            A dictionary containing 'Uauc', 'Upval', and 'summary' matrices.
        """

        from statsmodels.stats.multitest import multipletests

        Uauc = plierRes["U"].copy()
        Uauc[:] = 0
        Up = plierRes["U"].copy()
        Up[:] = 1
        out = np.empty((0, 4))
        ii = np.where(np.sum(plierRes["U"], axis=0) > 0)[0]

        for i in ii:
            iipath = np.where(plierRes["U"][:, i] > 0)[0]
            if len(iipath) > 1:
                for j in iipath:
                    iiheldout = np.where(
                        (prior_mat[:, iipath].sum(axis=1) == 0)
                        | ((prior_mat[:, j] > 0) & (prior_mat_cv[:, j] == 0))
                    )[0]
                    true = prior_mat[iiheldout, j]
                    value = plierRes["Z"][iiheldout, i]
                    aucres = self.AUC(true, value)
                    out = np.vstack([out, [j, i, aucres["auc"], aucres["pval"]]])
                    Uauc[j, i] = aucres["auc"]
                    Up[j, i] = aucres["pval"]
            else:
                j = iipath[0]
                iiheldout = np.where(
                    (prior_mat[:, iipath].sum(axis=1) == 0)
                    | ((prior_mat[:, j] > 0) & (prior_mat_cv[:, j] == 0))
                )[0]

                true = prior_mat[iiheldout, j]
                value = plierRes["Z"][iiheldout, i]
                aucres = self.AUC(true, value)
                out = np.vstack([out, [j, i, aucres["auc"], aucres["pval"]]])
                Uauc[j, i] = aucres["auc"]
                Up[j, i] = aucres["pval"]

        out = pd.DataFrame(out, columns=["pathway", "LV index", "AUC", "p-value"])
        out[["AUC", "p-value"]] = out[["AUC", "p-value"]].astype(float)
        out["FDR"] = multipletests(out["p-value"], method="fdr_bh")[1]  # BH function

        return {"Uauc": Uauc, "Upval": Up, "summary": out}

    @staticmethod
    def AUC(true: np.ndarray, values: np.ndarray) -> dict:
        """
        Calculate AUC (Area Under the Curve) using roc_auc_score and Mann-Whitney U test.

        Parameters
        ----------
        labels : np.ndarray
            The binary labels.

        values : np.ndarray
            The corresponding values.

        Returns
        -------
        dict
            A dictionary containing the AUC, p-value, and confidence interval.
        """

        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu

        pos_indices = np.where(true > 0)[0]
        neg_indices = np.where(true <= 0)[0]
        pos_values = values[pos_indices]
        neg_values = values[neg_indices]

        if len(pos_indices) > 0 and len(neg_indices) > 0:
            auc = roc_auc_score(true, values)
            pvalue = mannwhitneyu(pos_values, neg_values, alternative="greater").pvalue
            res = {"auc": auc, "pval": pvalue}
        else:
            res = {"auc": 0.5, "pval": np.nan}
        return res

    def num_pc(
        self,
        data: np.ndarray,
        method: Literal["elbow", "permutation"] = "elbow",
        B=20,
        seed=None,
    ) -> int:  # tested
        """
        Estimates the number of 'significant' principle components for the SVD decomposition.
        This is the minimum k for PLIER.

        Parameters
        ----------
        data : dictionary
            The same data as to be used for PLIER (z-score recommended) or alternatively the result of an SVD calculation.

        method : str, optional
            Either "elbow" (fast) or "permutation" (slower, but less heuristic).

        B : int, optional
            Number of permutations. Only applicable if method is set to "permutation".

        seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        int
            The estimated number of significant principle components.
        """

        from pysmooth import smooth

        if method not in ["elbow", "permutation"]:
            raise ValueError("Method must be either 'elbow' or 'permutation'")

        if seed is not None:
            np.random.seed(seed)

        if not isinstance(data, dict):
            log(INFO, "Computing svd")
            n, _ = data.shape
            data = self.row_norm(data)
            k = n if n < 500 else max(200, round(n / 4))

            if k == n:
                u, s, vt = np.linalg.svd(data)

            else:
                u, s, vt = randomized_svd(data, n_components=k, n_iter=3)

            uu = {"u": u, "d": s, "v": vt.T}
        else:
            if "d" in data and method == "permutation":
                log(
                    INFO,
                    "Original data is needed for permutation method. Setting method to elbow",
                )
                method = "elbow"

            uu = data

        if method == "permutation":
            dstat = np.square(uu["d"][:k]) / np.sum(np.square(uu["d"][:k]))
            dstat0 = np.zeros((B, k))

            for i in range(B):
                dat0 = np.apply_along_axis(
                    np.random.permutation, 1, data
                ).T  # why 1 now, when normally in should be 0, since in R is backwards..

                if k == n:
                    u, s, vt = np.linalg.svd(dat0)

                else:
                    u, s, vt = randomized_svd(data, n_components=k, n_iter=3)

                uu0 = {"u": u, "d": s, "v": vt.T}
                dstat0[i, :] = np.square(uu0["d"][:k]) / np.sum(np.square(uu0["d"][:k]))

            psv = [np.mean(dstat0[:, i] >= dstat[i]) for i in range(k)]

            for i in range(1, k):
                psv[i] = max(psv[i - 1], psv[i])

            nsv = np.sum(np.array(psv) <= 0.1)

        elif method == "elbow":  # tested
            xraw = np.abs(np.diff(np.diff(uu["d"])))
            x = smooth(xraw, kind="3RS3R", twiceit=True)
            prob = scipy.stats.mstats.mquantiles(a=x, prob=0.5, alphap=1, betap=1)[0]
            nsv: int = np.where(x <= prob)[0][0] + 2

        return nsv

    @staticmethod
    def pinv_ridge(m: np.ndarray, alpha=0) -> np.ndarray:  # tested
        """
        Calculates the pseudo-inverse of a matrix using ridge regularization.

        Parameters
        ----------
        m : pd.DataFrame
            The input matrix.

        alpha : float, optional
            Ridge regularization parameter. Defaults to 0 (no regularization).

        Returns
        -------
        pd.DataFrame
            The pseudo-inverse of the input matrix.
        """

        # maybe use r svd
        msvd = np.linalg.svd(m, full_matrices=False)  # SVD decomposition
        u, s, vt = msvd

        if len(s) == 0:
            return np.zeros((m.shape[1], m.shape[0]))
        else:
            if alpha > 0:
                ss = s**2 + alpha**2
                s = ss / s
            res = vt.T @ np.diag(1.0 / s) @ u.T
            return res

    @staticmethod
    def solveU(
        Z: np.ndarray,
        C_hat: np.ndarray,
        prior_mat: np.ndarray,
        penalty_factor: np.ndarray,
        pathway_selection: Literal["fast", "complete"] = "fast",
        glm_alpha: float = 0.9,
        max_path: int = 10,
        target_frac: float = 0.7,
        L3: float | None = None,
        seed=1234,
    ) -> Union[dict, np.ndarray]:
        """
        Solves for the U coefficients making efficient utilization of the lasso path.

        Parameters
        ----------
        Z : np.ndarray
            Current Z estimate.
        C_hat : np.ndarray
            The inverse of the C matrix.
        prior_mat : np.ndarray
            The prior pathway or C matrix.
        penalty_factor : np.ndarray
            Penalties for different pathways, must have size ncol(prior_mat).
        pathway_selection : str
            Method to use for pathway selection.
        glm_alpha : float
            The elastic net alpha parameter.
        max_path : int
            The maximum number of pathways to consider.
        target_frac : float
            The target fraction of non-zero columns.
        L3 : np.ndarray
            Solve with a given L3, no search.

        Returns
        -------
        np.ndarray
            The U coefficients.
        """

        from scipy import stats
        from sklearn.linear_model import ElasticNet as SklearnElasticNet
        import numpy as np

        class CustomElasticNet:
            def __init__(
                self,
                alpha=0.5,
                lambda_path=None,
                lower_limits=None,
                fit_intercept=True,
                standardize=False,
                random_state=None,
            ):
                self.alpha = alpha  # l1_ratio in sklearn (mix between l1 and l2)
                self.lambda_path = (
                    lambda_path if lambda_path is not None else np.array([0.1])
                )
                self.positive = lower_limits is not None and lower_limits >= 0
                self.fit_intercept = fit_intercept
                self.standardize = standardize
                self.random_state = random_state
                self.coef_path_ = None
                self.intercept_path_ = None

            def fit(self, X, y, relative_penalties=None):
                n_lambdas = len(self.lambda_path)
                n_features = X.shape[1]

                self.coef_path_ = np.zeros((n_features, n_lambdas))
                self.intercept_path_ = np.zeros(n_lambdas)
                # Handle relative penalties if provided
                if relative_penalties is not None:
                    # Convert to numpy array if it's a list
                    relative_penalties = np.asarray(relative_penalties)

                    # In glmnet, features with higher penalties are penalized more
                    # We can implement this by scaling features before fitting
                    # (inverting because higher penalties should shrink coefficients more)
                    feature_weights = 1.0 / np.sqrt(relative_penalties)

                    # Prevent division by zero or infinity
                    feature_weights = np.where(
                        np.isfinite(feature_weights), feature_weights, 1.0
                    )

                    # Scale X columns by feature weights
                    X_scaled = X * feature_weights.reshape(1, -1)
                else:
                    X_scaled = X
                    feature_weights = np.ones(X.shape[1])

                for i, alpha_value in enumerate(self.lambda_path):
                    model = SklearnElasticNet(
                        alpha=alpha_value,
                        l1_ratio=self.alpha,
                        fit_intercept=self.fit_intercept,
                        positive=self.positive,
                        random_state=self.random_state,
                    )
                    model.fit(X_scaled, y)

                    self.coef_path_[:, i] = model.coef_
                    self.intercept_path_[i] = model.intercept_

                return self

        # from glmnet import ElasticNet
        # from glmnet.util import _check_user_lambda, _interpolate_model
        # from sklearn.utils import check_array

        # class CustomElasticNet(ElasticNet):
        #     def decision_function(self, X, lamb=None):
        #         # Copy the original method's logic
        #         lambda_best = None
        #         if hasattr(self, "lambda_best_"):
        #             lambda_best = self.lambda_best_

        #         lamb = _check_user_lambda(self.lambda_path_, lambda_best, lamb)
        #         coef, intercept = _interpolate_model(
        #             self.lambda_path_, self.coef_path_, self.intercept_path_, lamb
        #         )

        #         X = check_array(X, accept_sparse="csr")
        #         z = X.dot(coef) + intercept

        #         # # Apply fix: only squeeze if the last dimension has size 1
        #         # if lamb.shape[0] == 1 and z.shape[-1] == 1:
        #         #     z = z.squeeze(axis=-1)

        #         return z

        penalty_factor = np.asarray(penalty_factor)
        # prior_mat_columns = prior_mat.columns

        Ur = C_hat @ Z  # perform OLS regression

        Ur = stats.rankdata(-Ur, axis=0)

        Urm = np.min(Ur, axis=1)  # same as R if Ur is identical
        U = np.zeros((prior_mat.shape[1], Z.shape[1]))

        if L3 is None:
            results = []
            lambdas = np.exp(np.arange(-4, -12, -0.125))
            lMat = np.empty((len(lambdas), Z.shape[1]))

            for i in range(Z.shape[1]):
                if pathway_selection == "fast":
                    iip = np.where(Ur[:, i] <= max_path)[0]
                elif pathway_selection == "complete":
                    iip = np.where(Urm <= max_path)[0]

                if iip.size == 0:
                    log(INFO, f"Empty selection for column {i}")
                    continue  # Skip this iteration

                y = Z[:, i].copy()
                x = prior_mat[:, iip].copy()

                m = CustomElasticNet(
                    alpha=glm_alpha,
                    lambda_path=lambdas,
                    lower_limits=0,
                    fit_intercept=True,
                    standardize=False,
                    random_state=seed,
                )

                m = m.fit(x, y, relative_penalties=penalty_factor[iip])

                res_beta = m.coef_path_

                lMat[:, i] = np.sum(res_beta, axis=0) > 0
                results.append({"beta": res_beta, "iip": iip})

            fracs = (lMat > 0).mean(axis=1)
            iibest = np.argmin(np.abs(target_frac - fracs))  # same as R

            for i in range(Z.shape[1]):
                U[results[i]["iip"], i] = results[i]["beta"][:, iibest]

            return {"U": U, "L3": lambdas[iibest]}
        else:
            for i in range(Z.shape[1]):
                if pathway_selection == "fast":
                    iip = np.where(Ur[:, i] <= max_path)[0]
                elif pathway_selection == "complete":
                    iip = np.where(Urm <= max_path)[0]

                y = Z[:, i].copy()
                x = prior_mat[:, iip].copy()

                m = CustomElasticNet(
                    alpha=glm_alpha,
                    lambda_path=np.array([L3]),
                    lower_limits=0,
                    fit_intercept=True,
                    standardize=False,
                    random_state=seed,
                )
                m = m.fit(x, y, relative_penalties=penalty_factor[iip])

                res_beta = m.coef_path_

                # (nrows,) instead of (nrows, 1)
                U[iip, i] = res_beta.flatten()

            return U

    def _to_pandas(self) -> None:
        """
        Converts back all the np.ndarrays back to pd.DataFrames
        """

        self.U = pd.DataFrame(self.U)
        self.B = pd.DataFrame(self.B)
        self.Z = pd.DataFrame(self.Z)
        self.C = pd.DataFrame(self.C)
        # self.out["Uauc"] = pd.DataFrame(self.out["Uauc"])
        # self.out["Up"] = pd.DataFrame(self.out["Up"])

        self.B.index = self.Z.columns = self.U.columns = self.LVs
        self.B.columns = self.samples
        self.Z.index = self.sig_genes
        self.U.index = self.signatures
        self.C.index, self.C.columns = self.sig_genes, self.signatures

        # # Update output of plier after calculating AUC and stuff
        # self.out["B"] = self.B
        # self.out["U"] = self.U
        # self.out["Z"] = self.Z
        # self.out["C"] = self.C

        # self.out["Uauc"].index = self.U.index
        # self.out["Up"].index = self.U.index

        # self.out["Uauc"].columns = self.U.columns
        # self.out["Up"].columns = self.U.columns


class FplierPreprocessor:
    @staticmethod
    def get_common_genes(data, data_genes, prior_mat, sig_genes, all_genes=False):
        """
        Process input data and the prior matrix to ensure consistent gene set.

        If the gene set in data and prior_mat is not consistent, there are two strategies:
        - If all_genes is False, find the common genes in data and prior_mat, and filter both matrices to only contain those.
        - If all_genes is True, expand the prior matrix to include all genes in the data. The extra rows in prior_mat are filled with zeros.

        Parameters
        ----------
        data : np.ndarray
            The input data matrix
        data_genes : list
            List of gene names in the data matrix
        prior_mat : np.ndarray
            The prior matrix
        sig_genes : list
            List of gene names in the prior matrix
        all_genes : bool, optional
            Whether to include all genes or only common ones, by default False

        Returns
        -------
        tuple
            (filtered_data, filtered_data_genes, filtered_prior_mat, filtered_sig_genes)
        """
        # Create copies to avoid modifying original data
        Y = data.copy()
        prior_mat_copy = prior_mat.copy()
        data_genes_copy = data_genes.copy()
        sig_genes_copy = sig_genes.copy()

        # define row name to index mapping for data and prior_mat
        sig_rn_to_index = {name: index for index, name in enumerate(sig_genes_copy)}
        data_rn_to_index = {name: index for index, name in enumerate(data_genes_copy)}

        # find extra genes in data_genes not present in sig_genes
        extra_genes = list(set(data_genes_copy) - set(sig_genes_copy))

        if len(prior_mat_copy) != len(Y) or not np.array_equal(
            sig_genes_copy, data_genes_copy
        ):
            if not all_genes:
                # Get common names using FplierUtils
                cm = FplierUtils.common_names(data_genes_copy, sig_genes_copy)
                log(INFO, f"[i] Selecting common genes: {len(cm)}")
                # get indices corresponding to cm
                sig_indices = np.array([sig_rn_to_index[name] for name in cm])
                data_indices = np.array([data_rn_to_index[name] for name in cm])

                # subset prior_mat and Y using cm_indices
                filtered_prior_mat = prior_mat_copy[sig_indices, :]
                filtered_Y = Y[data_indices, :]
                # update the column and row names
                filtered_sig_genes = filtered_data_genes = cm
            else:
                # create a zero-filled matrix for extra genes
                eMat = np.zeros((len(extra_genes), prior_mat_copy.shape[1]))
                # add extra genes to prior_mat
                extended_prior_mat = np.vstack([prior_mat_copy, eMat])
                # extend sig_genes with the extra genes
                extended_sig_genes = list(sig_genes_copy) + extra_genes
                # update the prior_mat rowname to index mapping with extra genes
                extended_sig_rn_to_index = sig_rn_to_index.copy()
                extended_sig_rn_to_index.update(
                    {
                        name: index
                        for index, name in enumerate(
                            extra_genes, start=len(sig_rn_to_index)
                        )
                    }
                )
                # reorder prior_mat and sig_genes to match the order of data_genes
                filtered_prior_mat = extended_prior_mat[
                    np.array(
                        [extended_sig_rn_to_index[name] for name in data_genes_copy]
                    ),
                    :,
                ]
                filtered_sig_genes = filtered_data_genes = np.array(
                    [
                        extended_sig_genes[extended_sig_rn_to_index[name]]
                        for name in data_genes_copy
                    ]
                )
                filtered_Y = Y
        else:
            # No adjustment needed
            filtered_Y = Y
            filtered_data_genes = data_genes_copy
            filtered_prior_mat = prior_mat_copy
            filtered_sig_genes = sig_genes_copy

        return filtered_Y, filtered_data_genes, filtered_prior_mat, filtered_sig_genes

    @staticmethod
    def get_C(prior_mat, sig_genes, signatures, do_crossval=False, seed=None):
        """
        Perform cross-validation setup on prior matrix if do_crossval is True.

        In the cross-validation setup, for each pathway in the prior matrix, about 20% of the genes are randomly selected and held out in the prior_mat_cv,
        i.e., their pathway association is set to zero. The held out genes are stored in held_out_genes for each pathway.
        The seed for random sampling can be set by the seed argument for reproducibility.

        Parameters
        ----------
        prior_mat : np.ndarray
            The prior matrix
        sig_genes : list
            List of gene names in the prior matrix
        signatures : list
            List of signature names (column names of prior_mat)
        do_crossval : bool, optional
            Whether to perform cross-validation, by default False
        seed : int, optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, dict] | Tuple[np.ndarray, dict]
            The C matrix, typically extracted from literature and agreed among clients
            (if do_crossval) A matrix same as prior_mat parameter, with some random samples that are 0, to use it in cross validation.
            The held out genes
        """
        held_out_genes = dict()

        # branch if doing cross validation or not
        if do_crossval:
            prior_mat_cv = prior_mat.copy()

            if seed is not None:
                np.random.seed(seed)

            for j in range(prior_mat_cv.shape[1]):
                iipos = np.where(prior_mat_cv[:, j] > 0)[0]  # this is same as in R
                iipos_sample = np.random.choice(
                    iipos, size=round(len(iipos) / 5), replace=False
                )

                prior_mat_cv[iipos_sample, j] = 0
                held_out_genes[signatures[j]] = np.asarray(sig_genes)[
                    iipos_sample
                ].tolist()

            C = prior_mat_cv
            return C, prior_mat_cv, held_out_genes
        else:
            C = prior_mat.copy()
            return C, held_out_genes
