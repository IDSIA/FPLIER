"""Manages the Flower ClientApp for federated learning."""

import time
import pandas as pd
import numpy as np
from typing import Literal

from logging import INFO, DEBUG, CRITICAL, WARNING

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, log

from app.task import (
    # load_data,
    load_data_simulation,
    get_random_vector,
    FplierUtils,
    FplierPreprocessor,
)

import json
import os
import toml
from sklearn.utils.extmath import randomized_svd


# Define Flower Client
class FlowerClient(
    NumPyClient,
    FplierUtils,
):

    # Initilize Flower Client
    def __init__(
        self,
        timeout,  # for flower
        client_id,
        # ------------
        # already calculated before first Z
        Y,
        C,
        B,
        Z,
        prior_mat_cv,
        held_out_genes,
        data_genes,
        sig_genes,
        signatures,
        samples,
        ns,
        # ------------
        svdres: dict | None = None,
        C_hat: np.ndarray | None = None,
        k: int | None = None,
        L1: float | None = None,
        L2: float | None = None,
        L3: float | None = None,
        frac: float = 0.7,
        max_iter: int = 350,
        trace: bool = False,
        scale: bool = True,
        max_path: int = 10,
        do_crossval: bool = True,
        penalty_factor: np.ndarray | None = None,
        glm_alpha: float = 0.9,
        min_genes: int = 10,
        tol: float = 1e-6,
        seed: int = 12345,
        all_genes: bool = False,
        rseed: int | None = None,
        pathway_selection: Literal["complete", "fast"] = "complete",
    ):
        self.timeout = timeout
        self.client_id = client_id
        temp_ = toml.load("pyproject.toml")
        self.num_clients = temp_["tool"]["flwr"]["federations"]["local-simulation"][
            "options"
        ]["num-supernodes"]

        self.Bdiff = -1
        self.Bdiff_count = 0
        # + 1 because the first iteration are used to calculate L1 and L2
        self.iter_full: int = 20
        self.iter_full_start: int = 20

        # Initialize PLIERobject
        # self.data = data
        # self.prior_mat = prior_mat
        self.svdres = svdres
        self.k: int = k if isinstance(k, int) else None

        # self.L1 = L1
        # self.L2 = L2
        # self.L3 = L3
        self.frac = frac if frac <= 1 else frac / 100
        self.max_iter = max_iter
        self.trace = trace
        self.scale = scale
        self.C_hat = C_hat
        self.max_path = max_path
        self.do_crossval = do_crossval
        self.penalty_factor = penalty_factor
        self.glm_alpha = glm_alpha
        self.min_genes = min_genes
        self.tol = tol
        self.seed = int(seed)
        self.all_genes = bool(all_genes)
        self.rseed = rseed
        self.pathway_selection = pathway_selection

        self.Y = Y
        self.C = C
        self.C_hat = C_hat
        self.prior_mat_cv = prior_mat_cv
        self.held_out_genes = held_out_genes
        self.ns = ns

        # initially setting U to the zero matrix
        # self.U = np.zeros((self.C.shape[1], self.k))

        self.data_genes = data_genes  # Y rows
        self.sig_genes = sig_genes  # C rows
        self.signatures = signatures  # C columns
        self.samples = samples  # Y columns

        self.LVs = [f"LV{i:0{len(str(self.k))}}" for i in range(1, self.k + 1)]

        # Check if previously calculated Lambda values exist
        self.iter_start = f"tmp/iter_start_{self.client_id}.json"
        if os.path.exists(self.iter_start):
            with open(self.iter_start, "r") as f:
                vals = json.load(f)
                self.Bdiff, self.Bdiff_count, self.iter_full, self.iter_full_start = (
                    vals["Bdiff"],
                    vals["Bdiff_count"],
                    vals["iter_full"],
                    vals["iter_full_start"],
                )

        # Check if previously calculated Lambda values exist
        self.Lambda_path = f"tmp/Lambdas_{self.client_id}.json"
        if os.path.exists(self.Lambda_path):
            with open(self.Lambda_path, "r") as f:
                vals = json.load(f)
                self.L1, self.L2 = vals["L1"], vals["L2"]
        else:
            self.L1, self.L2 = L1, L2

        # Check if previously calculated U matrix exists
        self.U_path = f"tmp/U_{self.client_id}.npy"
        if os.path.exists(self.U_path):
            self.U = np.load(self.U_path).astype(np.float64)
        else:
            self.U = np.zeros((self.C.shape[1], self.k))

        self.B_path = f"tmp/B_{self.client_id}.npy"
        if os.path.exists(self.B_path):
            self.B = np.load(self.B_path).astype(np.float64)
        else:
            self.B = B

        self.Z_path = f"tmp/Z_{self.client_id}.npy"
        if os.path.exists(self.Z_path):
            self.Z = np.load(self.Z_path).astype(np.float64)
        else:
            self.Z = Z

        # Check if previously calculated L3 value exists
        self.L3_path = f"tmp/L3_{self.client_id}.json"
        if os.path.exists(self.L3_path):
            with open(self.L3_path, "r") as f:
                vals = json.load(f)
                self.L3 = vals["L3"]
        else:
            self.L3 = L3

    # Generate a random N-component vector locally
    def fit(self, parameters, config):

        def get_new_Z():
            # Retrieve previous round's aggregate (if available, otherwise use initial Z)
            if "aggregated_matrix" in config:
                return np.array(json.loads(config["aggregated_matrix"]))

        i = int(config["round_number"])  # current iteration

        # Now, since Z is calculated at the end of the round, we get that updated Z at the beginning of the next round.
        # Therefore, here we get the Upadted Z to calculate B,

        # Use new Z to calculate B
        old_b = self.B.copy()

        if i > 1:
            new_Z = (
                get_new_Z()
            )  # returns the Z that is calculated using data from last iteration.

            new_Z[new_Z < 0] = 0

            new_B = (
                np.linalg.inv((new_Z.T @ new_Z) + (self.L2 * np.eye(self.k)))
                @ new_Z.T
                @ self.Y
            )
        ######################################################################################

        # for loop start
        # if not self.trace:
        #     log(INFO, f"[i] Converging... iteration {i}/{self.max_iter}")
        if i >= self.iter_full_start:
            # NOTE: if this condition is not met, U is a 0 matrix, since it is initialized to 0, and at each round the client is re-initialized
            # # branch if L3 is given, find L3 and U, and save it in a dict
            if i == self.iter_full and self.L3 is None:
                # update L3 to the target fraction if not provided
                Ulist = self.solveU(
                    self.Z,  # this is the "old" Z
                    self.C_hat,
                    self.C,
                    self.penalty_factor,
                    self.pathway_selection,
                    self.glm_alpha,
                    self.max_path,
                    target_frac=self.frac,
                    seed=self.seed,
                )

                self.U = Ulist["U"]
                self.L3 = Ulist["L3"]
                log(
                    INFO,
                    f"[new] L3 is set to {round(self.L3, 7)}",
                )
                with open(self.L3_path, "w") as f:
                    json.dump({"L3": self.L3}, f)
                    # log(
                    #     INFO,
                    #     f"[i] L3 saved to {self.L3_path}",
                    # )

                self.iter_full += self.iter_full_start

            else:  # otherwise if L3 is given return just U
                self.U = self.solveU(
                    self.Z,
                    self.C_hat,
                    self.C,
                    self.penalty_factor,
                    self.pathway_selection,
                    self.glm_alpha,
                    self.max_path,
                    L3=self.L3,
                    seed=self.seed,
                )

        # stuff for calculating the new Z. This is basically half of "run_fed()" in the simulation code
        # still use old B to calculate new Z

        K, L = Z_t_plus_1(
            self.Y, old_b, self.C, self.U, L1=self.L1, ncli=self.num_clients
        )

        np.save(self.U_path, self.U)
        # log(
        #     INFO,
        #     f"[i] U saved to {self.U_path}",
        # )

        # new_Z in this round will be the old_Z in the next, that will be used to calculate U
        if i > 1:
            np.save(self.Z_path, new_Z)
            # log(
            #     INFO,
            #     f"[i] Z saved to {self.Z_path}",
            # )
            # save B and U and iteration start variables to file for next round
            self.B = new_B  # update old B with new B, so also next Z will change
            # self.U = new_U
            np.save(self.B_path, self.B)
            # log(
            #     INFO,
            #     f"[i] B saved to {self.B_path}",
            # )

        # save to file iter_start etc.
        with open(self.iter_start, "w") as f:
            json.dump(
                {
                    "Bdiff": self.Bdiff,
                    "Bdiff_count": self.Bdiff_count,
                    "iter_full": self.iter_full,
                    "iter_full_start": self.iter_full_start,
                },
                f,
            )
            # log(
            #     INFO,
            #     f"[i] iter_start values saved to {self.iter_start}",
            # )

        # clients calculate this and send to server
        # K = Y @ B.T + ((L1 / number_of_clients) * (self.C @ self.U))
        # L = B @ B.T + ((L1 / number_of_clients) * I)
        # server recieves and aggregates Ks and Ls from clients, and calculates Z:
        # Z = aggregated_Ks @ inv(aggregated_Ls)

        if i > 1:
            self.Bdiff = float(
                np.sum((new_B - old_b) ** 2) / np.sum(new_B ** 2)
            )

        metrics = {
            "Bdiff": self.Bdiff,
            "errorY": float(np.mean((self.Y - (self.Z @ self.B)) ** 2)),
        }

        if i == self.max_iter:
            self._to_pandas()  # doing in place

            # print(self.Z.head())
            self.U.to_csv(f"tmp/U_{self.client_id}.csv")
            self.B.to_csv(f"tmp/B_{self.client_id}.csv")
            self.Z.to_csv(f"tmp/Z_{self.client_id}.csv")

        return [K, L], 2, metrics  # [res1, res2, ...], len_of_results, {metrics}


def Z_t_plus_1(
    Y, B, C, U, L1, ncli
):

    k_B = B.shape[0]
    I = np.eye(k_B)

    K = Y @ B.T + (
        (L1 / ncli) * (C @ U)
    )
    L = B @ B.T + ((L1 / ncli) * I)

    return K.astype(np.float64), L.astype(np.float64)


def client_fn(context: Context) -> FlowerClient:
    """Client function to be called by the Flower ClientApp."""
    # Extract configuration
    timeout = context.run_config["timeout"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    seed_value = context.run_config["seed"]

    # Extract general parameters
    params = {
        "k": context.run_config["k"] if context.run_config["k"] != "null" else None,
        "L1": context.run_config["L1"] if context.run_config["L1"] != "null" else None,
        "L2": context.run_config["L2"] if context.run_config["L2"] != "null" else None,
        "L3": context.run_config["L3"] if context.run_config["L3"] != "null" else None,
        "all_genes": context.run_config["all-genes"],
        "scale": context.run_config["scale"],
        "frac": context.run_config["frac"],
        "max_iter": context.run_config["max-iter"],
        "do_crossval": context.run_config["do-crossval"],
        "trace": context.run_config["trace"],
        "max_path": context.run_config["max-path"],
        "glm_alpha": context.run_config["glm-alpha"],
        "min_genes": context.run_config["min-genes"],
        "tol": context.run_config["tol"],
        "seed": context.run_config["seed"],
        "rseed": (
            context.run_config["rseed"]
            if context.run_config["rseed"] != "null"
            else None
        ),
        "pathway_selection": (
            context.run_config["pathway-selection"]
            if context.run_config["pathway-selection"] != "null"
            else None  # TODO: load from path if provided
        ),
        "C_hat": (
            context.run_config["C-hat"]
            if context.run_config["C-hat"] != "null"
            else None  # TODO: load from path if provided
        ),
        "penalty_factor": (
            context.run_config["penalty-factor"]
            if context.run_config["penalty-factor"] != "null"
            else None  # TODO: load from path if provided
        ),
    }

    # Load the data (only once per client)
    # gene_data = load_data(partition_id, num_partitions, "Y", seed_value)
    # prior_mat = load_data(partition_id, num_partitions, "C", seed_value)

    gene_data, prior_mat = load_data_simulation(partition_id)

    # NOTE: Pre-process the data (only once per client)
    preprocessed_data = preprocess_client_data(gene_data, prior_mat, params)

    # Create and return the client
    return FlowerClient(
        timeout=timeout,
        client_id=partition_id,
        **preprocessed_data,
        **params,
    ).to_client()


def preprocess_client_data(gene_data, prior_mat, params):
    """Preprocess the data once before training rounds begin."""
    log(INFO, f"Genes data loaded shape: {gene_data.shape}")
    log(INFO, f"Prior knowledge loaded shape: {prior_mat.shape}")

    # Extract metadata
    data_genes = gene_data.index.tolist()
    sig_genes = prior_mat.index.tolist()
    signatures = prior_mat.columns
    samples = gene_data.columns

    # Convert to numpy for processing
    Y = (
        gene_data.values.astype(np.float64) if not params["scale"] else None
    )  # TODO: Handle scaling if needed with row_norm (TO IMPLEMENT)
    prior_mat = prior_mat.values.astype(np.float64)

    # set penalty factor if none is passed
    if params["penalty_factor"] is None:
        params["penalty_factor"] = np.array([1] * prior_mat.shape[1])

    params["pathway_selection"] = (
        params["pathway_selection"]
        if params["pathway_selection"] in ["complete", "fast"]
        else "complete"
    )  # Default value

    # Process common genes
    Y, data_genes, prior_mat, sig_genes = FplierPreprocessor.get_common_genes(
        Y, data_genes, prior_mat, sig_genes, all_genes=params["all_genes"]
    )

    # Filter pathways with too few genes
    num_genes = np.sum(prior_mat, axis=0)
    iibad = np.where(num_genes < params["min_genes"])[0]
    prior_mat[:, iibad] = 0
    log(INFO, f"[i] Removing {len(iibad)} pathways with too few genes")

    # Generate C matrices
    if params["do_crossval"]:
        C, prior_mat_cv, held_out_genes = FplierPreprocessor.get_C(
            prior_mat, sig_genes, signatures, do_crossval=True, seed=params["seed"]
        )
    else:
        C, held_out_genes = FplierPreprocessor.get_C(
            prior_mat, sig_genes, signatures, do_crossval=False, seed=params["seed"]
        )
        prior_mat_cv = None

    # Calculate C_hat if not provided
    if params["C_hat"] is None:
        # directly save the C_hat inside the params dict
        params["C_hat"] = FplierUtils.pinv_ridge(C.T @ C, 5) @ C.T

    # np.save("testing/C.npy", C)

    # Get the number of samples (columns) in the matrix Y
    n_samples = Y.shape[1]

    # calculate SVDloc (local SVD for the current client's data)
    if n_samples > 500:
        np.random.seed(params["seed"])
        n_components = min(n_samples, round(max(200, n_samples / 4)))
        u, s, vt = randomized_svd(
            Y,
            n_components=n_components,
            n_iter=3,
        )
        
        # For randomized_svd: s is already 1D, vt is already transposed
        svdres = {"u": u, "d": s, "v": vt.T}  # Note: vt.T to get V

    else:
        u, s, vt = np.linalg.svd(Y, full_matrices=False)
        
        # For np.linalg.svd: s is 1D, vt needs to be transposed to get V
        print(u.shape, s.shape, vt.shape)
        svdres = {"u": u, "d": s, "v": vt.T}

    v = svdres["v"]
    d = np.diag(svdres["d"])

    if params["k"] > d.shape[0]:
        zeros = np.zeros(shape=(params["k"] - d.shape[0], d.shape[1]))
        d = np.vstack((d, zeros))
    else:
        d = d[: params["k"], :]

    # Ensure d has the right shape for matrix multiplication
    if len(d.shape) == 1:
        d = np.diag(d)
    
    # Take only the first k components if needed
    d_k = d[:params["k"], :params["k"]] if d.shape[0] >= params["k"] else d
    v_k = v[:, :params["k"]] if v.shape[1] >= params["k"] else v

    B = np.sqrt(d_k) @ v_k.T

    m, _ = Y.shape  # m: rows in Y, n: columns in Y (also rows in B.T)
    _, k = B.T.shape  # n: rows in B.T (matches Y), k: columns in B.T

    # Directly set Z with the desired shape (m, k)
    # (in the case of smallmat (829, 100)) filled with zeros or any dummy data
    Z = np.zeros(shape=(m, k))

    return {
        "Y": Y,
        "C": C,
        "B": B,
        "Z": Z,
        "prior_mat_cv": prior_mat_cv,
        "held_out_genes": held_out_genes,
        "data_genes": data_genes,
        "sig_genes": sig_genes,
        "signatures": signatures,
        "samples": samples,
        "ns": n_samples,
    }


# ----------------------------------------------------------------------------
# Flower ClientApp
_run_secagg = (
    toml.load("pyproject.toml")["tool"]["flwr"]["app"]["config"].get("run-secagg", False)
)

app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod] if _run_secagg else [],
)
