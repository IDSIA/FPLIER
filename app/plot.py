"""Reconstruction plot comparing federated and centralized PLIER results."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import scipy.stats
import seaborn as sns

from app.task import load_data_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_federated_result(num_clients: int = 3, tmp_dir: str = "tmp") -> tuple[np.ndarray, np.ndarray]:
    """
    Load the federated PLIER result from tmp/.

    Z is shared across all clients (use client 0).
    B is per-client and concatenated along the samples axis.

    Returns
    -------
    Z : np.ndarray, shape (n_genes, k)
    B : np.ndarray, shape (k, total_samples)
    """
    Z = np.load(os.path.join(tmp_dir, "Z_0.npy")).astype(np.float64)
    B = np.concatenate(
        [np.load(os.path.join(tmp_dir, f"B_{i}.npy")).astype(np.float64)
         for i in range(num_clients)],
        axis=1,
    )
    return Z, B


def load_full_expression(num_clients: int = 3) -> pd.DataFrame:
    """
    Load and concatenate expression matrices from all clients in sample order.

    Returns a DataFrame with genes as index and all samples as columns,
    preserving the same column order used when building B.
    """
    frames = [load_data_simulation(i)[0] for i in range(num_clients)]
    return pd.concat(frames, axis=1)


def load_server_expression(server_data_dir: str = "data/server") -> pd.DataFrame:
    """
    Load the complete gene expression matrix from the server data directory.

    Returns a DataFrame with genes as index and all samples as columns.
    """
    return pyreadr.read_r(os.path.join(server_data_dir, "gene_matrix_test.RDS"))[None]


def build_centralized_result(
    fed_Z: np.ndarray,
    data_genes: list[str],
    server_df: pd.DataFrame,
    L2: float,
) -> dict:
    """
    Compute the optimal centralized B from the full server dataset using the
    federated Z via ridge regression: B = inv(Z.T @ Z + L2·I) @ Z.T @ Y.

    Parameters
    ----------
    fed_Z : np.ndarray, shape (n_genes, k)
        Federated gene-loading matrix.
    data_genes : list[str]
        Gene names for the rows of ``fed_Z``.
    server_df : pd.DataFrame, shape (n_genes, n_samples)
        Full expression matrix loaded from ``data/server``.
    L2 : float
        Ridge regularisation parameter (same L2 used during federated training).

    Returns
    -------
    dict with keys ``"Z"``, ``"B"``, ``"data_genes"`` ready for
    ``plot_reconstruction``.
    """
    gene_to_idx = {g: i for i, g in enumerate(data_genes)}
    common = sorted(set(data_genes) & set(server_df.index))

    z_idx = np.array([gene_to_idx[g] for g in common])
    Z_sub = fed_Z[z_idx, :]                      # (n_common, k)
    Y = server_df.loc[common, :].values          # (n_common, n_samples)

    k = Z_sub.shape[1]
    B_cen = np.linalg.inv(Z_sub.T @ Z_sub + L2 * np.eye(k)) @ Z_sub.T @ Y

    return {"Z": Z_sub, "B": B_cen, "data_genes": common}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _rec_corr(
    Z: np.ndarray,
    B: np.ndarray,
    data_genes: list[str],
    exprs_df: pd.DataFrame,
) -> list[float]:
    """
    Compute per-gene Spearman correlation between the original and reconstructed
    expression matrix.

    Parameters
    ----------
    Z : np.ndarray, shape (n_genes, k)
        Gene-loading matrix.
    B : np.ndarray, shape (k, n_samples)
        Sample-loading matrix.
    data_genes : list[str]
        Gene names corresponding to the rows of Z.
    exprs_df : pd.DataFrame, shape (n_genes, n_samples)
        Original expression matrix, indexed by gene names.

    Returns
    -------
    list[float]
        Per-gene Spearman correlations between original and reconstruction.
    """
    gene_to_idx = {g: i for i, g in enumerate(data_genes)}
    common = sorted(set(data_genes) & set(exprs_df.index))

    z_idx = np.array([gene_to_idx[g] for g in common])
    Z_sub = Z[z_idx, :]               # (n_common, k)
    Y_rec = Z_sub @ B                 # (n_common, n_samples)
    Y_orig = exprs_df.loc[common, :].values  # (n_common, n_samples)

    return [
        scipy.stats.spearmanr(Y_orig[i, :], Y_rec[i, :])[0]
        for i in range(Y_orig.shape[0])
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstruction(
    exprs_df: pd.DataFrame | None = None,
    fed_Z: np.ndarray | None = None,
    fed_B: np.ndarray | None = None,
    fed_data_genes: list[str] | None = None,
    centralized: dict | None = None,
    num_clients: int = 3,
    tmp_dir: str = "tmp",
    save: bool = False,
    save_dir: str = "figures",
    suffix: str = "",
) -> None:
    """
    KDE plot of per-gene Spearman reconstruction correlations for the federated
    result and, optionally, a centralized baseline.

    Parameters
    ----------
    exprs_df : pd.DataFrame, optional
        Full expression matrix (genes × samples). If None, loaded automatically
        from all clients via ``load_full_expression``.
    fed_Z : np.ndarray, optional
        Federated Z matrix (genes × k). If None, loaded from ``tmp_dir``.
    fed_B : np.ndarray, optional
        Federated B matrix (k × total_samples). If None, loaded from ``tmp_dir``.
    fed_data_genes : list[str], optional
        Gene names for the rows of ``fed_Z``. If None, inferred from client 0 data.
    centralized : dict, optional
        Centralized PLIER result for comparison. Expected keys:
          - ``"Z"``          : np.ndarray, shape (n_genes, k)
          - ``"B"``          : np.ndarray, shape (k, n_samples)
          - ``"data_genes"`` : list[str]
    num_clients : int
        Number of federated clients (used for auto-loading).
    tmp_dir : str
        Directory where the federated ``Z_*.npy`` / ``B_*.npy`` files are stored.
    save : bool
        If True, save the figure to ``save_dir`` instead of displaying it.
    save_dir : str
        Directory to save figures when ``save=True``.
    suffix : str
        Optional suffix appended to the saved filename.
    """
    # --- auto-load federated result ---
    if fed_Z is None or fed_B is None:
        fed_Z, fed_B = load_federated_result(num_clients, tmp_dir)

    if fed_data_genes is None:
        df0, _ = load_data_simulation(0)
        fed_data_genes = df0.index.tolist()

    # --- auto-load expression data ---
    if exprs_df is None:
        exprs_df = load_full_expression(num_clients)

    k = fed_Z.shape[1]

    # --- compute correlations ---
    fed_corr = _rec_corr(fed_Z, fed_B, fed_data_genes, exprs_df)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(x=fed_corr, fill=True, alpha=0.7, ax=ax, clip=[None, 1], label="Federated")

    if centralized is not None:
        cen_corr = _rec_corr(
            centralized["Z"],
            centralized["B"],
            centralized["data_genes"],
            exprs_df,
        )
        sns.kdeplot(x=cen_corr, fill=True, alpha=0.2, ax=ax, clip=[None, 1], label="Centralized")

    ax.set_title(f"Reconstruction test — All LVs, k = {k}")
    ax.set_xlabel("Per-gene Spearman Correlation")
    ax.legend(title="Method", loc="upper left")
    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"reconstruction_plot{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
