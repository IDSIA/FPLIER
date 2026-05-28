"""SecureSum: A Flower for custom secure sum strategy using SecAgg+."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logging import DEBUG, INFO, WARNING

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler

from flwr.common.record import ParametersRecord
from flwr.server import (
    Grid,
    LegacyContext,
    ServerApp,
    ServerAppComponents,
    ServerConfig,
)
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD

from app.task import get_dummy_start, load_data_simulation
from app.custom_strategy import FedSumEarlyStopping, EarlyStopException
from app.plot import (
    plot_reconstruction,
    load_federated_result,
    load_server_expression,
    build_centralized_result,
)
import os
import shutil


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Path to the tmp directory
    tmp_dir = "tmp/"

    # Check if tmp directory exists
    if os.path.exists(tmp_dir):
        # Remove all files and subdirectories in tmp/
        for item in os.listdir(tmp_dir):
            item_path = os.path.join(tmp_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                log(INFO, f"Deleted {item_path}")
            except Exception as e:
                log(WARNING, f"Error deleting {item_path}: {e}")
        log(INFO, "Cleared all contents from tmp/ directory")
    else:
        log(WARNING, "tmp/ directory does not exist")


    tol = float(context.run_config["tol"])
    min_rounds = int(context.run_config["iter-full-start"])

    # Define strategy
    strategy = FedSumEarlyStopping(
        fraction_fit=1.0,
        # Interrupt if any client fails
        accept_failures=False,
        # Disable evaluation
        fraction_evaluate=0.0,
        initial_parameters=ndarrays_to_parameters([get_dummy_start()]),
        tol=tol,
        min_rounds=min_rounds,
    )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=context.run_config["max-iter"]),
        strategy=strategy,
    )

    if context.run_config["run-secagg"]:

        # ------------------------------ SecAgg+ ------------------------------
        log(
            WARNING,
            "Running with SecAgg+",
        )

        # Create fit workflow
        fit_workflow = SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            timeout=context.run_config["timeout"],
        )

        # Create the workflow
        workflow = DefaultWorkflow(fit_workflow=fit_workflow)
        # ----------------------------- End SecAgg+ -----------------------------

    else:
        log(
            WARNING,
            "Running without SecAgg+",
        )
        workflow = DefaultWorkflow()

    # Execute
    try:
        workflow(grid, context)
    except EarlyStopException:
        log(INFO, "Early stopping triggered - training halted.")

    # Final result
    paramsrecord = context.state[MAIN_PARAMS_RECORD]
    ndarrays = ParametersRecord.to_numpy_ndarrays(paramsrecord)

    # Plot errorY convergence curve
    if strategy._errorY_history:
        rounds = list(range(1, len(strategy._errorY_history) + 1))
        plt.figure()
        plt.plot(rounds, strategy._errorY_history, linewidth=1.5)
        plt.xlabel("Round")
        plt.ylabel("errorY (MSE)")
        plt.title("errorY per round")
        plt.tight_layout()
        plot_path = os.path.join("figures", "errorY_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(INFO, f"errorY convergence plot saved to {plot_path}")

    # Plot reconstruction
    num_clients = len([f for f in os.listdir(tmp_dir) if f.startswith("B_") and f.endswith(".npy")])
    fed_Z, fed_B = load_federated_result(num_clients=num_clients, tmp_dir=tmp_dir)
    server_df = load_server_expression()
    L2 = float(context.run_config["L2"])
    dataset, signatures = load_data_simulation(0)
    data_genes = sorted(set(dataset.index.tolist()) & set(signatures.index.tolist()))
    centralized = build_centralized_result(fed_Z, data_genes, server_df, L2)
    plot_reconstruction(
        exprs_df=server_df,
        fed_Z=fed_Z,
        fed_B=fed_B,
        fed_data_genes=data_genes,
        centralized=centralized,
        save=True,
    )
    log(INFO, "Reconstruction plot saved to figures/reconstruction_plot.png")
