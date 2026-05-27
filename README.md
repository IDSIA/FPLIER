# FPLIER: Federated Pathway-Level Information ExtractoR

Keywords: Transcriptomics, Gene-set-aware factorization, Deconvolution, Federated learning, Membership inference attack, Privacy

## Abstract:
In transcriptomics, gene-set-aware factorization methods such as the Pathway Level Information Extractor (PLIER) are most effective when trained on large, heterogeneous expression compendia. Yet, many clinically relevant cohorts cannot be pooled into a single dataset due to privacy and governance constraints. We present FPLIER, a federated extension of PLIER that enables distributed training across multiple data holders while incorporating publicly available datasets. Through secure aggregation, FPLIER produces training updates algebraically equivalent to those of a centralized pooled-data approach while keeping expression data local. We evaluate FPLIER across multiple scenarios in two simulated consortia (from the K-CLIER and MultiPLIER studies) and demonstrate stable convergence. We further conduct a systematic analysis of membership inference attacks targeting both intermediate training statistics and the released model. Our results show that privacy risk is governed by the rank of the training expression matrix. Incorporating public data or reducing data dimensionality increases this rank, moving the system toward a full-rank regime in which training and non-training samples become indistinguishable to the attacker, and membership-inference performance approaches random guessing.


## Code

For more information on Flower, we refer the reader to [Flower's webpage](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html).


### Run the federated simulation with the Simulation Engine

In the `.` directory, use `flwr run` to run a local simulation of the federated process:

```bash
flwr run .
```
