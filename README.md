# Efficient Marginal Reconstruction

This repository contains the code for the paper [Efficient and Private Marginal Reconstruction with Local Non-Negativity](https://arxiv.org/abs/2410.01091), which appeared at NeurIPS 2024.

## Description
Residuals-to-Marginals (ReM) is a privacy-preserving framework for reconstructing answers to marginal queries from noisy answers to residual queries. 

This repo contains the code for several marginal reconstruction algorithms based on ReM, including:
- **GReM-MLE**: an instantiation of ReM under Gaussian noise that uses maximum likelihood estimation to reconstruct marginals.
- **GReM-LNN**: an instantiation of ReM under Gaussian noise with local non-negativity constraints that uses maximum likelihood estimation to reconstruct marginals.
- **EMP** (Efficient Marginal Pseudoinversion): reconstructs answers to marginals from noisy answers to marginals using GReM-MLE. 

This repo additionally contains two mechanisms for differentially private query answering that use the above reconstruction methods:
- [**ResidualPlanner**](https://github.com/dkifer/ResidualPlanner): a mechanism for answering marginal queries that minimizes the noise added to the measurements.
- **Scalable MWEM**: a data-dependent mechanism for answering marginal queries that scales to high-dimensional datasets.

See the [Demo Notebook](demo.ipynb) for a demonstration of how to use these algorithms.

Outside of these reconstruction and query answering algorithms, this repo has the functionality to efficiently answer and manipulate large workloads of marginal and residual queries. 

## Setup 

``` bash
pip install git+https://github.com/bcmullins/efficient-marginal-reconstruction.git
```

<!-- ## Documentation

In this section, we provide a brief overview of the codebase.

### rem.algebra

This file contains classes to implicitly represent workloads of queries. Here is a summary of the workload classes:
- Workload: a general class for representing any workload that is structured as a Kronecker product. This is a parent for all other workload.
- MarginalWorkload: implicit represent for a single marginal workload.
- ResidualWorkload2: implicit representation for a single residual workload.
- ResidualWorkload2Pinv: implicit representation for the pseudoinverse of a single residual workload.

We also have classes to represent vertical and horizontal block matrices of workloads:
- VStack: a vertical stack of workloads.
- HStack: a horizontal stack of workloads.
- Block: a vertical stack of HStacks. -->