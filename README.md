# cosmo-sp

This repository contains the code used in the article "Reduced floating-point precision in regional climate simulations: An ensemble-based statistical verification" (Submitted).

The core of the testing methodology consists in multiple rounds of subsampling, grid-point-level testing using the Kolmogorov-Smirnov test, and averaging.
The most computationally expensive parts are coded using `CuPy` to work on GPUs, providing a major speedup.
The definitions of the core functions including the statistical tests are found in `scripts/utils.py`, while other python scripts there are meant to be run with `slurm` on a HPC cluster to perform the tests (`one_ks.py`) or other steps of the methodology (`concat.py`, `decisions.py`, `FDR.py`).
All the plots in the article and other side-analyses are done in the main jupyter notebook `cosmo-sp.ipynb`. 

Input parameters for the COSMO runs presented in the article can be found in the subfolder `COSMO_INPUTS`.
