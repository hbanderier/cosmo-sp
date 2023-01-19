#!/usr/bin/env python3
import click
import pickle as pkl
import numpy as np
import cupy as cp
import xarray as xr
from util import PATHBASE, N_MONTHS, loaddarr, ks, ks_p


@click.command()
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
def main(ana):
    with open(f"{PATHBASE}/big{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)
        
    variablemap = metadata["variablemap"]
    ensembles = metadata["ensembles"][ana]
    bs = metadata["boundary_size"]

    alpha = 0.1
    decisions = np.empty(len(variablemap), dtype=list)
    for k, varname in enumerate(variablemap):
        decisions[k] = []
        bigname = variablemap[varname][1]
        for s in range(N_MONTHS):
            darr = loaddarr(varname, bigname, ensembles, s, ana, True, True, bs)
            ite = notref[1:]
            p = cp.empty((len(ite), *darr.shape[1:4]))
            b = cp.asarray(darr[2])
            for j, i in enumerate(ite):
                a = cp.asarray(darr[i])
                p[j] = ks(a, b)
            p = ks_p(p, darr.shape[-1])  # recycling yay
            p = cp.sort(p.reshape((p.shape[0], p.shape[1], p.shape[2] * p.shape[3])), axis=-1)
            p = p <= cp.arange(1, p.shape[-1] + 1) / p.shape[-1] * alpha
            decisions[k].append(cp.any(p, axis=2).get())
            cp.cuda.Device().synchronize()
        decisions[k] = np.concatenate(decisions[k], axis=1)
        with open(f"{PATHBASE}/results/fdr_decisions_{ana}_{varname}.pkl", "wb") as handle:
            pkl.dump(decisions[k], handle)
            
    with open(f"{PATHBASE}/results/fdr_decisions_{ana}.pkl", "wb") as handle:
        pkl.dump(decisions, handle)

    
if __name__ == "__main__":
    main()
