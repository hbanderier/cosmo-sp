#!/usr/bin/env python3
import click
import pickle as pkl
import numpy as np
import cupy as cp
import xarray as xr
from util import PATHBASE, N_MONTHS, loaddarr, coords_avgdecs, ks, ks_p


@click.command()
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
def main(ana, freq):
    with open(f"{PATHBASE}/big{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)
        
    variablemap = metadata["variablemap"]
    decision_quantile = metadata["decision_quantile"]
    ensembles = metadata["ensembles"][ana]
    ensembles_in_decisions = metadata["ensembles_in_decisions"][ana]
    bs = metadata["boundary_size"]
    test = "KS"

    alpha = 0.1
    for varname in variablemap:
        actualfreq = "12h" if (freq == "1D" and variablemap[varname][0][:2] == "12") else freq
        coords = coords_avgdecs(varname, ana, actualfreq, ensembles_in_decisions)
        shape = [len(x) for x in coords.values()]
        decisions = xr.DataArray(np.zeros(shape), coords=coords)
        bigname = variablemap[varname][1]
        j = 0
        for k in range(N_MONTHS):
            darr = loaddarr(varname, bigname, ensembles, k, ana, big=True, values=False, bs=bs)
            p = cp.zeros((len(ensembles_in_decisions), *darr.shape[1:4]))
            b = cp.asarray(darr.sel(ensemble="ref").values)
            for i, ens in enumerate(ensembles_in_decisions):
                a = cp.asarray(darr.sel(ensemble=ens))
                p[i] = ks(a, b)
            p = ks_p(p, darr.shape[-1])  # recycling yay
            p = cp.sort(p.reshape((p.shape[0], p.shape[1], p.shape[2] * p.shape[3])), axis=-1)
            p = p <= cp.arange(1, p.shape[-1] + 1) / p.shape[-1] * alpha
            l = j + darr.shape[1]
            decisions[:, j:l] = ((p.shape[-1] - cp.argmax(p[:, :, ::-1], axis=-1)) / p.shape[-1]).get()
            j = l
            cp.cuda.Device().synchronize()
        decisions.to_netcdf(f"{PATHBASE}/results/{ana}_{freq}/FDR_decisions_{test}_{varname}.nc")


    
if __name__ == "__main__":
    main()
