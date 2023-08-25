#!/usr/bin/env python3
import click
import os
import pickle as pkl
import numpy as np
import cupy as cp
import xarray as xr
from util import PATHBASE, N_MONTHS, loaddarr, coords_avgdecs, sanitize, wraptest, p_wrapper


@click.command()
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), default="main", help="Which of the two analyses to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
def main(ana, freq, test):
    with open(f"{PATHBASE}/{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)
        
    variablemap = metadata["variablemap"]
    decision_quantile = metadata["decision_quantile"]
    ensembles = metadata["ensembles"][ana]
    ensembles_in_decisions = metadata["ensembles_in_decisions"][ana]
    bs = metadata["boundary_size"]
    rounding = 4

    alpha = 0.1
    
    subset = [
        't_850hPa',
        'u_200hPa',
        'qv_500hPa',
        'fi_500hPa',
        'tot_prec',
        'w_snow',
        'w_so_third',
        'ashfl_s',
        'clct',
        'ps'
    ]
    for varname in variablemap:
        ofile = f"{PATHBASE}/results/{ana}_{freq}/FDR_decisions_{test}_{varname}.nc"
        if os.path.isfile(ofile):
            continue
        actualfreq = "12h" if (freq == "1D" and variablemap[varname][0][:2] == "12") else freq
        coords = coords_avgdecs(varname, ana, actualfreq, ensembles_in_decisions)
        shape = [len(x) for x in coords.values()]
        decisions = xr.DataArray(np.zeros(shape), coords=coords)
        bigname = variablemap[varname][1]
        j = 0
        for k in range(N_MONTHS):
            darr = loaddarr(varname, bigname, ensembles, k, ana, values=False, bs=bs)
            p = cp.zeros((len(ensembles_in_decisions), *darr.shape[1:4]))
            b = cp.asarray(darr.sel(ensemble="ref").values)
            b = sanitize(b, rounding=rounding)
            to_do, other_args = wraptest(b, test)
            for i, ens in enumerate(ensembles_in_decisions):
                a = cp.asarray(darr.sel(ensemble=ens))
                a = sanitize(a, rounding=rounding)
                p[i] = to_do(a, *other_args)
                p[i] = p_wrapper(test, p[i], darr.shape[-1])
            p = cp.sort(p.reshape((p.shape[0], p.shape[1], p.shape[2] * p.shape[3])), axis=-1)
            l = j + darr.shape[1]
            decisions[:, j:l] = cp.any(
                p <= cp.arange(1, p.shape[-1] + 1) / p.shape[-1] * alpha, axis=-1
            ).get()
            j = l
            cp.cuda.Device().synchronize()
        decisions.to_netcdf(ofile)

    
if __name__ == "__main__":
    main()
