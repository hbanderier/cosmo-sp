#!/usr/bin/env python3
import click
import os
import pickle as pkl
import numpy as np
import numpy.random as ran
import cupy as cp
import xarray as xr
from util import PATHBASE, MONTHS, loaddarr, one_s, oversample, coords_results


@click.command()
@click.argument("varname")
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
@click.option("--ana", type=click.Choice(["main", "sensi", "comb"], case_sensitive=True), help="Which of the two analyses to perform")
def main(varname, test, freq, ana):
    with open(f"{PATHBASE}/{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)

    vmap = metadata["variablemap"][varname]
    bigname = vmap[1]
    h = vmap[0][:2]
    ensembles = metadata["ensembles"][ana]
    ensembles_in_results = metadata["ensembles_in_results"][ana]
    ref = metadata["ref"][ana]
    notref = metadata["notref"][ana]
    n_sel = metadata["n_sel"]
    n_sam = metadata["n_sam"][freq]
    n_mem = metadata["n_mem"]
    replace = metadata["replace"]
    bs = metadata["boundary_size"]
    crit_val = metadata["crit_val"][freq]
    rounding = metadata["rounding"]
    
    glavgres = [] # Compute spatial averages on the fly for Christian's method
    
    for i, date in enumerate(MONTHS):
        ofile = f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}_{date}.nc"
        if os.path.isfile(ofile):
            continue
        darr = loaddarr(varname, bigname, ensembles, i, ana, False, bs)
        darr = oversample(darr, freq)
        darrcp = cp.asarray(darr.values)
        results = np.empty((len(notref), *darr.shape[1:4], n_sel))
        results = xr.DataArray(
            results, 
            coords=coords_results(varname, ana, freq, ensembles_in_results, bs, i, results.shape)[0]
        )
        for s in range(n_sel):
            results[..., s] = one_s(darrcp, ref, notref, n_sam, replace, test, crit_val, rounding=rounding).get()
        glavgres = results.mean(dim=darr.dims[2:4])
        results.to_netcdf(ofile)
        glavgres.to_netcdf(f"{PATHBASE}/results/{ana}_{freq}/avg_{varname}_{test}_{date}.nc")
    
if __name__ == "__main__":
    main()
