#!/usr/bin/env python3
import click
import pickle as pkl
import numpy as np
import numpy.random as ran
import cupy as cp
import xarray as xr
from util import PATHBASE, MONTHS, one_s, oversample


@click.command()
@click.argument("varname")
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
def main(varname, test, freq, ana):
    with open(f"{PATHBASE}/big{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)

    varstuff = metadata["variablemap"][varname]
    bigname = varstuff[1]
    h = varstuff[0][:2]
    comps = metadata["comps"][ana]
    notref = np.where(comps != "ref")[0]
    ref = np.where(comps == "ref")[0][0]
    n_sel = metadata["n_sel"]
    n_sam = metadata["n_sam"]
    n_mem = metadata["n_mem"]
    replace = metadata["replace"]
    bs = metadata["boundary_size"]
    crit_val = metadata["crit_val"]
    files_to_load = metadata["files_to_load"]

    glavgres = [] # Compute spatial averages on the fly for Christian's method

    for i, filename in enumerate(files_to_load):
        date = MONTHS[i]
        fname = f"{PATHBASE}/big{ana}/{varname}{date}.nc"
        darr = xr.open_dataset(fname)[bigname].squeeze()  # squeeze because ncecat leaves a dangling length-one axis when selecting a pressure, z or soil level in a bigger dataarray.
        darr = darr.coarsen(member=len(comps)).construct(member=("member", "comp"))
        darr = darr.transpose("comp", "time", ..., "member")
        darr = darr[:, :, bs:-bs, bs:-bs, :]
        darr = oversample(darr, freq)
        darrcp = cp.asarray(darr.values)
        results = np.empty((len(notref), *darr.shape[1:4], n_sel))
        results = xr.DataArray(
            results, 
            dims=["comp", *darr.dims[1:4], "sel"]
        )
        for s in range(n_sel):
            results[..., s] = one_s(darrcp, ref, notref, n_sam, replace, test, crit_val).get()
        glavgres.append(results.mean(dim=darr.dims[2:4]))
        results.to_netcdf(
            f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}_{date}.nc"
        )
    glavgres = xr.concat(glavgres, dim="newtime")
    glavgres.to_netcdf(
        f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}.nc"
    )

    
if __name__ == "__main__":
    main()
