#!/usr/bin/env python3
import click
import pickle as pkl
import numpy as np
import cupy as cp
import xarray as xr
from util import PATHBASE, MONTHS, cupy_decisions, open_results, coords_decisions, coords_avgdecs


@click.command()
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
def main(test, freq, ana):
    with open(f"{PATHBASE}/big{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)

    variablemap = metadata["variablemap"]
    control= metadata["control"][ana]
    notcontrol = metadata["notcontrol"][ana]
    decision_quantile = metadata["decision_quantile"]
    ensembles_in_decisions = metadata["ensembles_in_decisions"][ana]
    bs = metadata["boundary_size"]
    
    for i, varname in enumerate(variablemap):
        actualfreq = "12h" if (freq == "1D" and variablemap[varname][1][:2] == "12") else freq
        coords1 = coords_decisions(varname, ana, actualfreq, ensembles_in_decisions, bs)
        coords2 = coords_avgdecs(varname, ana, actualfreq, ensembles_in_decisions)
        shape1 = [len(x) for x in coords1.values()]
        shape2 = [len(x) for x in coords2.values()]
        decisions = xr.DataArray(np.empty(shape1, dtype=bool), coords=coords1)
        avgdecs = xr.DataArray(np.empty(shape2, dtype=bool), coords=coords2)
        j = 0
        for k, month in enumerate(MONTHS):
            results = open_results(varname, ana, freq, test, k)
            a, b = cupy_decisions(results.values, decision_quantile, control, notcontrol)
            l = j + a.shape[1]
            decisions[:, j:l, ...] = a.get()
            avgdecs[:, j:l] = b.get()
            j = l
        decisions.to_netcdf(f"{PATHBASE}/results/{ana}_{freq}/decisions_{test}_{varname}.nc")
        avgdecs.to_netcdf(f"{PATHBASE}/results/{ana}_{freq}/avgdecs_{test}_{varname}.nc")
            
            
if __name__ == "__main__":
    main()
