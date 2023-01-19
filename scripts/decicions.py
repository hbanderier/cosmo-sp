#!/usr/bin/env python3
import click
import pickle as pkl
import numpy as np
import cupy as cp
import xarray as xr
from util import PATHBASE, MONTHS, cupy_decisions


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

    for i, varname in enumerate(variablemap):
        decisions = []
        avgdecs = []
        for j, month in enumerate(MONTHS):
            path = f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}_{month}.nc"
            results = xr.open_dataarray(path).values
            a, b = cupy_decisions(results, decision_quantile, control, notcontrol)
            decisions.append(a.get())
            avgdecs.append(b.get())
        decisions = np.concatenate(decisions, axis=1)
        avgdecs = np.concatenate(avgdecs, axis=1)
        with open(f"{PATHBASE}/results/{ana}_{freq}/decisions_{varname}.pkl", "wb") as handle:
            pkl.dump(decisions, handle)
        with open(f"{PATHBASE}/results/{ana}_{freq}/avgdecs_{varname}.pkl", "wb") as handle:
            pkl.dump(avgdecs, handle)
            
            
if __name__ == "__main__":
    main()
