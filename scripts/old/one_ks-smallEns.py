#!/usr/bin/env python3

from contextlib import redirect_stderr
import click
import gc
import pickle as pkl
import sys
import time as timer

import dateutil as du
import numpy as np
import numpy.random as ran
import cupy as cp
import pandas as pd
import xarray as xr

PATHBASE = "/scratch/snx3000/hbanderi/data"

def ks(a, b): # Kolmogorov-Smirnov metric using cupy
    x = cp.concatenate([a, b], axis=-1) # Concat all data
    nx = cp.sum(~cp.isnan(a), axis=-1) # Sum of nonnan instead of length to get actual number of samples, because time-oversampling may give you nans in the time axis (expected behaviour, see implementation)
    idxs_ks = cp.argsort(x, axis=-1)
    x = cp.take_along_axis(x, idxs_ks, axis=-1) # The x-axis of the ks plots, take_along_axis instead of sorting again. I need it to check for too close values, and problems with sp
    nx = nx[:, :, :, cp.newaxis] # Will need to be cast against a 4d array 
    # Cumulative distribution function using cumsum. The indices inferior to nx come from a the others come from b. 
    # Creates y1 (y2) the following way : for each other axis, iterates over the data in the member axis and adds 1/nx everytime it hits a value coming from a (b).
    y1 = cp.cumsum(idxs_ks < nx, axis=-1) / nx 
    y2 = cp.cumsum(idxs_ks >= nx, axis=-1) / nx
    # If the ks distance is found at 0 or 1, there probably is a rounding error problem. This could be made faster I think.
    invalid_idx = np.logical_or(np.isclose(x, 0), np.isclose(x, 1))
    ds = cp.abs(y1 - y2)
    ds[invalid_idx] = 0
    return cp.amax(ds, axis=-1)


def ttest(a, mub, stdb): # T-test metric
    mua = cp.nanmean(a, axis=-1)
    stda = cp.nanstd(a, axis=-1)
    std = stda**2 + stdb**2
    return cp.sqrt(a.shape[-1]) * (mua - mub) / cp.sqrt(std)


def mwu(a, b): # Mann-Whitney U metric
    ua = cp.zeros_like(a[:, :, :, 0])
    # Has to be a double loop otherwise the arrays would be too big to fit in the GPU. Anyways I'm looping over a small index (20 x 20 or 100 x 100)
    for i in range(b.shape[-1]):
        for j in range(a.shape[-1]):
            u += (b[..., j] > a[..., i]) + 0.5 * (b[..., j] == a[..., i])
    
    return np.amax([ua, a.shape[-1] ** 2 - ua], axis=0)
    
    
def one_s(darr, ref, notref, n_sam, replace, test, crit_val): # Performs one chunk worth of test.
    # Draw reference. Should redraw for every test maybe ? Shouldn't matter
    idxs_ref = ran.choice(darr.shape[-1], n_sam, replace=replace)
    b = darr[ref, ..., idxs_ref].transpose((1, 2, 3, 0))
    # Predefine ref to be filled for every test in notref
    rej = cp.empty((len(notref), *darr.shape[1:4]), dtype=bool)
    # Some test-specific definitions, a bit ugly
    if test == "KS":
        to_do = ks
        other_args = [b]
    elif test == "T":
        to_do = ttest
        mub = cp.nanmean(b, axis=-1)
        stdb = cp.nanstd(b, axis=-1)
        other_args = [mub, stdb]
    elif test == "MWU":
        to_do = mwu
        other_args = [b]
    else:
        print("Wrong test specifier")
        return -1 # replace with an exception
    for n in range(len(notref)):
        # Draw test
        idxs = ran.choice(darr.shape[-1], n_sam, replace=replace)
        a = darr[notref[n], ..., idxs].transpose((1, 2, 3, 0))
        # Do the do
        rej[n, ...] = to_do(a, *other_args) > crit_val[test]
    return rej


def oversample(darr, freq): # See thesis for explanation of why we would want to do this
    if freq != "1D" and freq != "12h": # Take care of the conditions here. Maybe refactor into 2 functions ?
        dims = list(darr.dims)
        # This is basically a fancy reshaping. (n_time, ..., n_mem) -> (n_time/freq, ..., n_mem * freq). freq is meant to be a period like 3 days, 1 week,... I know. I know....
        groups = darr.resample(time=freq).groups
        # Iterate over each groups, select their time values in the original DataArray, stack time and member axes into single new axis "memb" and storr in list
        subdarrs = [
            darr.isel(time=value).stack(memb=("time", "member")).reset_index("memb")
            for value in groups.values()
        ]
        # Some definitions for the creation of the new DataArray
        maxntime = np.amax([subdarr.shape[-1] for subdarr in subdarrs])
        newdims = dims.copy()
        newdims[1] = "newtime"
        newdims[-1] = "memb"
        # Creation of the new dataarray by concatenation, and padding if necessary (at the end of the time series for example) to ensure same shape
        newdarr = xr.concat(
            [subdarr.pad(memb=(0, maxntime - subdarr.shape[-1])) for subdarr in subdarrs],
            dim="newtime",
        ).transpose(*newdims)
        # newdarr should know its own resampling frequency
        newdarr.attrs["freq"] = freq
        # Reindex to be compliant with the tests
        newdarr = newdarr.reindex(
            {
                "newtime": pd.date_range(
                    start=newdarr["time"][0][0].values,
                    periods=newdarr.shape[1],
                    freq=freq,
                )
            }
        )
    else:
        # Rename to be compliant with the oversampled array, even if not oversampled. This should really go the other way be this is easier
        newdarr = darr.rename({"time": "newtime", "member": "memb"})
    return newdarr


@click.command()
@click.argument("varname")
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
def main(varname, test, freq, ana):
    with open(f'logs/{test}_{varname}_{freq}.python.out', 'w') as stderr, redirect_stderr(stderr):

        with open(f"{PATHBASE}/results/{ana}_{freq}/metadata.pickle", "rb") as handle:
            metadata = pkl.load(handle)

        variablemap = metadata["variablemap"]
        bigname = variablemap[varname][1]
        h = variablemap[varname][1][:2]
        comps = metadata["comps"][ana]
        notref = np.where(comps != "ref")[0]
        ref = np.where(comps == "ref")[0][0]
        n_sel = metadata["n_sel"]
        # n_sel = 2
        n_sam = metadata["n_sam"]
        # n_sam = 7
        nx = metadata["nx"]
        ny = metadata["ny"]
        n_mem = metadata["n_mem"]
        n_months = metadata["n_months"]
        replace = metadata["replace"]
        months_per_chunk = metadata["months_per_chunk"]
        tsta = metadata["tsta"]
        bs = metadata["boundary_size"]
        crit_val = metadata["crit_val"]

        n_chunks = int(n_months / months_per_chunk)
        file_letter = "s"  # 's' for semesterly, "m" for monthly
        glavgres = [] # Compute spatial averages on the fly for Christian's method

        for i in range(0, n_chunks):
            fname = f"{PATHBASE}/{ana}/{varname}{file_letter}{i}.nc"
            darr = xr.open_dataset(fname)[bigname].load()
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
                f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}_s{i}.nc"
            )
        glavgres = xr.concat(glavgres, dim="newtime")
        glavgres.to_netcdf(
            f"{PATHBASE}/results/{ana}_{freq}/{varname}_{test}.nc"
        )

    
if __name__ == "__main__":
    main()
