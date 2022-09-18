#!/usr/bin/env python3

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
from joblib import Parallel, delayed


def ks(a, b):
    x = cp.concatenate([a, b], axis=-1)
    nx = cp.sum(~cp.isnan(a), axis=-1)
    idxs_ks = cp.argsort(x, axis=-1)
    x = cp.take_along_axis(x, idxs_ks, axis=-1)
    nx = nx[:, :, :, cp.newaxis]
    y1 = cp.cumsum(idxs_ks < nx, axis=-1) / nx
    y2 = cp.cumsum(idxs_ks >= nx, axis=-1) / nx
    invalid_idx = np.logical_or(np.isclose(x, 0), np.isclose(x, 1))
    ds = cp.abs(y1 - y2)
    ds[invalid_idx] = 0
    return cp.amax(ds, axis=-1)


def ttest(a, mub, stdb):
    mua = cp.nanmean(a, axis=-1)
    stda = cp.nanstd(a, axis=-1)
    std = stda**2 + stdb**2
    return cp.sqrt(a.shape[-1]) * (mua - mub) / cp.sqrt(std)


def mwu(a, b):
    ua = cp.zeros_like(a[:, :, :, 0])
    for i in range(b.shape[-1]):
        for j in range(a.shape[-1]):
            u += (b[..., j] > a[..., i]) + 0.5 * (b[..., j] == a[..., i])
    
    return np.amax([ua, a.shape[-1] ** 2 - ua], axis=0)
    
    
def one_s(darr, ref, notref, n_sam, replace, test, crit_val):
    idxs_ref = ran.choice(darr.shape[-1], n_sam, replace=replace)
    rej = cp.empty((len(notref), *darr.shape[1:4]), dtype=bool)
    b = darr[ref, ..., idxs_ref].transpose((1, 2, 3, 0))
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
        return -1
    for n in range(len(notref)):
        idxs = ran.choice(darr.shape[-1], n_sam, replace=replace)
        a = darr[notref[n], ..., idxs].transpose((1, 2, 3, 0))
        rej[n, ...] = to_do(a, *other_args) > crit_val[test]
    return rej


def oversample(darr, freq):
    if freq != "1D":
        dims = list(darr.dims)
        groups = darr.resample(time=freq).groups
        subdarrs = [
            darr.isel(time=value).stack(memb=("time", "member")).reset_index("memb")
            for value in groups.values()
        ]
        maxntime = np.amax([subdarr.shape[-1] for subdarr in subdarrs])
        newdims = dims.copy()
        newdims[1] = "newtime"
        newdims[-1] = "memb"
        newdarr = xr.concat(
            [subdarr.pad(memb=(0, maxntime - subdarr.shape[-1])) for subdarr in subdarrs],
            dim="newtime",
        ).transpose(*newdims)
        newdarr.attrs["freq"] = freq
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
        newdarr = darr.rename({"time": "newtime", "member": "memb"})
    return newdarr


@click.command()
@click.argument("varname")
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "3D", "5D", "1w", "2w", "1M", "3M"], case_sensitive=True), default="1D", help="Resampling frequency")
def main(varname, test, freq):
    sys.stdout = open(f'.out/{varname}_{freq}_python.out', 'w')
    sys.stderr = sys.stdout
    datapath = "/scratch/snx3000/hbanderi/data"
    
    with open(f"{datapath}/results/{freq}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)

    variablemap = metadata["variablemap"]
    bigname = variablemap[varname][1]
    h = variablemap[varname][1][:2]
    comps = metadata["comps"]
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
    file_letter = "s"  # 's' for semester
    glavgres = []
    
    if freq=="1D" and h=="12":
        freq="12h"

    for i in range(0, n_chunks):
        fname = f"{datapath}/main/{varname}s{i}.nc"
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
            f"{datapath}/results/{freq}/{varname}_{test}_s{i}.nc"
        )
    glavgres = xr.concat(glavgres, dim="newtime")
    glavgres.to_netcdf(
        f"{datapath}/results/{freq}/{varname}_{test}.nc"
    )

    
if __name__ == "__main__":
    main()
