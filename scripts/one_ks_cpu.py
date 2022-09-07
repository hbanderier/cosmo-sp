#!/usr/bin/env python3

import click
import gc
import pickle as pkl
import sys
import time as timer

import dateutil as du
import numpy as np
import numpy.random as ran
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


def ttest(a, b, t_crit, rtol=5e-4, atol=1e-8):
    mub = np.mean(b, axis=-1)
    stdb = np.std(b, axis=-1)
    mua = np.mean(a, axis=-1)
    stda = np.std(a, axis=-1)
    ss = stda ** 2 + stdb ** 2
    t = np.sqrt(a.shape[-1]) * (mua - mub) / np.sqrt(stda ** 2 + stdb ** 2)
    rej = t > t_crit
    rej[np.isclose(ss, 0, atol=atol)] = False
    return rej


def ks(x, c_alpha, rtol=5e-4, atol=1e-8):
    idxs = np.argsort(x, axis=-1)
    nx = (np.sum(~np.isnan(x), axis=-1) / 2).astype(int)
    x = np.take_along_axis(x, idxs, axis=-1)
    exact = np.isclose(x[..., 0], x[..., -1], rtol=rtol, atol=atol)
    nx = nx[:, :, :, np.newaxis]
    y1 = np.cumsum(idxs < nx, axis=-1) / nx
    y2 = np.cumsum(idxs >= nx, axis=-1) / nx
    d = np.amax(np.abs(y1 - y2), axis=-1)

    rej = d > c_alpha * np.sqrt(2 / nx[:, :, :, 0])
    rej[exact] = False
    return rej


def mwu(a, b, u_crit, rtol=5e-4, atol=1e-8):
    ua = np.zeros_like(a[:, :, :, 0])
    ub = np.zeros_like(b[:, :, :, 0])
    for i in range(b.shape[-1]):
        for j in range(a.shape[-1]):
            close = np.isclose(b[..., i], a[..., i], rtol=rtol, atol=atol)
            ua += ((b[..., j] > a[..., i]) & ~close) + 0.5 * close
            ub += ((a[..., i] > b[..., j]) & ~close) + 0.5 * close
    
    return np.amin([ua, ub], axis=0) < u_crit


def one_s(darr, test, n_sam, ref, notref, nbins, replace, c_alpha, u_crit, t_crit, rtol=5e-4, atol=1e-8):
    dims = list(darr.dims)
    idxs_ref = ran.choice(darr.shape[-1], n_sam, replace=replace)
    b = darr[ref, :, :, :, idxs_ref].values
    rej = xr.DataArray(
        np.empty(darr.shape[1:4], dtype=bool),
        coords={
            dims[i]: darr[dims[i]] for i in range(1, 4)
        }
    )
    avgrej = []
    results = []
    for nr in range(len(notref)):
        idxs = ran.choice(darr.shape[-1], n_sam, replace=replace)
        a = darr[notref[nr], :, :, :, idxs].values
        if test == "KS":
            rej.loc[:, :, :] = ks(
                np.concatenate([a, b], axis=-1),
                n_sam, c_alpha, rtol, atol
            )
        if test == "MWU":
            rej.loc[:, :, :] = mwu(a, b, u_crit, rtol, atol)   
        if test == "T":
            rej.loc[:, :, :] = ttest(a, b, t_crit, rtol, atol)
        del a 
        gc.collect()
        avgrej.append(rej.resample(newtime="1M").mean())
        results.append(rej.mean(dim=[dims[2], dims[3]]))
    del rej, b
    gc.collect()
    return xr.concat(avgrej, dim="comp"), xr.concat(results, dim="comp")


@click.command()
@click.argument("varname")
@click.option("--at_once", type=int, default=1, help="How many semesters to load in memory at once")
@click.option("--test", type=click.Choice(["MWU", "KS", "T"], case_sensitive=True), default="KS", help="Which Statistical test to perform")
@click.option("--freq", type=click.Choice(["1D", "2D", "1w", "2w", "1M"], case_sensitive=True), default="1D", help="Resampling frequency")
def main(varname, test, at_once, freq):
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
    nbins = metadata["nbins"]
    c_alpha = metadata["c_alpha"]
    u_crit = metadata["u_crit"]
    t_crit = metadata["t_crit"]
    n_months = metadata["n_months"]
    replace = metadata["replace"]
    months_per_chunk = metadata["months_per_chunk"]
    tsta = metadata["tsta"]
    rtol = 5e-4
    atol = 1 if varname == "ashfl_s" else 1e-3

    n_chunks = int(n_months / months_per_chunk)
    file_letter = "s"  # 's' for semester
    
    rejcont = None
    avgrejection = []
    results = []

    n_proc = 20
    # n_proc = 1
    if freq=="1D" and h=="12":
        freq="12h"

    for i in range(0, n_chunks, at_once):
        fnames = [
            f"{datapath}/main/{varname}s{j}.nc" 
            for j in range(i, i + at_once)
        ]
        darr = xr.open_mfdataset(fnames)[bigname].load()
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
        darr = xr.concat(
            [
                subdarr.pad(memb=(0, maxntime - subdarr.shape[-1]))
                for subdarr in subdarrs
            ],
            dim="newtime",
        ).transpose(*newdims)
        darr.attrs["freq"] = freq
        darr = darr.reindex(
            {
                "newtime": pd.date_range(
                    start=darr["time"][0][0].values,
                    periods=darr.shape[1],
                    freq=freq,
                )
            }
        )
        
        theseavgrej, theseres = zip(
            *Parallel(
                n_jobs=n_proc,
                verbose=52,
                max_nbytes=1e5,
                batch_size="auto",
                # backend="multiprocessing"
                prefer="threads"
            )(
                delayed(one_s)(
                    darr, test, n_sam, ref, notref, 
                    nbins, replace, c_alpha, u_crit, t_crit, rtol, atol
                )
                for s in range(n_sel)
            )
        )
        avgrejection.append(xr.concat(theseavgrej, dim="sel"))
        results.append(xr.concat(theseres, dim="sel"))
        del darr, theseavgrej, theseres
        gc.collect()
    results = xr.concat(results, dim="newtime")
    avgrejection = xr.concat(avgrejection, dim="newtime")
    results.to_netcdf(f"{datapath}/results/{freq}/{varname}_{test}.nc")
    avgrejection.to_netcdf(f"{datapath}/rejection/{freq}/{varname}_{test}_avg.nc")
    del results, avgrejection
    gc.collect()

    
if __name__ == "__main__":
    main()