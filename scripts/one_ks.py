#!/usr/bin/env python3

import argparse
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


def ks(x, n_sam, c_alpha):
    idxs = np.argsort(x, axis=-1)
    nx = (np.sum(~np.isnan(x), axis=-1) / 2).astype(int)
    x = np.take_along_axis(x, idxs, axis=-1)
    exact = np.isclose(x[..., 0], x[..., -1], rtol=1e-3)
    y1 = np.cumsum(idxs < n_sam, axis=-1) / nx[:, :, :, np.newaxis]
    y2 = np.cumsum(idxs >= n_sam, axis=-1) / nx[:, :, :, np.newaxis]
    d = np.amax(np.abs(y1 - y2), axis=-1)

    rej = d > c_alpha * np.sqrt(2 / nx[:, :, :])
    rej[exact] = False
    return rej


def one_s(darr, n_sam, ref, notref, nbins, replace, c_alpha):
    dims = list(darr.dims)
    idxs_ref = ran.choice(darr.shape[-1], n_sam, replace=replace)
    b = darr[ref, :, :, :, idxs_ref].values
    rej = xr.DataArray(
        np.empty(darr.shape[1:4]),
        coords={
            dims[i]: darr[dims[i]] for i in range(1, 4)
        }
    )
    avgrej = []
    results = []
    for nr in range(len(notref)):
        idxs = ran.choice(darr.shape[-1], n_sam, replace=replace)
        a = darr[notref[nr], :, :, :, idxs].values
        rej.loc[:, :, :] = ks(
            np.concatenate([a, b], axis=-1),
            n_sam,
            c_alpha,
        )
        del a 
        gc.collect()
        avgrej.append(rej.resample(newtime="1M").mean())
        results.append(rej.mean(dim=[dims[2], dims[3]]))
    del rej, b
    gc.collect()
    return xr.concat(avgrej, dim="comp"), xr.concat(results, dim="comp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("varname")
    parser.add_argument("at_once")
    parser.add_argument("freq")
    
    varname = parser.parse_args().varname
    at_once = int(parser.parse_args().at_once)
    freq = parser.parse_args().freq
    
    sys.stdout = open(f'.out/{varname}_{freq}_python.out', 'w')
    sys.stderr = sys.stdout
    
    with open(f"../../hbanderi/data/results/{freq}/metadata.pickle", "rb") as handle:
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
    n_months = metadata["n_months"]
    replace = metadata["replace"]
    months_per_chunk = metadata["months_per_chunk"]
    tsta = metadata["tsta"]

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
            f"../../hbanderi/data/main/{varname}s{j}.nc" 
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
                    darr, n_sam, ref, notref, 
                    nbins, replace, c_alpha
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
    results.to_netcdf(f"../../hbanderi/data/results/{freq}/{varname}.nc")
    avgrejection.to_netcdf(f"../../hbanderi/data/rejection/{freq}/{varname}_avg.nc")
    del results, avgrejection
    gc.collect()
