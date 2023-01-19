import numpy as np
import numpy.random as ran
import xarray as xr
import cupy as cp
import pandas as pd
import pickle as pkl
from collections.abc import Iterable

PATHBASE = "/scratch/snx3000/hbanderi/data"
TSTA = "19890101"
N_MONTHS = 120
MONTHS = pd.date_range(TSTA, periods=N_MONTHS, freq="1MS").strftime("%Y%m")


def loaddarr(varname, bigname, ensembles, k, ana="main", big=True, values=True, bs=None):
    if big:
        suffix = MONTHS
        anaprefix = "big"
    else:
        suffix = [f"s{k}" for k in range(20)]
        anaprefix = ""
    basename = f"{PATHBASE}/{anaprefix}{ana}/{varname}"
    if isinstance(k, int) or isinstance(k, str):
        darr = xr.open_dataset(f"{basename}{suffix[k]}.nc")[bigname].squeeze()  # squeeze because ncecat may leave a dangling length-one dimension when selecting a p / z / soil level
    elif isinstance(k, Iterable):
        fnames = [f"{basename}{suffix[j]}.nc" for j in k]
        darr = xr.open_mfdataset(fnames)[bigname].squeeze()
    if big:
        darr = darr.coarsen(member=len(ensembles)).construct(member=("member", "ensemble"))
        darr = darr.transpose("ensemble", "time", ..., "member")
    if values:
        darr = darr.values
    else:
        darr = darr.assign_coords({"ensemble": ensembles})  # might be useful
    if bs is not None:
        return darr[:, :, bs:-bs, bs:-bs, :]
    return darr


def month_range(month, freq):
    return pd.date_range(pd.to_datetime(month, format="%Y%m"), pd.to_datetime(month, format="%Y%m") + pd.DateOffset(months=1), freq=freq, inclusive="left") 
    
    
def full_range(freq):
    return pd.date_range(pd.to_datetime(MONTHS[0], format="%Y%m"), pd.to_datetime(MONTHS[-1], format="%Y%m") + pd.DateOffset(months=1), freq=freq, inclusive="left") 


def get_grid(varname, bs=None):
    thislatdim = "srlat" if varname in ["v_200hPa", "v_100m"] else "rlat"
    thislatcoord = xr.open_dataarray(f"{PATHBASE}/gridinfo/{thislatdim}.nc")
    thislondim = "srlon" if varname in ["u_200hPa", "u_100m"] else "rlon"
    thisloncoord = xr.open_dataarray(f"{PATHBASE}/gridinfo/{thislondim}.nc")
    if bs is not None:
        return thislatdim, thislatcoord[bs:-bs], thislondim, thisloncoord[bs:-bs]
    return thislatdim, thislatcoord, thislondim, thisloncoord
    

def open_results(varname, ana, freq, ensembles_in_results, bs, k):
    results = xr.open_dataarray(f"{PATHBASE}/results/{ana}_{freq}/{varname}_KS_{MONTHS[k]}.nc", engine="h5netcdf")
    try:
        results = results.rename({"comp": "ensemble", "newtime": "time"})
    except ValueError:
        pass
    else:
        thislatdim, thislatcoord, thislondim, thisloncoord = get_grid(varname, bs)
        if freq == "1D" and results.shape[1] > 31:  # check if 12h
            freq = "12h"
        results = results.assign_coords({
            "ensemble" : ensembles_in_results, 
            "time": month_range(MONTHS[k], freq), 
            thislatdim: thislatcoord,
            thislondim: thisloncoord,
            "sel": np.arange(results.shape[-1]),
        })
    return results


def open_decisions_pickle(varname, ana, freq, ensembles_in_results, bs):

    with open(f"{PATHBASE}/results/{ana}_{freq}/decisions_{varname}.pkl", "rb") as handle:
        decisions = pkl.load(handle)
    thislatdim, thislatcoord, thislondim, thisloncoord = get_grid(varname, bs)
    if freq == "1D" and decisions.shape[1] > 366 * 10:  # check if 12h
        freq = "12h"
    decisions = xr.DataArray(
        decisions, 
        coords={
            "ensemble" : [ens for ens in ensembles_in_results if ens != "control"], 
            "time": full_range(freq), 
            thislatdim: thislatcoord,
            thislondim: thisloncoord,
    })
    return decisions

def open_avgdecs_pickle(varname, ana, freq, ensembles_in_results):
    with open(f"{PATHBASE}/results/{ana}_{freq}/avgdecs_{varname}.pkl", "rb") as handle:
        avgdecs = pkl.load(handle)
    if freq == "1D" and avgdecs.shape[1] > 366 * 10:  # check if 12h
        freq = "12h"
    avgdecs = xr.DataArray(
        avgdecs, 
        coords={
            "comp" : [ens for ens in ensembles_in_results if ens != "control"], 
            "time": full_range(freq), 
    })
    return avgdecs


def ks(a, b):
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
    
    return cp.amax([ua, a.shape[-1] ** 2 - ua], axis=0)


def ks_p(d, n): # p-values of the KS test, from the distance (output of ks(a, b))
    return cp.exp(- d ** 2 * n)


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
    if freq in ["12h", "1D"]: # Those mean no resampling
        return darr
    dims = list(darr.dims)
    # This is basically a fancy reshaping. (n_time, ..., n_mem) -> (n_time/freq, ..., n_mem * freq). freq is meant to be a period like 3 days, 1 week,... I know. I know....
    groups = darr.resample(time=freq).groups
    # Iterate over each groups, select their time values in the original DataArray, stack time and member axes into single new axis "memb" and storr in list
    subdarrs = [
        darr.isel(time=value).stack(memb=("time", "member")).reset_index("memb", drop=True).rename({"memb": "member"})
        for value in groups.values()
    ]
    # Some definitions for the creation of the new DataArray
    maxntime = np.amax([subdarr.shape[-1] for subdarr in subdarrs])
    newdims = dims.copy()
    # Creation of the new dataarray by concatenation, and padding if necessary (at the end of the time series for example) to ensure same shape
    newdarr = xr.concat(
        [subdarr.pad(member=(0, maxntime - subdarr.shape[-1])) for subdarr in subdarrs],
        dim="time",
    ).transpose(*newdims)
    # newdarr should know its own resampling frequency
    newdarr.attrs["freq"] = freq
    # Reindex to be compliant with the tests
    return newdarr.reindex(
        {
            "time": pd.date_range(
                start=newdarr.time.values[0],
                periods=newdarr.shape[1],
                freq=freq,
            )
        }
    )


def cupy_decisions(results, quantile, control, notcontrol):
    n = len(notcontrol)
    results = cp.asarray(results)
    avgres = cp.mean(results, axis=(2, 3))
    decision = cp.empty((n, *results.shape[1:4]), dtype=bool)
    avgdec = cp.empty((n, results.shape[1]), dtype=bool)
    for i, j in enumerate(notcontrol):
        decision[i, ...] = cp.mean(results[j], axis=-1) > cp.quantile(results[control], quantile, axis=-1)
        avgdec[i, ...] = cp.mean(avgres[j], axis=-1) > cp.quantile(avgres[control], quantile, axis=-1)
    return decision, avgdec