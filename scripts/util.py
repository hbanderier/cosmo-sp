import numpy as np
import numpy.random as ran
import xarray as xr
import cupy as cp
import pandas as pd
import dateutil.parser as ps
import dateutil.relativedelta as rd
from collections.abc import Iterable

PATHBASE = "/scratch/snx3000/hbanderi/data"
TSTA = "19890101"
N_MONTHS = 120
MONTHS = [
    (ps.parse(TSTA) + rd.relativedelta(months=x)).strftime("%Y%m")
    for x in range(N_MONTHS)
]


def loaddarr(varname, bigname, comps, k, ana="main", big=True, values=True, bs=None):
    if big:
        suffix = MONTHS
        anaprefix = "big"
    else:
        suffix = [f"s{k}" for k in range(20)]
        anaprefix = ""
    basename = f"{PATHBASE}/{anaprefix}{ana}/{varname}"
    if isinstance(k, int) or isinstance(k, str):
        darr = xr.open_dataset(f"{basename}{suffix[k]}.nc")[bigname].squeeze()
    elif isinstance(k, Iterable):
        fnames = [f"{basename}{suffix[j]}.nc" for j in k]
        darr = xr.open_mfdataset(fnames)[bigname].squeeze()
    if big:
        darr = darr.coarsen(member=len(comps)).construct(member=("member", "comp"))
        darr = darr.transpose("comp", "time", ..., "member")
    if values:
        darr = darr.values
    if bs is not None:
        return darr[:, :, bs:-bs, bs:-bs, :]
    return darr



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