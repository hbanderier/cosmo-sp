import os
import numpy as np
import numpy.random as ran
import xarray as xr
import cupy as cp
import pandas as pd
import pickle as pkl
import cartopy as ctp
from cupyx.scipy.special import erf as cupy_erf
from collections.abc import Iterable
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_array(z):  # if we use cupy
    try:
        return z.get()
    except AttributeError:
        return z


SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rcParams["animation.html"] = "jshtml"

CMAP = LinearSegmentedColormap.from_list(
    "gr",
    [
        [21 / 256, 176 / 256, 26 / 256, 1],
        [1, 1, 1, 1],
        [229 / 256, 0, 0, 1],
    ],
    N=50,
)
CMAP_HEX = [rgb2hex(c) for c in CMAP(np.linspace(0, 1, 50))]
HERE = "__xarray_dataarray_variable__"
TEXTWIDTH_IN = 0.0138889 * 503.61377

COLORS = [  # https://coolors.co/palette/ef476f-ffd166-06d6a0-118ab2-073b4c
    "#ef476f",  # pinky red
    "#ffd166",  # yellow
    "#06d6a0",  # cyany green
    "#118ab2",  # light blue
    "#073b4c",  # dark blue
]

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


def get_grid(varname, bs=None, full=False):
    rlonname = 'rlon'
    lonname = 'lon'
    rlatname = 'rlat'
    latname = 'lat'
    if varname in ['u_200hPa', 'u_100m']:
        rlonname = 'srlon'
        latname = 'slatu'
        lonname = 'slonu'
    elif varname in ['v_200hPa', 'v_100m']:
        rlatname = 'srlat'
        latname = 'slatv'
        lonname = 'slonv'
    rlon = xr.open_dataarray(f'{PATHBASE}/gridinfo/{rlonname}.nc')
    rlat = xr.open_dataarray(f'{PATHBASE}/gridinfo/{rlatname}.nc')
    if not full:
        if bs is not None:
            return {rlatname: rlat[bs:-bs], rlonname: rlon[bs:-bs]}
        return {rlatname: rlat, rlonname: rlon}
    lon = xr.open_dataarray(f'{PATHBASE}/gridinfo/{lonname}.nc').reset_coords([lonname, latname], drop=True)
    lat = xr.open_dataarray(f'{PATHBASE}/gridinfo/{latname}.nc').reset_coords([lonname, latname], drop=True)
    if bs is not None:
        return {rlatname: rlat[bs:-bs], rlonname: rlon[bs:-bs]}, {latname: lat[bs:-bs, bs:-bs], lonname: lon[bs:-bs, bs:-bs]}
    return {rlatname: rlat, rlonname: rlon}, {latname: lat, lonname: lon}


def coords_results(varname, ana, freq, ensembles_in_results, bs, k, shape):
    dims, coords = get_grid(varname, bs, full=True)
    if freq == '1D' and shape[1] > 31:  # check if 12h
        freq = '12h'
    fulldims = dict(
        ensemble=ensembles_in_results, 
        time=month_range(MONTHS[k], freq), 
        **dims,
        sel=np.arange(shape[-1])
    )
    return fulldims, coords

def open_results_old(varname, ana, freq, test, ensembles_in_results, bs, k):
    results = xr.open_dataarray(f'{PATHBASE}/oldresults/{ana}_{freq}/{varname}_{test}_{MONTHS[k]}.nc', engine='h5netcdf')
    try:
        results = results.rename({'comp': 'ensemble', 'newtime': 'time'})
    except ValueError:
        pass
    else:
        dims, coords = coords_results(varname, ana, freq, ensembles_in_results, bs, k, results.shape)
        results = results.assign_coords(dims).assign_coords(coords)
    return results


def open_results(varname, ana, freq, test, k):
    results = xr.open_dataarray(f'{PATHBASE}/results/{ana}_{freq}/{varname}_{test}_{MONTHS[k]}.nc', engine='h5netcdf')
    return results


def coords_decisions(varname, ana, freq, ensembles_in_decisions, bs, shape=None):
    dims, coords = get_grid(varname, bs, full=True)
    if shape is not None and (freq == '1D' and shape[1] > 366 * 10):  # check if 12h
        freq = '12h'
    fulldims = dict(
        ensemble=ensembles_in_decisions, 
        time=full_range(freq), 
        **dims,
    )
    return fulldims, coords


def open_decisions_pickle(varname, ana, freq, ensembles_in_decisions, bs):

    with open(f'{PATHBASE}/oldresults/{ana}_{freq}/decisions_{varname}.pkl', 'rb') as handle:
        decisions = pkl.load(handle)
    dims, coords = coords_decisions(varname, ana, freq, ensembles_in_decisions, bs, decisions.shape)
    decisions = xr.DataArray(
        decisions, 
        coords=dims
    ).assign_coords(coords)
    return decisions


def open_decisions(varname, ana, freq, test):
    decisions = xr.open_dataarray(f'{PATHBASE}/results/{ana}_{freq}/decisions_{test}_{varname}.nc')
    return decisions


def coords_avgdecs(varname, ana, freq, ensembles_in_decisions, shape=None):
    if shape is not None and (freq == '1D' and shape[1] > 366 * 10):  # check if 12h
        freq = '12h'
    return {
        'ensemble' : ensembles_in_decisions, 
        'time': full_range(freq), 
    }


def open_avgdecs_pickle(varname, ana, freq, ensembles_in_decisions):
    with open(f"{PATHBASE}/oldresults/{ana}_{freq}/avgdecs_{varname}.pkl", "rb") as handle:
        avgdecs = pkl.load(handle)
    avgdecs = xr.DataArray(
        avgdecs, 
        coords=coords_avgdecs(varname, ana, freq, ensembles_in_decisions, avgdecs.shape))
    return avgdecs


def open_avgdecs(varname, ana, freq, test):
    avgdecs = xr.open_dataarray(f"{PATHBASE}/results/{ana}_{freq}/avgdecs_{test}_{varname}.nc")
    return avgdecs


def ks_cumsum(a, b):  # old, faster but slightly wrong version of the ks test : does not solve the tie problem satisfactorily
    x = cp.concatenate([a, b], axis=-1) # Concat all data
    nx = cp.sum(~cp.isnan(a), axis=-1) # Sum of nonnan instead of length to get actual number of samples, because time-oversampling may give you nans in the time axis (expected behaviour, see implementation)
    idxs_ks = cp.argsort(x, axis=-1)
    x = cp.take_along_axis(x, idxs_ks, axis=-1) # The x-axis of the ks plots, take_along_axis instead of sorting again. I need it to check for too close values, and problems with sp
    nx = nx[..., cp.newaxis] # Will need to be cast against a 4d array 
    # Cumulative distribution function using cumsum. The indices inferior to nx come from a the others come from b. 
    # Creates y1 (y2) the following way : for each other axis, iterates over the data in the member axis and adds 1/nx everytime it hits a value coming from a (b).
    y1 = cp.cumsum(idxs_ks < nx, axis=-1) / nx 
    y2 = cp.cumsum(idxs_ks >= nx, axis=-1) / nx
    # If the ks distance is found at 0 or 1, there probably is a rounding error problem. This could be made faster I think.
    invalid_idx = np.logical_or(np.isclose(x, 0), np.isclose(x, 1))
    ds = cp.abs(y1 - y2)
    ds[invalid_idx] = 0
    return cp.amax(ds, axis=-1)


def searchsortednd(a, x, **kwargs):  # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy + cupy + reshapes
    orig_shape, n = a.shape[:-1], a.shape[-1]
    m = np.prod(orig_shape)
    a = a.reshape(m, n)
    x = x.reshape(m, 2 * n)
    max_num = cp.maximum(cp.nanmax(a) - cp.nanmin(a), cp.nanmax(x) - cp.nanmin(x)) + 1
    r = max_num * cp.arange(m)[:, None]
    p = cp.searchsorted((a + r).ravel(), (x + r).ravel(), side="right").reshape(m, -1)
    return (p - n * (cp.arange(m)[:, None])).reshape((*orig_shape, -1))


def sanitize(x, rounding):
    if rounding is not None:
        x = cp.around(x, rounding)
    return cp.nan_to_num(x, nan=0)


def ks(a, b):  # scipy.stats-like implementation using cupy and vectorized searchsorted
    a, b = cp.sort(a, axis=-1), cp.sort(b, axis=-1)
    nx = cp.sum(~cp.isnan(a), axis=-1)[..., cp.newaxis]
    x = cp.concatenate([a, b], axis=-1) # Concat all data
    y1 = searchsortednd(a, x, side="right") / nx
    y2 = searchsortednd(b, x, side="right") / nx
    ds = cp.abs(y1 - y2)
    return cp.nanmax(ds, axis=-1)


def ttest(a, mub, varb): # T-test metric
    mua = cp.nanmean(a, axis=-1)
    vara = cp.nanvar(a, axis=-1, ddof=1)
    var = vara + varb
    t = cp.sqrt(a.shape[-1]) * (mua - mub) / cp.sqrt(var)
    t =cp.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t[cp.isclose(var, 0)] = 0
    return t


def mwu(a, b): # Mann-Whitney U metric
    u = cp.zeros_like(a[:, :, :, 0])
    # Has to be a double loop otherwise the arrays would be too big to fit in the GPU. Anyways I'm looping over a small index (20 x 20 or 100 x 100)
    for i in range(b.shape[-1]):
        for j in range(a.shape[-1]):
            u += (b[..., j] > a[..., i]) + 0.5 * (b[..., j] == a[..., i])
    
    return cp.amax(cp.asarray([u, a.shape[-1] ** 2 - u]), axis=0)


def wraptest(b, test):
    if test == "KS":
        to_do = ks
        other_args = [b]
    elif test == "T": # never round it does not matter for t test 
        to_do = ttest
        mub = cp.nanmean(b, axis=-1)
        varb = cp.nanvar(b, axis=-1, ddof=1)
        other_args = [mub, varb]
    elif test == "MWU":
        to_do = mwu
        other_args = [b]
    else:
        raise ValueError("Wrong test specifier")
    return to_do, other_args


def one_s(darr, ref, notref, n_sam, replace, test, crit_val, rounding: int = None): # Performs one chunk worth of test.
    # Draw reference. Should redraw for every test maybe ? Shouldn't matter
    idxs_ref = ran.choice(darr.shape[-1], n_sam, replace=replace)
    b = darr[ref][..., idxs_ref]
    b = sanitize(b, rounding)
    # Predefine ref to be filled for every test in notref
    rej = cp.empty((len(notref), *darr.shape[1:4]), dtype=bool)
    # Some test-specific definitions, a bit ugly
    to_do, other_args = wraptest(b, test)
    for n in range(len(notref)):
        # Draw test
        idxs = ran.choice(darr.shape[-1], n_sam, replace=replace)
        a = darr[notref[n]][..., idxs]
        a = sanitize(a, rounding) # cloud cover vars causing issues
        # Do the do
        rej[n, ...] = cp.abs(to_do(a, *other_args)) > crit_val[test]
    return rej


def ks_p(d, n): # p-values of the KS test, from the distance (output of ks(a, b))
    return cp.exp(- d ** 2 * n)


def t_p(t): # only valid for high number of dof (> 30), otherwise implement t_p(t, n) that uses the exact distribution
    return 0.5 * (1 + cupy_erf(t))


def p_wrapper(test, metric, n):
    if test == "KS":
        return ks_p(metric, n)
    if test == "T":
        if n >= 30:
            return t_p(metric)
        else:
            raise ValueError(f"{test}, n < 30, not implemented")
    raise ValueError(f"Wrong test specifier : {test}")


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



def create_axes(m, n):
    with open("/users/hbanderi/cosmo-sp/rotated_pole.pkl", "rb") as handle:
        rotated_pole = pkl.load(handle)
    pole_lat = rotated_pole["grid_north_pole_latitude"]
    pole_lon = rotated_pole["grid_north_pole_longitude"]

    # Transform for rotated lat/lon
    crs_rot = ctp.crs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

    # Figure
    projection = crs_rot
    fig, axes = plt.subplots(
        m,
        n,
        subplot_kw={"projection": projection},
        constrained_layout=True,
        figsize=(int(6.5 * n), int(6.5 * m)),
    )
    coastline = ctp.feature.NaturalEarthFeature(
        "physical", "coastline", "10m", edgecolor="black", facecolor="none"
    )
    borders = ctp.feature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "10m",
        edgecolor="grey",
        facecolor="none",
    )
    for ax in np.atleast_1d(axes).flatten():
        ax.set_xlim([-33.93, 23.71])  # heh
        ax.set_ylim([-28.93, 27.39])
        ax.add_feature(coastline)
        ax.add_feature(borders)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
    return fig, axes


def create_plot(to_plot, titles, level, twolevel=False, startindex=-1, cmap=CMAP):
    transform = ctp.crs.PlateCarree()

    if twolevel:
        n = len(to_plot) // 2
        m = 2
    else:
        n = len(to_plot)
        m = 1

    fig, axes = create_plot(m, n)
    axes = np.atleast_1d(axes)
    axes = axes.flatten()
    # axes = np.atleast_1d(axes)

    # Add coastline and boarders

    plt_rej = []
    cbar = [None] * len(to_plot)
    for j in range(len(to_plot)):
        ax = axes[j]
        plt_rej.append(
            ax.contourf(
                lon,
                lat,
                to_plot[j][startindex],
                levels=levels[j],
                transform=transform,
                transform_first=True,
                cmap="coolwarm" if j != 2 else cmap,
                zorder=0,
            )
        )
        ax.set_title(f"Day {startindex}, {titles[j]}")

        cbar[j] = fig.colorbar(plt_rej[j], ax=ax, fraction=0.046, pad=0.04)

    def animate_all(i):
        global plt_rej
        for j in range(len(to_plot)):
            ax = axes[j]
            for c in plt_rej[j].collections:
                c.remove()
            plt_rej[j] = ax.contourf(
                lon,
                lat,
                # lon,
                to_plot[j][i],
                levels=levels[j],
                transform=transform,
                transform_first=True,
                cmap="coolwarm" if j != 2 else cmap,
                zorder=0,
            )
            ax.set_title(
                f"Day {i + 1}, {titles[j]}, g.a : {np.mean(to_plot[j][i]):.4f}"
            )
            cbar[j] = fig.colorbar(plt_rej[j], cax=fig.axes[len(axes) + j])
        return plt_rej

    return fig, axes, plt_rej, animate_all