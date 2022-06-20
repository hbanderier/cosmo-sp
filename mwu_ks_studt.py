#!/usr/bin/env python

"""
Perform the Mann-Whitney U test, the Kolmogorov-Smirnov test, and the Student's
t-test for the following ensembles:
- GPU double precision (reference & control)
- CPU double precision
- GPU single precision
- GPU double precision with additional explicit diffusion

Make sure to compile the cpp files for the Mann-Whitney U test and the
Kolmogorov-Smirnov test first before running this script (see mannwhitneyu.cpp
and kolmogorov_smirnov.cpp).

Copyright (c) 2022 ETH Zurich, Christian Zeman
MIT License
"""

import numpy as np
import xarray as xr
import pickle
import mannwhitneyu as mwu
import kolmogorov_smirnov as ks

rpert = 'e4'        # prefix
n_runs = 50         # total number of runs
n_sel = 100         # how many times we randomly select runs
alpha = 0.05        # significance level
nm = 20             # members per ensemble
u_crit = 127        # nm = 20
t_crit = 2.024      # nm = 20
replace = False     # to bootstrap or not to bootstrap
nbins = 100         # Kolmogorov-Smirnov

# Some arrays to make life easier
tests = ['mwu', 'ks', 't']
comparisons = ['c', 'cpu', 'sp', 'diff']

# Variable
variables = ['t_850hPa', 'fi_500hPa', 'u_10m', 't_2m', 'precip', 'asob_t',
             'athb_t', 'ps']

path_gpu = '../data/10d_gpu_cpu_sp_diff/gpu_dycore/'
path_cpu = '../data/10d_gpu_cpu_sp_diff/cpu_nodycore/'
path_gpu_sp = '../data/10d_gpu_cpu_sp_diff/gpu_dycore_sp/'
path_gpu_diff = '../data/10d_gpu_cpu_sp_diff/gpu_dycore_diff/'

# Final rejection rates
rej_rates = {}
for comp in comparisons:
    rej_rates[comp] = {}
    for vname in variables:
        rej_rates[comp][vname] = {}

runs_r = {}
runs_c = {}
runs_cpu = {}
runs_sp = {}
runs_diff = {}

# Load data for gpu (reference and control) and cpu
for i in range(n_runs):
    i_str_r = str(i).zfill(4)
    i_str_c = str(i+n_runs).zfill(4)
    fname_r = path_gpu + rpert + '_' + i_str_r + '.nc'
    fname_c = path_gpu + rpert + '_' + i_str_c + '.nc'
    fname_cpu = path_cpu + rpert + '_' + i_str_r + '.nc'
    fname_sp = path_gpu_sp + rpert + '_' + i_str_r + '.nc'
    fname_diff = path_gpu_diff + rpert + '_' + i_str_r + '.nc'
    runs_r[i] = {}
    runs_c[i] = {}
    runs_cpu[i] = {}
    runs_sp[i] = {}
    runs_diff[i] = {}
    runs_r[i]['dset'] = xr.open_dataset(fname_r)
    runs_c[i]['dset'] = xr.open_dataset(fname_c)
    runs_cpu[i]['dset'] = xr.open_dataset(fname_cpu)
    runs_sp[i]['dset'] = xr.open_dataset(fname_sp)
    runs_diff[i]['dset'] = xr.open_dataset(fname_diff)

# Test for each variable
for vname in variables:
    print("----------------------------")
    print("Working on " + vname + " ...")
    print("----------------------------")

    # initialize arrays
    nt, ny, nx = runs_r[0]['dset'][vname].shape
    values_r = np.zeros((nt, ny, nx, nm))
    values_c = np.zeros((nt, ny, nx, nm))
    values_cpu = np.zeros((nt, ny, nx, nm))
    values_sp = np.zeros((nt, ny, nx, nm))
    values_diff = np.zeros((nt, ny, nx, nm))

    # For the results
    results = {}
    for test in tests:
        results[test] = {}
        for comp in comparisons:
            results[test][comp] = np.zeros((n_sel, nt))

    # Do test multiple times with random selection of ensemble members
    for s in range(n_sel):
        if ((s+1) % 10 == 0):
            print(str(s+1) + " / " + str(n_sel))

        # Pick random samples for comparison
        idxs_r = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_c = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_cpu = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_sp = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_diff = np.random.choice(np.arange(n_runs), nm, replace=replace)

        # ============================================================
        # Mann-Whitney U test
        # ============================================================
        test = 'mwu'
        # Put together arrays
        for i in range(nm):
            values_r[:,:,:,i] = runs_r[idxs_r[i]]['dset'][vname].values
            values_c[:,:,:,i] = runs_c[idxs_c[i]]['dset'][vname].values
            values_cpu[:,:,:,i] = runs_cpu[idxs_cpu[i]]['dset'][vname].values
            values_sp[:,:,:,i] = runs_sp[idxs_sp[i]]['dset'][vname].values
            values_diff[:,:,:,i] = runs_diff[idxs_diff[i]]['dset'][vname].values

        # Call test
        reject_c = mwu.mwu(values_r, values_c, u_crit)
        reject_cpu = mwu.mwu(values_r, values_cpu, u_crit)
        reject_sp = mwu.mwu(values_r, values_sp, u_crit)
        reject_diff = mwu.mwu(values_r, values_diff, u_crit)
        results[test]['c'][s] = np.mean(reject_c, axis=(1,2))
        results[test]['cpu'][s] = np.mean(reject_cpu, axis=(1,2))
        results[test]['sp'][s] = np.mean(reject_sp, axis=(1,2))
        results[test]['diff'][s] = np.mean(reject_diff, axis=(1,2))

        # ============================================================
        # Kolmogorov-Smirnov test
        # ============================================================
        test = 'ks'
        # Call test
        reject_c = ks.ks(values_r, values_c, nbins)
        reject_cpu = ks.ks(values_r, values_cpu, nbins)
        reject_sp = ks.ks(values_r, values_sp, nbins)
        reject_diff = ks.ks(values_r, values_diff, nbins)
        results[test]['c'][s] = np.mean(reject_c, axis=(1,2))
        results[test]['cpu'][s] = np.mean(reject_cpu, axis=(1,2))
        results[test]['sp'][s] = np.mean(reject_sp, axis=(1,2))
        results[test]['diff'][s] = np.mean(reject_diff, axis=(1,2))

        # ============================================================
        # Student's t-test
        # ============================================================
        test = 't'
        # Means
        mean_r = np.mean(values_r, axis=-1)
        mean_c = np.mean(values_c, axis=-1)
        mean_cpu = np.mean(values_cpu, axis=-1)
        mean_sp = np.mean(values_sp, axis=-1)
        mean_diff = np.mean(values_diff, axis=-1)

        # Variance
        var_r = np.zeros((nt, ny, nx))
        var_c = np.zeros((nt, ny, nx))
        var_cpu = np.zeros((nt, ny, nx))
        var_sp = np.zeros((nt, ny, nx))
        var_diff = np.zeros((nt, ny, nx))
        for i in range(nm):
            var_r += (values_r[:,:,:,i] - mean_r)**2
            var_c += (values_c[:,:,:,i] - mean_c)**2
            var_cpu += (values_cpu[:,:,:,i] - mean_cpu)**2
            var_sp += (values_sp[:,:,:,i] - mean_sp)**2
            var_diff += (values_diff[:,:,:,i] - mean_diff)**2

        # Unbiased estimator for standard deviation
        var_r /= (nm-1)
        var_c /= (nm-1)
        var_cpu /= (nm-1)
        var_sp /= (nm-1)
        var_diff /= (nm-1)
        stdev_c = np.sqrt(((nm-1) * var_r + (nm-1) * var_c) / (2*nm - 2))
        stdev_cpu = np.sqrt(((nm-1) * var_r + (nm-1) * var_cpu) / (2*nm - 2))
        stdev_sp = np.sqrt(((nm-1) * var_r + (nm-1) * var_sp) / (2*nm - 2))
        stdev_diff = np.sqrt(((nm-1) * var_r + (nm-1) * var_diff) / (2*nm - 2))

        # t-value
        t_c = np.abs((mean_r - mean_c) / (stdev_c * np.sqrt(2/nm)))
        t_cpu = np.abs((mean_r - mean_cpu) / (stdev_cpu * np.sqrt(2/nm)))
        t_sp = np.abs((mean_r - mean_sp) / (stdev_sp * np.sqrt(2/nm)))
        t_diff = np.abs((mean_r - mean_diff) / (stdev_diff * np.sqrt(2/nm)))

        # Rejection arrays
        reject_c = t_c > t_crit
        reject_cpu = t_cpu > t_crit
        reject_sp = t_sp > t_crit
        reject_diff = t_diff > t_crit
        results[test]['c'][s] = np.mean(reject_c, axis=(1,2))
        results[test]['cpu'][s] = np.mean(reject_cpu, axis=(1,2))
        results[test]['sp'][s] = np.mean(reject_sp, axis=(1,2))
        results[test]['diff'][s] = np.mean(reject_diff, axis=(1,2))

    # Store results
    for comp in comparisons:
        for test in tests:
            res = results[test][comp]
            rej_rates[comp][vname][test] = {}
            rr = rej_rates[comp][vname][test]
            rr['q_05'] = np.quantile(res, 0.5, axis=0)
            rr['q_005'] = np.quantile(res, 0.05, axis=0)
            rr['q_095'] = np.quantile(res, 0.95, axis=0)
            rr['mean'] = np.mean(res, axis=0)
            rr['min'] = np.min(res, axis=0)
            rr['max'] = np.max(res, axis=0)
            rr['reject'] = res

# Save rejection rates
with open('rr_mwu_ks_studt.pickle', 'wb') as handle:
    pickle.dump(rej_rates, handle)
