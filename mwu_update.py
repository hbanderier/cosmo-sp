#!/usr/bin/env python

"""
Perform the Mann-Whitney U test for the following ensembles:
- Before update (reference & control)
- After update

Make sure to compile the cpp file for the Mann-Whitney U test first before
running this script (see mannwhitneyu.cpp).

Copyright (c) 2022 ETH Zurich, Christian Zeman
MIT License
"""

import numpy as np
import xarray as xr
import pickle
import mannwhitneyu as mwu

rpert = 'e4'        # prefix
n_runs = 50         # total number of runs
n_sel = 100         # how many times we randomly select runs
alpha = 0.05        # significance level
nm = 20             # members per ensemble
u_crit = 127        # nm = 20
rej_rates = {}      # results
replace = False     # to bootstrap or not to bootstrap

# Variable
variables = ['t_850hPa', 'fi_500hPa', 'u_10m', 't_2m', 'ps',
             'qv_500hPa']

path_before = '../data/10d_update/before_update/'
path_after = '../data/10d_update/after_update/'
rej_rates['c'] = {}
rej_rates['upd'] = {}

runs_r = {}
runs_c = {}
runs_upd = {}

# Load data for gpu (reference and control) and cpu
for i in range(n_runs):
    i_str_r = str(i).zfill(4)
    i_str_c = str(i+n_runs).zfill(4)
    fname_r = path_before + rpert + '_' + i_str_r + '.nc'
    fname_c = path_before + rpert + '_' + i_str_c + '.nc'
    fname_upd = path_after + rpert + '_' + i_str_r + '.nc'
    runs_r[i] = {}
    runs_c[i] = {}
    runs_upd[i] = {}
    runs_r[i]['dset'] = xr.open_dataset(fname_r)
    runs_c[i]['dset'] = xr.open_dataset(fname_c)
    runs_upd[i]['dset'] = xr.open_dataset(fname_upd)

# Test for each variable
for vname in variables:
    print("----------------------------")
    print("Working on " + vname + " ...")
    print("----------------------------")

    # initialize arrays
    nt, ny, nx = runs_r[0]['dset'][vname].shape
    values_r = np.zeros((nt, ny, nx, nm))
    values_c = np.zeros((nt, ny, nx, nm))
    values_upd = np.zeros((nt, ny, nx, nm))
    results_c = np.zeros((n_sel, nt))
    results_upd = np.zeros((n_sel, nt))

    # Do test multiple times with random selection of ensemble members
    for s in range(n_sel):
        if ((s+1) % 10 == 0):
            print(str(s+1) + " / " + str(n_sel))

        # Pick random samples for comparison
        idxs_r = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_c = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_upd = np.random.choice(np.arange(n_runs), nm, replace=replace)

        # Put together arrays
        for i in range(nm):
            values_r[:,:,:,i] = runs_r[idxs_r[i]]['dset'][vname].values
            values_c[:,:,:,i] = runs_c[idxs_c[i]]['dset'][vname].values
            values_upd[:,:,:,i] = runs_upd[idxs_upd[i]]['dset'][vname].values

        # Call test
        below_c = mwu.mwu(values_r, values_c, u_crit)
        below_upd = mwu.mwu(values_r, values_upd, u_crit)
        results_c[s] = np.mean(below_c, axis=(1,2))
        results_upd[s] = np.mean(below_upd, axis=(1,2))

    # Store results
    rej_rates['c'][vname] = {}
    rej_rates['c'][vname]['q_05'] = np.quantile(results_c, 0.5, axis=0)
    rej_rates['c'][vname]['q_005'] = np.quantile(results_c, 0.05, axis=0)
    rej_rates['c'][vname]['q_095'] = np.quantile(results_c, 0.95, axis=0)
    rej_rates['c'][vname]['mean'] = np.mean(results_c, axis=0)
    rej_rates['c'][vname]['min'] = np.min(results_c, axis=0)
    rej_rates['c'][vname]['max'] = np.max(results_c, axis=0)
    rej_rates['c'][vname]['reject'] = results_c

    rej_rates['upd'][vname] = {}
    rej_rates['upd'][vname]['q_05'] = np.quantile(results_upd, 0.5, axis=0)
    rej_rates['upd'][vname]['q_005'] = np.quantile(results_upd, 0.05, axis=0)
    rej_rates['upd'][vname]['q_095'] = np.quantile(results_upd, 0.95, axis=0)
    rej_rates['upd'][vname]['mean'] = np.mean(results_upd, axis=0)
    rej_rates['upd'][vname]['min'] = np.min(results_upd, axis=0)
    rej_rates['upd'][vname]['max'] = np.max(results_upd, axis=0)
    rej_rates['upd'][vname]['reject'] = results_upd

# Save rejection rates
with open('rr_mwu_update_' + str(nm) + '.pickle', 'wb') as handle:
    pickle.dump(rej_rates, handle)
