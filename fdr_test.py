#!/usr/bin/env python

"""
Perform the Student's t-test for the following ensembles:
- GPU double precision (reference & control)
- CPU double precision
- GPU single precision
- GPU double precision with additional explicit diffusion (0.01, 0.005, 0.001)

The test is performed with the use of subsampling and comparing the mean
of the evaluation ensemble to the 0.95 quantile of the control ensemble,
as well as with the FDR approach for determining field significance.

Copyright (c) 2022 ETH Zurich, Christian Zeman
MIT License
"""

import numpy as np
import xarray as xr
import pandas as pd
import pickle
import math
import scipy.special as sc
import mannwhitneyu as mwu
import kolmogorov_smirnov as ks
from statsmodels.stats.multitest import fdrcorrection

rpert = 'e4'
n_runs = 100 # total number of runs
n_sel = 100     # how many times we randomly select runs
alpha = 0.05
alpha_fdr = 0.05
nm = 50      # members per ensemble
dof = 2*nm - 2
u_crit = 127    # nm = 20
t_crit = 2.024  # nm = 20
replace = False # to bootstrap or not to bootstrap

# Some arrays to make life easier
tests = ['t']
comparisons = ['c', 's', 'sp', 'cpu', 'diff_001', 'diff_0005', 'diff_0001']

# Variable
variables = ['fi_500hPa', 'precip', 'tqc', 't_2m', 'ps', 'u_10m',
             'tqv', 'tqi', 'clct', 't_s', 'asob_t', 'athb_t',
             'alhfl_s', 'ashfl_s', 't_850hPa', 'u_100m']
n_vars = len(variables)

mean = ''
path = '../data/verification/rp4_d5_new/'
path_gpu = path + 'gpu_dycore/' + mean + '/'
path_gpu_diff_001 =  path + 'gpu_dycore_df_001/' + mean + '/'
path_gpu_diff_0005 = path + 'gpu_dycore_df_0005/' + mean + '/'
path_gpu_diff_0001 = path + 'gpu_dycore_df_0001/' + mean + '/'
path_gpu_sp = path + 'gpu_dycore_sp/' + mean + '/'
path_cpu = path + 'cpu_nodycore/' + mean + '/'

def pdf_t(t, dof):
    p = 1 / (math.sqrt(dof)*sc.beta(0.5, dof/2)) * (1+t**2/dof)**(-(dof+1)/2)
    return p

# Final rejection rates
rej_rates = {}
rej_rates_fdr = {}
rej_rates_fdr_p = {}
for comp in comparisons:
    rej_rates[comp] = {}
    rej_rates_fdr[comp] = {}
    rej_rates_fdr_p[comp] = {}
    for vname in variables:
        rej_rates[comp][vname] = {}
        rej_rates_fdr[comp][vname] = {}
        rej_rates_fdr_p[comp][vname] = {}

runs_r = {}
runs_c = {}
runs_s = {}
runs_sp = {}
runs_diff_001 = {}
runs_diff_0005 = {}
runs_diff_0001 = {}
runs_cpu = {}

# Load data for gpu (reference and control) and cpu
for i in range(n_runs):
    i_str_r = str(i).zfill(4)
    i_str_c = str(i+n_runs).zfill(4)
    i_str_s = str(i+2*n_runs).zfill(4)
    fname_r = path_gpu + rpert + '_' + i_str_r + '.nc'
    fname_c = path_gpu + rpert + '_' + i_str_c + '.nc'
    fname_s = path_gpu + rpert + '_' + i_str_s + '.nc'
    fname_sp = path_gpu_sp + rpert + '_' + i_str_r + '.nc'
    fname_diff_001 = path_gpu_diff_001 + rpert + '_' + i_str_r + '.nc'
    fname_diff_0005 = path_gpu_diff_0005 + rpert + '_' + i_str_r + '.nc'
    fname_diff_0001 = path_gpu_diff_0001 + rpert + '_' + i_str_r + '.nc'
    fname_cpu = path_cpu + rpert + '_' + i_str_r + '.nc'
    runs_r[i] = {}
    runs_c[i] = {}
    runs_s[i] = {}
    runs_sp[i] = {}
    runs_diff_001[i] = {}
    runs_diff_0005[i] = {}
    runs_diff_0001[i] = {}
    runs_cpu[i] = {}
    runs_r[i]['dset'] = xr.open_dataset(fname_r)
    runs_c[i]['dset'] = xr.open_dataset(fname_c)
    runs_s[i]['dset'] = xr.open_dataset(fname_s)
    runs_sp[i]['dset'] = xr.open_dataset(fname_sp)
    runs_diff_001[i]['dset'] = xr.open_dataset(fname_diff_001)
    runs_diff_0005[i]['dset'] = xr.open_dataset(fname_diff_0005)
    runs_diff_0001[i]['dset'] = xr.open_dataset(fname_diff_0001)
    runs_cpu[i]['dset'] = xr.open_dataset(fname_cpu)

# Test for each variable
for vname in variables:
    print("----------------------------")
    print("Working on " + vname + " ...")
    print("----------------------------")

    # initialize arrays
    nt, ny, nx = runs_r[0]['dset'][vname].shape
    values_r = np.zeros((nt, ny, nx, nm))
    values_c = np.zeros((nt, ny, nx, nm))
    values_s = np.zeros((nt, ny, nx, nm))
    values_sp = np.zeros((nt, ny, nx, nm))
    values_diff_001 = np.zeros((nt, ny, nx, nm))
    values_diff_0005 = np.zeros((nt, ny, nx, nm))
    values_diff_0001 = np.zeros((nt, ny, nx, nm))
    values_cpu = np.zeros((nt, ny, nx, nm))

    reject_c_fdr = np.zeros((nt, ny, nx))
    reject_s_fdr = np.zeros((nt, ny, nx))
    reject_sp_fdr = np.zeros((nt, ny, nx))
    reject_diff_001_fdr = np.zeros((nt, ny, nx))
    reject_diff_0005_fdr = np.zeros((nt, ny, nx))
    reject_diff_0001_fdr = np.zeros((nt, ny, nx))
    reject_cpu_fdr = np.zeros((nt, ny, nx))
    reject_c_fdr_p = np.zeros((nt, ny, nx))
    reject_s_fdr_p = np.zeros((nt, ny, nx))
    reject_sp_fdr_p = np.zeros((nt, ny, nx))
    reject_diff_001_fdr_p = np.zeros((nt, ny, nx))
    reject_diff_0005_fdr_p = np.zeros((nt, ny, nx))
    reject_diff_0001_fdr_p = np.zeros((nt, ny, nx))
    reject_cpu_fdr_p = np.zeros((nt, ny, nx))

    # For the results
    results = {}
    results_p = {}
    results_fdr = {}
    results_fdr_p = {}
    pvals = {}
    for test in tests:
        results[test] = {}
        results_p[test] = {}
        results_fdr[test] = {}
        results_fdr_p[test] = {}
        pvals[test] = {}
        for comp in comparisons:
            results[test][comp] = np.zeros((n_sel, nt))
            results_p[test][comp] = np.zeros((n_sel, nt))
            results_fdr[test][comp] = np.zeros((n_sel, nt))
            results_fdr_p[test][comp] = np.zeros((n_sel, nt))
            pvals[test][comp] = np.zeros((n_sel, nt))

    # Do test multiple times with random selection of ensemble members
    for s in range(n_sel):
        if ((s+1) % 10 == 0):
            print(str(s+1) + " / " + str(n_sel))

        # Pick random samples for comparison
        idxs_r = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_c = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_s = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_sp = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_diff_001 = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_diff_0005 = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_diff_0001 = np.random.choice(np.arange(n_runs), nm, replace=replace)
        idxs_cpu = np.random.choice(np.arange(n_runs), nm, replace=replace)

        # Fill in values
        for i in range(nm):
            values_r[:,:,:,i] = runs_r[idxs_r[i]]['dset'][vname].values
            values_c[:,:,:,i] = runs_c[idxs_c[i]]['dset'][vname].values
            values_s[:,:,:,i] = runs_s[idxs_s[i]]['dset'][vname].values
            values_sp[:,:,:,i] = runs_sp[idxs_sp[i]]['dset'][vname].values
            values_diff_001[:,:,:,i] = runs_diff_001[idxs_diff_001[i]]['dset'][vname].values
            values_diff_0005[:,:,:,i] = runs_diff_0005[idxs_diff_0005[i]]['dset'][vname].values
            values_diff_0001[:,:,:,i] = runs_diff_0001[idxs_diff_0001[i]]['dset'][vname].values
            values_cpu[:,:,:,i] = runs_cpu[idxs_cpu[i]]['dset'][vname].values

        # ============================================================
        # Student's t-test
        # ============================================================
        test = 't'
        # Means
        mean_r = np.mean(values_r, axis=-1)
        mean_c = np.mean(values_c, axis=-1)
        mean_s = np.mean(values_s, axis=-1)
        mean_sp = np.mean(values_sp, axis=-1)
        mean_diff_001 = np.mean(values_diff_001, axis=-1)
        mean_diff_0005 = np.mean(values_diff_0005, axis=-1)
        mean_diff_0001 = np.mean(values_diff_0001, axis=-1)
        mean_cpu = np.mean(values_cpu, axis=-1)

        # Variance
        var_r = np.zeros((nt, ny, nx))
        var_c = np.zeros((nt, ny, nx))
        var_s = np.zeros((nt, ny, nx))
        var_sp = np.zeros((nt, ny, nx))
        var_diff_001 = np.zeros((nt, ny, nx))
        var_diff_0005 = np.zeros((nt, ny, nx))
        var_diff_0001 = np.zeros((nt, ny, nx))
        var_cpu = np.zeros((nt, ny, nx))
        for i in range(nm):
            var_r += (values_r[:,:,:,i] - mean_r)**2
            var_c += (values_c[:,:,:,i] - mean_c)**2
            var_s += (values_s[:,:,:,i] - mean_s)**2
            var_sp += (values_sp[:,:,:,i] - mean_sp)**2
            var_diff_001 += (values_diff_001[:,:,:,i] - mean_diff_001)**2
            var_diff_0005 += (values_diff_0005[:,:,:,i] - mean_diff_0005)**2
            var_diff_0001 += (values_diff_0001[:,:,:,i] - mean_diff_0001)**2
            var_cpu += (values_cpu[:,:,:,i] - mean_cpu)**2

        # Unbiased estimator for standard deviation
        var_r /= (nm-1)
        var_c /= (nm-1)
        var_s /= (nm-1)
        var_cpu /= (nm-1)
        var_sp /= (nm-1)
        var_diff_001 /= (nm-1)
        var_diff_0005 /= (nm-1)
        var_diff_0001 /= (nm-1)
        stdev_c = np.sqrt(((nm-1) * var_r + (nm-1) * var_c) / (2*nm - 2))
        stdev_s = np.sqrt(((nm-1) * var_r + (nm-1) * var_s) / (2*nm - 2))
        stdev_sp = np.sqrt(((nm-1) * var_r + (nm-1) * var_sp) / (2*nm - 2))
        stdev_diff_001 = np.sqrt(((nm-1) * var_r + (nm-1) * var_diff_001) / (2*nm - 2))
        stdev_diff_0005 = np.sqrt(((nm-1) * var_r + (nm-1) * var_diff_0005) / (2*nm - 2))
        stdev_diff_0001 = np.sqrt(((nm-1) * var_r + (nm-1) * var_diff_0001) / (2*nm - 2))
        stdev_cpu = np.sqrt(((nm-1) * var_r + (nm-1) * var_cpu) / (2*nm - 2))

        # t-value
        t_c = np.abs((mean_r - mean_c) / (stdev_c * np.sqrt(2/nm)))
        t_s = np.abs((mean_r - mean_s) / (stdev_s * np.sqrt(2/nm)))
        t_sp = np.abs((mean_r - mean_sp) / (stdev_sp * np.sqrt(2/nm)))
        t_diff_001 = np.abs((mean_r - mean_diff_001) / (stdev_diff_001 * np.sqrt(2/nm)))
        t_diff_0005 = np.abs((mean_r - mean_diff_0005) / (stdev_diff_0005 * np.sqrt(2/nm)))
        t_diff_0001 = np.abs((mean_r - mean_diff_0001) / (stdev_diff_0001 * np.sqrt(2/nm)))
        t_cpu = np.abs((mean_r - mean_cpu) / (stdev_cpu * np.sqrt(2/nm)))

        # p-value
        p_c = pdf_t(t_c, dof)
        p_s = pdf_t(t_s, dof)
        p_sp = pdf_t(t_sp, dof)
        p_diff_001 = pdf_t(t_diff_001, dof)
        p_diff_0005 = pdf_t(t_diff_0005, dof)
        p_diff_0001 = pdf_t(t_diff_0001, dof)
        p_cpu = pdf_t(t_cpu, dof)

        # fdr
        for i in range(nt):
            rej_c_fdr, p_c_fdr = fdrcorrection(p_c[i].flatten(), alpha_fdr)
            rej_s_fdr, p_s_fdr = fdrcorrection(p_s[i].flatten(), alpha_fdr)
            rej_sp_fdr, p_sp_fdr = fdrcorrection(p_sp[i].flatten(), alpha_fdr)
            rej_diff_001_fdr, p_diff_001_fdr = fdrcorrection(p_diff_001[i].flatten(), alpha_fdr)
            rej_diff_0005_fdr, p_diff_0005_fdr = fdrcorrection(p_diff_0005[i].flatten(), alpha_fdr)
            rej_diff_0001_fdr, p_diff_0001_fdr = fdrcorrection(p_diff_0001[i].flatten(), alpha_fdr)
            rej_cpu_fdr, p_cpu_fdr = fdrcorrection(p_cpu[i].flatten(), alpha_fdr)

            rej_c_fdr_p = p_c_fdr < alpha
            rej_s_fdr_p = p_s_fdr < alpha
            rej_sp_fdr_p = p_sp_fdr < alpha
            rej_diff_001_fdr_p = p_diff_001_fdr < alpha
            rej_diff_0005_fdr_p = p_diff_0005_fdr < alpha
            rej_diff_0001_fdr_p = p_diff_0001_fdr < alpha
            rej_cpu_fdr_p = p_cpu_fdr < alpha

            reject_c_fdr[i] = rej_c_fdr.reshape((ny, nx))
            reject_s_fdr[i] = rej_s_fdr.reshape((ny, nx))
            reject_sp_fdr[i] = rej_sp_fdr.reshape((ny, nx))
            reject_diff_001_fdr[i] = rej_diff_001_fdr.reshape((ny, nx))
            reject_diff_0005_fdr[i] = rej_diff_0005_fdr.reshape((ny, nx))
            reject_diff_0001_fdr[i] = rej_diff_0001_fdr.reshape((ny, nx))
            reject_cpu_fdr[i] = rej_cpu_fdr.reshape((ny, nx))

            reject_c_fdr_p[i] = rej_c_fdr_p.reshape((ny, nx))
            reject_s_fdr_p[i] = rej_s_fdr_p.reshape((ny, nx))
            reject_sp_fdr_p[i] = rej_sp_fdr_p.reshape((ny, nx))
            reject_diff_001_fdr_p[i] = rej_diff_001_fdr_p.reshape((ny, nx))
            reject_diff_0005_fdr_p[i] = rej_diff_0005_fdr_p.reshape((ny, nx))
            reject_diff_0001_fdr_p[i] = rej_diff_0001_fdr_p.reshape((ny, nx))
            reject_cpu_fdr_p[i] = rej_cpu_fdr_p.reshape((ny, nx))

        # Rejection arrays
        reject_c = t_c > t_crit
        reject_s = t_s > t_crit
        reject_sp = t_sp > t_crit
        reject_diff_001 = t_diff_001 > t_crit
        reject_diff_0005 = t_diff_0005 > t_crit
        reject_diff_0001 = t_diff_0001 > t_crit
        reject_cpu = t_cpu > t_crit
        reject_c_p = p_c < alpha
        reject_s_p = p_s < alpha
        reject_sp_p = p_sp < alpha
        reject_diff_001_p = p_diff_001 < alpha
        reject_diff_0005_p = p_diff_0005 < alpha
        reject_diff_0001_p = p_diff_0001 < alpha
        reject_cpu_p = p_cpu < alpha
        results[test]['c'][s] = np.mean(reject_c, axis=(1,2))
        results[test]['s'][s] = np.mean(reject_s, axis=(1,2))
        results[test]['sp'][s] = np.mean(reject_sp, axis=(1,2))
        results[test]['diff_001'][s] = np.mean(reject_diff_001, axis=(1,2))
        results[test]['diff_0005'][s] = np.mean(reject_diff_0005, axis=(1,2))
        results[test]['diff_0001'][s] = np.mean(reject_diff_0001, axis=(1,2))
        results[test]['cpu'][s] = np.mean(reject_cpu, axis=(1,2))
        results_fdr[test]['c'][s] = np.mean(reject_c_fdr, axis=(1,2))
        results_fdr[test]['s'][s] = np.mean(reject_s_fdr, axis=(1,2))
        results_fdr[test]['sp'][s] = np.mean(reject_sp_fdr, axis=(1,2))
        results_fdr[test]['diff_001'][s] = np.mean(reject_diff_001_fdr, axis=(1,2))
        results_fdr[test]['diff_0005'][s] = np.mean(reject_diff_0005_fdr, axis=(1,2))
        results_fdr[test]['diff_0001'][s] = np.mean(reject_diff_0001_fdr, axis=(1,2))
        results_fdr[test]['cpu'][s] = np.mean(reject_cpu_fdr, axis=(1,2))
        results_p[test]['c'][s] = np.mean(reject_c_p, axis=(1,2))
        results_p[test]['s'][s] = np.mean(reject_s_p, axis=(1,2))
        results_p[test]['sp'][s] = np.mean(reject_sp_p, axis=(1,2))
        results_p[test]['diff_001'][s] = np.mean(reject_diff_001_p, axis=(1,2))
        results_p[test]['diff_0005'][s] = np.mean(reject_diff_0005_p, axis=(1,2))
        results_p[test]['diff_0001'][s] = np.mean(reject_diff_0001_p, axis=(1,2))
        results_p[test]['cpu'][s] = np.mean(reject_cpu_p, axis=(1,2))
        results_fdr_p[test]['c'][s] = np.mean(reject_c_fdr_p, axis=(1,2))
        results_fdr_p[test]['s'][s] = np.mean(reject_s_fdr_p, axis=(1,2))
        results_fdr_p[test]['sp'][s] = np.mean(reject_sp_fdr_p, axis=(1,2))
        results_fdr_p[test]['diff_001'][s] = np.mean(reject_diff_001_fdr_p, axis=(1,2))
        results_fdr_p[test]['diff_0005'][s] = np.mean(reject_diff_0005_fdr_p, axis=(1,2))
        results_fdr_p[test]['diff_0001'][s] = np.mean(reject_diff_0001_fdr_p, axis=(1,2))
        results_fdr_p[test]['cpu'][s] = np.mean(reject_cpu_fdr_p, axis=(1,2))

    # Store results
    for comp in comparisons:
        for test in tests:
            res = results[test][comp]
            res_fdr = results_fdr[test][comp]
            res_fdr_p = results_fdr_p[test][comp]
            res_p = results_p[test][comp]
            rej_rates[comp][vname][test] = {}
            rr = rej_rates[comp][vname][test]
            rr['q_05'] = np.quantile(res, 0.5, axis=0)
            rr['q_005'] = np.quantile(res, 0.05, axis=0)
            rr['q_095'] = np.quantile(res, 0.95, axis=0)
            rr['mean'] = np.mean(res, axis=0)
            rr['min'] = np.min(res, axis=0)
            rr['max'] = np.max(res, axis=0)
            rr['reject'] = res
            rr['reject_fdr'] = res_fdr
            rr['reject_fdr_p'] = res_fdr_p
            rr['reject_p'] = res_p

# Save rejection rates
with open('rr_t_fdr_new_100_50.pickle', 'wb') as handle:
    pickle.dump(rej_rates, handle)
