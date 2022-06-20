#!/usr/bin/env python

"""
Perform the Mann-Whitney U test in parallel for the following ensembles:
- GPU double precision (reference & control)
- CPU double precision
- GPU single precision
- GPU double precision with additional explicit diffusion (0.01, 0.005, 0.001)

Make sure to compile the cpp file for the Mann-Whitney U test first before
running this script (see mannwhitneyu.cpp).

Copyright (c) 2022 ETH Zurich, Christian Zeman
MIT License
"""


import numpy as np
import xarray as xr
import pickle
import timeit
import mannwhitneyu as mwu
import multiprocessing
from joblib import Parallel, delayed
from itertools import repeat

rpert = 'e4'        # prefix
nruns_max = 200     # Maximum number of runs
alpha = 0.05        # significance level
replace = False     # to bootstrap or not to bootstrap

# Number of ensembles and samples
nruns = [200, 100, 50]
nsels = [150, 100, 50, 20]
ntimes = [100]
critvalues = {20:127, 10:23, 5:2}

# Variable
variables = ['fi_500hPa', 'precip', 'tqc', 't_2m', 'ps', 'u_10m',
             'tqv', 'tqi', 'clct', 't_s', 'asob_t', 'athb_t',
             'alhfl_s', 'ashfl_s', 't_850hPa', 'u_100m']
n_vars = len(variables)
mean = ''
# mean = 'gbmean_2_2'
# mean = 'gbmean_4_4'
# mean = 'gbmean_8_8'
# mean = 'gbmean_16_16'
# mean = 'fldmean'

path = '../data/verification/rp4_d5_new/'
path_gpu = path + 'gpu_dycore/' + mean + '/'
path_gpu_diff_001 =  path + 'gpu_dycore_df_001/' + mean + '/'
path_gpu_diff_0005 = path + 'gpu_dycore_df_0005/' + mean + '/'
path_gpu_diff_0001 = path + 'gpu_dycore_df_0001/' + mean + '/'
path_gpu_sp = path + 'gpu_dycore_sp/' + mean + '/'
path_cpu = path + 'cpu_nodycore/' + mean + '/'


runs_r = {}
runs_c = {}
runs_s = {}
runs_sp = {}
runs_cpu = {}
runs_diff_001 = {}
runs_diff_0005 = {}
runs_diff_0001 = {}

# Function to be run in parallel
def get_rrate(idxs, vname, runs_r, runs_c, runs_s, runs_sp, runs_cpu,
              runs_diff_001, runs_diff_0005, runs_diff_0001, nm, u_crit):

    # initialize arrays
    nt, ny, nx = runs_r[0]['dset'][vname].shape
    values_r = np.zeros((nt, ny, nx, nm))
    values_c = np.zeros((nt, ny, nx, nm))
    values_s = np.zeros((nt, ny, nx, nm))
    values_sp = np.zeros((nt, ny, nx, nm))
    values_cpu = np.zeros((nt, ny, nx, nm))
    values_diff_001 = np.zeros((nt, ny, nx, nm))
    values_diff_0005 = np.zeros((nt, ny, nx, nm))
    values_diff_0001 = np.zeros((nt, ny, nx, nm))

    # Put together arrays
    for i in range(nm):
        values_r[:,:,:,i] = runs_r[idxs[i]]['dset'][vname].values
        values_c[:,:,:,i] = runs_c[idxs[i]]['dset'][vname].values
        values_s[:,:,:,i] = runs_s[idxs[i]]['dset'][vname].values
        values_sp[:,:,:,i] = runs_sp[idxs[i]]['dset'][vname].values
        values_cpu[:,:,:,i] = runs_cpu[idxs[i]]['dset'][vname].values
        values_diff_001[:,:,:,i] = runs_diff_001[idxs[i]]['dset'][vname].values
        values_diff_0005[:,:,:,i] = runs_diff_0005[idxs[i]]['dset'][vname].values
        values_diff_0001[:,:,:,i] = runs_diff_0001[idxs[i]]['dset'][vname].values

    # Call test
    below_c = mwu.mwu(values_r, values_c, u_crit)
    below_s = mwu.mwu(values_r, values_s, u_crit)
    below_sp = mwu.mwu(values_r, values_sp, u_crit)
    below_cpu = mwu.mwu(values_r, values_cpu, u_crit)
    below_diff_001 = mwu.mwu(values_r, values_diff_001, u_crit)
    below_diff_0005 = mwu.mwu(values_r, values_diff_0005, u_crit)
    below_diff_0001 = mwu.mwu(values_r, values_diff_0001, u_crit)
    results_c = np.mean(below_c, axis=(1,2))
    results_s = np.mean(below_s, axis=(1,2))
    results_sp = np.mean(below_sp, axis=(1,2))
    results_cpu = np.mean(below_cpu, axis=(1,2))
    results_diff_001 = np.mean(below_diff_001, axis=(1,2))
    results_diff_0005 = np.mean(below_diff_0005, axis=(1,2))
    results_diff_0001 = np.mean(below_diff_0001, axis=(1,2))

    return results_c, results_s, results_sp, results_cpu, results_diff_001, \
            results_diff_0005, results_diff_0001

# Load data
for i in range(nruns_max):
    i_str_r = str(i).zfill(4)
    i_str_c = str(i+nruns_max).zfill(4)
    i_str_s = str(i+2*nruns_max).zfill(4)
    fname_r = path_gpu + rpert + '_' + i_str_r + '.nc'
    fname_c = path_gpu + rpert + '_' + i_str_c + '.nc'
    fname_s = path_gpu + rpert + '_' + i_str_s + '.nc'
    fname_sp = path_gpu_sp + rpert + '_' + i_str_r + '.nc'
    fname_cpu = path_cpu + rpert + '_' + i_str_r + '.nc'
    fname_diff_001 = path_gpu_diff_001 + rpert + '_' + i_str_r + '.nc'
    fname_diff_0005 = path_gpu_diff_0005 + rpert + '_' + i_str_r + '.nc'
    fname_diff_0001 = path_gpu_diff_0001 + rpert + '_' + i_str_r + '.nc'
    runs_r[i] = {}
    runs_c[i] = {}
    runs_s[i] = {}
    runs_sp[i] = {}
    runs_cpu[i] = {}
    runs_diff_001[i] = {}
    runs_diff_0005[i] = {}
    runs_diff_0001[i] = {}
    runs_r[i]['dset'] = xr.open_dataset(fname_r)
    runs_c[i]['dset'] = xr.open_dataset(fname_c)
    runs_s[i]['dset'] = xr.open_dataset(fname_s)
    runs_sp[i]['dset'] = xr.open_dataset(fname_sp)
    runs_cpu[i]['dset'] = xr.open_dataset(fname_cpu)
    runs_diff_001[i]['dset'] = xr.open_dataset(fname_diff_001)
    runs_diff_0005[i]['dset'] = xr.open_dataset(fname_diff_0005)
    runs_diff_0001[i]['dset'] = xr.open_dataset(fname_diff_0001)

# Shape
nt, ny, nx = runs_r[0]['dset'][variables[0]].shape

# All size combinations
rej_rates = {}
for nr in nruns:
    rej_rates[nr] = {}
    for ns in nsels:
        if ns >= nr:
            continue
        rej_rates[nr][ns] = {}
        if ns <= 20:
            uc = critvalues[ns]
        else:
            uc = 0
        for ntim in ntimes:
            start = timeit.default_timer()
            print("nruns = " + str(nr) + ", nsel = " + str(ns) + ", ntimes = "
                  + str(ntim))
            rej_rates[nr][ns][ntim] = {}
            idxs = []


            for t in range(ntim):
                idxs.append(np.random.choice(np.arange(nr), ns,
                                             replace=replace))


            # Do test for each subsample in parallel
            nprocs = 10
            pool = multiprocessing.Pool(nprocs)

            # For the results
            rr = {}
            rr['c'] = {}
            rr['s'] = {}
            rr['sp'] = {}
            rr['cpu'] = {}
            rr['diff_001'] = {}
            rr['diff_0005'] = {}
            rr['diff_0001'] = {}

            rej_rates[nr][ns][ntim]['c'] = {}
            rej_rates[nr][ns][ntim]['s'] = {}
            rej_rates[nr][ns][ntim]['sp'] = {}
            rej_rates[nr][ns][ntim]['cpu'] = {}
            rej_rates[nr][ns][ntim]['diff_001'] = {}
            rej_rates[nr][ns][ntim]['diff_0005'] = {}
            rej_rates[nr][ns][ntim]['diff_0001'] = {}

            for vname in variables:

                rej_rates[nr][ns][ntim]['c'][vname] = {}
                rej_rates[nr][ns][ntim]['s'][vname] = {}
                rej_rates[nr][ns][ntim]['sp'][vname] = {}
                rej_rates[nr][ns][ntim]['cpu'][vname] = {}
                rej_rates[nr][ns][ntim]['diff_001'][vname] = {}
                rej_rates[nr][ns][ntim]['diff_0005'][vname] = {}
                rej_rates[nr][ns][ntim]['diff_0001'][vname] = {}

                # Run test in parallel
                results_c_zip, results_s_zip, results_sp_zip, results_cpu_zip, \
                        results_diff_001_zip, results_diff_0005_zip, \
                        results_diff_0001_zip = \
                        zip(*pool.starmap(get_rrate,
                                      zip(idxs, repeat(vname), repeat(runs_r),
                                          repeat(runs_c),
                                          repeat(runs_s),
                                          repeat(runs_sp),
                                          repeat(runs_cpu),
                                          repeat(runs_diff_001),
                                          repeat(runs_diff_0005),
                                          repeat(runs_diff_0001),
                                          repeat(ns),
                                          repeat(uc))))

                # Store results in numpy arrays
                results_c = np.zeros((ntim, nt))
                results_s = np.zeros((ntim, nt))
                results_sp = np.zeros((ntim, nt))
                results_cpu = np.zeros((ntim, nt))
                results_diff_001 = np.zeros((ntim, nt))
                results_diff_0005 = np.zeros((ntim, nt))
                results_diff_0001 = np.zeros((ntim, nt))
                for t in range(ntim):
                    results_c[t] = results_c_zip[t]
                    results_s[t] = results_s_zip[t]
                    results_sp[t] = results_sp_zip[t]
                    results_cpu[t] = results_cpu_zip[t]
                    results_diff_001[t] = results_diff_001_zip[t]
                    results_diff_0005[t] = results_diff_0005_zip[t]
                    results_diff_0001[t] = results_diff_0001_zip[t]


                rr['c'][vname] = {}
                rr['c'][vname]['q_05'] = np.quantile(results_c, 0.5, axis=0)
                rr['c'][vname]['q_005'] = np.quantile(results_c, 0.05, axis=0)
                rr['c'][vname]['q_095'] = np.quantile(results_c, 0.95, axis=0)
                rr['c'][vname]['mean'] = np.mean(results_c, axis=0)
                rr['c'][vname]['min'] = np.min(results_c, axis=0)
                rr['c'][vname]['max'] = np.max(results_c, axis=0)
                rr['c'][vname]['reject'] = results_c

                rr['s'][vname] = {}
                rr['s'][vname]['q_05'] = np.quantile(results_s, 0.5, axis=0)
                rr['s'][vname]['q_005'] = np.quantile(results_s, 0.05, axis=0)
                rr['s'][vname]['q_095'] = np.quantile(results_s, 0.95, axis=0)
                rr['s'][vname]['mean'] = np.mean(results_s, axis=0)
                rr['s'][vname]['min'] = np.min(results_s, axis=0)
                rr['s'][vname]['max'] = np.max(results_s, axis=0)
                rr['s'][vname]['reject'] = results_s

                rr['sp'][vname] = {}
                rr['sp'][vname]['q_05'] = np.quantile(results_sp, 0.5, axis=0)
                rr['sp'][vname]['q_005'] = np.quantile(results_sp, 0.05, axis=0)
                rr['sp'][vname]['q_095'] = np.quantile(results_sp, 0.95, axis=0)
                rr['sp'][vname]['mean'] = np.mean(results_sp, axis=0)
                rr['sp'][vname]['min'] = np.min(results_sp, axis=0)
                rr['sp'][vname]['max'] = np.max(results_sp, axis=0)
                rr['sp'][vname]['reject'] = results_sp

                rr['cpu'][vname] = {}
                rr['cpu'][vname]['q_05'] = np.quantile(results_cpu, 0.5, axis=0)
                rr['cpu'][vname]['q_005'] = np.quantile(results_cpu, 0.05, axis=0)
                rr['cpu'][vname]['q_095'] = np.quantile(results_cpu, 0.95, axis=0)
                rr['cpu'][vname]['mean'] = np.mean(results_cpu, axis=0)
                rr['cpu'][vname]['min'] = np.min(results_cpu, axis=0)
                rr['cpu'][vname]['max'] = np.max(results_cpu, axis=0)
                rr['cpu'][vname]['reject'] = results_cpu

                rr['diff_001'][vname] = {}
                rr['diff_001'][vname]['q_05'] = np.quantile(results_diff_001, 0.5, axis=0)
                rr['diff_001'][vname]['q_005'] = np.quantile(results_diff_001, 0.05, axis=0)
                rr['diff_001'][vname]['q_095'] = np.quantile(results_diff_001, 0.95, axis=0)
                rr['diff_001'][vname]['mean'] = np.mean(results_diff_001, axis=0)
                rr['diff_001'][vname]['min'] = np.min(results_diff_001, axis=0)
                rr['diff_001'][vname]['max'] = np.max(results_diff_001, axis=0)
                rr['diff_001'][vname]['reject'] = results_diff_001

                rr['diff_0005'][vname] = {}
                rr['diff_0005'][vname]['q_05'] = np.quantile(results_diff_0005, 0.5, axis=0)
                rr['diff_0005'][vname]['q_005'] = np.quantile(results_diff_0005, 0.05, axis=0)
                rr['diff_0005'][vname]['q_095'] = np.quantile(results_diff_0005, 0.95, axis=0)
                rr['diff_0005'][vname]['mean'] = np.mean(results_diff_0005, axis=0)
                rr['diff_0005'][vname]['min'] = np.min(results_diff_0005, axis=0)
                rr['diff_0005'][vname]['max'] = np.max(results_diff_0005, axis=0)
                rr['diff_0005'][vname]['reject'] = results_diff_0005

                rr['diff_0001'][vname] = {}
                rr['diff_0001'][vname]['q_05'] = np.quantile(results_diff_0001, 0.5, axis=0)
                rr['diff_0001'][vname]['q_005'] = np.quantile(results_diff_0001, 0.05, axis=0)
                rr['diff_0001'][vname]['q_095'] = np.quantile(results_diff_0001, 0.95, axis=0)
                rr['diff_0001'][vname]['mean'] = np.mean(results_diff_0001, axis=0)
                rr['diff_0001'][vname]['min'] = np.min(results_diff_0001, axis=0)
                rr['diff_0001'][vname]['max'] = np.max(results_diff_0001, axis=0)
                rr['diff_0001'][vname]['reject'] = results_diff_0001

                # Concatenate dictionaries
                rej_rates[nr][ns][ntim]['c'][vname].update(rr['c'][vname])
                rej_rates[nr][ns][ntim]['s'][vname].update(rr['s'][vname])
                rej_rates[nr][ns][ntim]['sp'][vname].update(rr['sp'][vname])
                rej_rates[nr][ns][ntim]['cpu'][vname].update(rr['cpu'][vname])
                rej_rates[nr][ns][ntim]['diff_001'][vname].update(rr['diff_001'][vname])
                rej_rates[nr][ns][ntim]['diff_0005'][vname].update(rr['diff_0005'][vname])
                rej_rates[nr][ns][ntim]['diff_0001'][vname].update(rr['diff_0001'][vname])

            stop = timeit.default_timer()
            print('calculated in ' + str(stop-start) + ' s\n')

# Save rejection rates
with open('rr_mwu_rp4_d5_new_diff_cpu_sp_' + mean + '.pickle', 'wb') as handle:
    pickle.dump(rej_rates, handle)
