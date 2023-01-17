#!/usr/bin/env python3

import argparse
import gc
import pickle as pkl

import numpy as np
import numpy.random as ran
from joblib import Parallel, delayed
import sys


def ks(x, n_sam, c_alpha, these_weeks):
    idxs = np.argsort(x, axis=-1)
    x = np.take_along_axis(x, idxs, axis=-1)
    exact = np.isclose(x[..., 0], x[..., -1])
    y1 = np.cumsum(idxs < n_sam, axis=-1) / n_sam
    y2 = np.cumsum(idxs >= n_sam, axis=-1) / n_sam
    d = np.amax(np.abs(y1 - y2), axis=-1)
    
    rej = d > c_alpha * np.sqrt(2 / n_sam)
    rej[exact] = False
    results = np.mean(rej, axis=(1, 2))
    rej = np.stack([np.mean(rej[(k * 4):((k + 1) * 4), :, :], axis=0) for k in range(int(these_weeks / 4))], axis=0)
    return rej, results
    

def one_s(arr, n_sam, ref, notref, nbins, replace, c_alpha):
    small_shape = list(arr.shape)
    n_comp, these_weeks, nx, ny, n_mem = small_shape
    idxs_ref = ran.choice(arr.shape[-1], n_sam, replace=replace)
    avgrej = np.empty((len(notref), int(these_weeks / 4), nx, ny))
    results = np.empty((len(notref), these_weeks))
    for nr in range(len(notref)):
        idxs = ran.choice(arr.shape[-1], n_sam, replace=replace)
        avgrej[nr, :, :, :], results[nr, :] = ks(
            np.concatenate(
                [
                    np.transpose(arr[notref[nr], :, :, :, idxs], (1, 2, 3, 0)),
                    np.transpose(arr[ref, :, :, :, idxs_ref], (1, 2, 3, 0))
                ], 
                axis = -1
            ),
            n_sam,
            c_alpha,
            these_weeks
        )
    return avgrej, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("varname")
    parser.add_argument("at_once")
    varname = parser.parse_args().varname
    at_once = int(parser.parse_args().at_once)

    sys.stdout = f = open(f".out/{varname}_python.out", "w")
    with open(f"../../data/resampled/main/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)

    variablemap = metadata["variablemap"]
    comps = metadata["comps"]
    notref = np.where(comps != "ref")[0]
    ref = np.where(comps == "ref")[0][0]
    n_sel = metadata["n_sel"]
    # n_sel = 2
    # n_sam = metadata["n_sam"]
    n_sam = 7
    nx = metadata["nx"]
    ny = metadata["ny"]
    n_mem = metadata["n_mem"]
    nbins = metadata["nbins"]
    c_alpha = metadata["c_alpha"]
    n_months = metadata["n_months"]
    replace = metadata["replace"]
    months_per_chunk = metadata["months_per_chunk"]

    n_chunks = int(n_months / months_per_chunk)
    file_letter = "s"  # 's' for semester

    maxndays = {"12": 62, "24": 31}
    minndays = {"12": 56, "24": 28}

    h = variablemap[varname][0][:2]
    maxndays = maxndays[h]
    minndays = minndays[h]
    
    with open(f"mask24.pickle", "rb") as handle:
        mask = pkl.load(handle) 
    
    n_days = n_months * maxndays
    n_weeks = int(n_months * 31 / 7)
    days_per_chunk = months_per_chunk * maxndays
    a = np.array(
        [
            [
                (l, maxndays * i + j)
                for i in range(n_mem)
            ]
            for l in range(months_per_chunk * at_once)
            for j in range(maxndays)
        ]
    )
    b = np
    avgrejection = np.empty((n_sel, len(notref), n_months, nx, ny))
    # rejection = np.empty((n_sel, len(notref), n_days, nx, ny))
    results = np.empty((n_sel, len(notref), n_days))
    
    # n_proc = 1
    n_proc = 12
    
    for i in range(0, n_chunks, at_once):
        arr = []
        for j in range(i, i + at_once):
            with open(
                f"../../data/resampled/main/{varname}s{j}.pickle", "rb"
            ) as handle:
                arr.append(pkl.load(handle))
        arr = np.ma.concatenate(arr, axis=1)
        arr = arr[:, a[:,:,0], :, :, a[:,:,1]].transpose((2, 0, 3, 4, 1))
        indices = np.arange(
            i * days_per_chunk, days_per_chunk * min(n_chunks, i + at_once)
        )
        idxs_rej = np.arange(
            i * months_per_chunk, months_per_chunk * min(n_chunks, i + at_once)
        )
        avgrejection[:, :, idxs_rej, :, :], results[:, :, indices] = zip(
            *Parallel(
                n_jobs=n_proc, 
                verbose=12, 
                max_nbytes=1e5,
                batch_size='auto',
                pre_dispatch=n_proc
            )
            (
                delayed(one_s)(
                    arr, n_sam, ref, notref, nbins, replace, c_alpha, maxndays
                )
                for s in range(n_sel)
            )
        )

        del arr
        gc.collect()
    with open(f"../../data/results/main/{varname}.pickle", "wb") as handle:
        pkl.dump(results, handle)
    # with open(f"../../data/rejection/main/{varname}.pickle", "wb") as handle:
    #     pkl.dump(rejection, handle)
    with open(f"../../data/rejection/main/{varname}_avg.pickle", "wb") as handle:
        pkl.dump(avgrejection, handle)
        
    del results, avgrejection
    gc.collect()

    f.close()
