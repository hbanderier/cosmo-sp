#!/usr/bin/env python3

import argparse
import gc
import pickle as pkl

import numpy as np
import numpy.random as ran
from joblib import Parallel, delayed
import sys
sys.path.append('../')
import kolmogorov_smirnov as ks


def one_s(arr, shape, okay_indices, ref, notref, nbins, replace):
    [n_comp, these_months, nx, ny, n_sam] = shape
    idxs_ref = ran.choice(okay_indices, n_sam, replace=replace)
    rej = np.empty((len(notref), these_months, nx, ny))
    results = np.empty((len(notref), these_months))
    for nr in range(len(notref)):
        idxs = ran.choice(okay_indices, n_sam, replace=replace)
        rej[nr, :, :, :] = ks.ks(
            np.transpose(arr[notref[nr], :, :, :, idxs], (1, 2, 3, 0)),
            np.transpose(arr[ref, :, :, :, idxs_ref], (1, 2, 3, 0)),
            nbins,
        )
        results[nr, :] = np.mean(
            rej[nr, :, :, :],
            axis=(1, 2),
        )
    return rej, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("varname")
    parser.add_argument("at_once")
    varname = parser.parse_args().varname
    at_once = int(parser.parse_args().at_once)
    
    sys.stdout = open(f'.out/{varname}_python.out', 'w')
    with open(f"../../data/resampled/main/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)
    
    variablemap = metadata["variablemap"]
    comps = metadata["comps"]
    notref = np.where(comps != "ref")[0]
    ref = np.where(comps == "ref")[0][0]
    # n_sel = metadata["n_sel"]
    n_sel = 6
    n_sam = metadata["n_sam"]
    nx = metadata["nx"]
    ny = metadata["ny"]
    n_mem = metadata["n_mem"]
    nbins = metadata["nbins"]
    n_months = metadata["n_months"]
    replace = metadata["replace"]
    months_per_chunk = metadata["months_per_chunk"]
    
    n_chunks = int(n_months / months_per_chunk)
    file_letter = "s"  # 's' for semester

    maxndays = {"12": 62, "24": 31}
    minndays = {"12": 56, "24": 28}
    okay_indices = {
        key: np.array(
            [i * maxndays[key] + np.arange(minndays[key]) for i in range(n_mem)]
        ).flatten()
        for key in maxndays.keys()
    }
    rej = np.empty((n_sel, len(notref), n_months, nx, ny))
    results = np.empty((n_sel, len(notref), n_months))
    n_proc = 6
    
    for i in range(0, n_chunks, at_once):
        arr = []
        for j in range(i, i + at_once):
            with open(f"../../data/resampled/main/{varname}s{j}.pickle", "rb") as handle:
                arr.append(pkl.load(handle))
        arr = np.concatenate(arr, axis=1)
        small_shape = list(arr.shape)
        small_shape[-1] = n_sam
        h = variablemap[varname][0][:2]
        indices = np.arange(
            i * months_per_chunk, months_per_chunk * min(n_chunks, i + at_once)
        )
        rej[:, :, indices, :, :], results[:, :, indices] = zip(*Parallel(n_jobs=n_proc, verbose=12, max_nbytes=1e7)(
            delayed(one_s)(arr, small_shape, okay_indices[h], ref, notref, nbins, replace)
            for s in range(n_sel)
        ))

        del arr
        gc.collect()
    with open(f"../../data/results/main/{varname}.pickle", "wb") as handle:
        pkl.dump(results, handle)
    with open(f"../../data/rejection/main/{varname}.pickle", "wb") as handle:
        pkl.dump(rej, handle)
    
    sys.stdout.close()
    