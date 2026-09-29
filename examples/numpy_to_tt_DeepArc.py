# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

"""Script for reading the DeepArc data and converting it to the TT format"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2026-09-22'

import functools
import timeit
import numpy as np
import pitts_py
import cv2


def getCurrentMemoryUsage():
    """Memory usage in GB"""
    # from stackoverflow: https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    return int(memusage.strip()) / 10**6


def timer(func):
    """measure runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_fun(*args, **kwargs):
        mem_before = getCurrentMemoryUsage()
        wtime = timeit.default_timer()
        value = func(*args, **kwargs)
        wtime = timeit.default_timer() - wtime
        mem_after = getCurrentMemoryUsage()
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(func.__name__, "wtime:", wtime, "memory useage (GB) before/after:", mem_before, mem_after)
        return value
    return wrapper_fun


@timer
def read_data(file_name, reorder=None):
    x_on_disk = np.load(file_name, mmap_mode='r')
    
    if reorder is None:
        X = np.zeros(x_on_disk.shape, dtype=np.float32)
        X[...] = x_on_disk
        return X

    assert len(reorder) == 200*200
    assert x_on_disk.shape == (18,19,300,200,200)
    sample = np.zeros(200*200, dtype=np.float32, order='F')
    X = np.zeros((18,19,300,200*200), dtype=np.float32)

    for i in range(18):
        for j in range(19):
            for k in range(300):
                sample[...] = x_on_disk[i,j,k,...].reshape(200*200, order='C', copy=False)
                X[i,j,k,reorder] = sample.reshape(reorder.shape, order='F', copy=False)

    return X


@timer
def copy_to_multivector(X, dims):
    n_total = np.prod(dims)
    Xm = pitts_py.MultiVector_float(n_total//dims[-1], dims[-1])
    Xm_view = np.array(Xm, copy=False)
    Xm_view[...] = X.reshape(Xm_view.shape, order='F')
    return Xm


@timer
def tensortrain_to_dict(Xtt):
    tt_dict = dict()
    tt_dict['dimensions'] = Xtt.dimensions()
    tt_dict['ranks'] = Xtt.getTTranks()
    cores = []
    for i in range(len(Xtt.dimensions())):
        cores += [Xtt.getSubTensor(i)]
    tt_dict['cores'] = np.array(cores, dtype=object)
    return tt_dict

def hierarchicalOrdering(*dims):
    print('dims', dims)
    dims_flat = []
    for dim in dims:
        dims_flat = dims_flat + dim
    n_total = np.prod(dims_flat)
    print('dims_flat', dims_flat)

    dims_idx = [list() for dim in dims]
    j = 0
    i = len(dims_flat)
    while i > 0:
        j = j - 1 if j > 0 else len(dims)-1
        if len(dims_idx[j]) < len(dims[j]):
            i = i - 1
            dims_idx[j].insert(0,i)
    print('dims_idx', dims_idx)

    dims_idx_flat = []
    for dim_idx in dims_idx:
        dims_idx_flat = dims_idx_flat + dim_idx

    print('dims_idx_flat', dims_idx_flat)

    idx = np.arange(n_total).reshape(dims_flat, order='F')
    print('original', idx.reshape([np.prod(dim) for dim in dims], order='F'))
    idx = np.moveaxis(idx, np.arange(len(dims_flat)), dims_idx_flat)
    idx = idx.reshape(n_total, order='F')
    #idx_reverse = idx
    #idx_reverse[idx] = np.arange(n_total)
    print('reordered', idx.reshape([np.prod(dim) for dim in dims], order='F'))
    #print('reversed', idx_reverse.reshape([np.prod(dim) for dim in dims], order='F'))
    return idx


@timer
def calculate_TT_SVD(Xm, work, dims):
    #Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.4, maxRank=1500)
    Xtt = pitts_py.fromDense_twoSided(Xm, work, dims, rankTolerance=0.09, maxRank=3000)
    return Xtt


if __name__ == '__main__':
    pitts_py.initialize(True)

    #reorder = hierarchicalOrdering([5,5,2,2,2], [5,5,2,2,2])
    reorder = None
    #X = read_data('/scratch/DeepArk_mirror/SANS_CM/stackedPixelData.npy', reorder)
    X = read_data('DeepArc_data.npy', reorder)

    X = np.swapaxes(X, 0, 2)
    # select subset such that dimensions are easier to split
    assert X.shape == (300,19,18,200,200)

    if reorder is None:
        dims = [150,2,19,2,3,3, 2*2,2,5,5, 2,2*2*5*5]
    else:
        dims = [150,2,19,2,3,3, 2,2,2,2,2,2,5,5*5*5]
    assert np.prod(dims) == np.prod(X.shape)
    dims = list(reversed(dims))
    X = X.T
    Xm = copy_to_multivector(X, dims)
    Xm_ref = pitts_py.MultiVector_float()
    pitts_py.copy(Xm, Xm_ref)

    work = pitts_py.MultiVector_float()
    # just for comparison
    pitts_py.copy(Xm, work)
    Xtt = calculate_TT_SVD(Xm, work, dims)
    print(Xtt)

    pitts_py.printPerformanceStatistics()

    # again for timing
    for i in range(3):
        pitts_py.copy(Xm_ref, Xm)
        Xtt = calculate_TT_SVD(Xm, work, dims)
        pitts_py.printPerformanceStatistics()

    xtt_dict = tensortrain_to_dict(Xtt)
    np.savez(file='Xtt_DeepArc.npz', **xtt_dict)


    pitts_py.finalize(True)
