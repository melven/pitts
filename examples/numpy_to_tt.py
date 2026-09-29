#!/usr/bin/env python3

# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause


#!/usr/bin/env python3

"""Script for reading some big data and converting it to TT format"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-05-27'

import functools
import timeit
import numpy as np
import pitts_py
import cv2


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
    #print('original', idx.reshape([np.prod(dim) for dim in dims], order='F'))
    idx = np.moveaxis(idx, np.arange(len(dims_flat)), dims_idx_flat)
    idx = idx.reshape(n_total, order='F')
    #idx_reverse = idx
    #idx_reverse[idx] = np.arange(n_total)
    #print('reordered', idx.reshape([np.prod(dim) for dim in dims], order='F'))
    #print('reversed', idx_reverse.reshape([np.prod(dim) for dim in dims], order='F'))
    return idx


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


def distribute(nElems, mpiComm):
    iProc = mpiComm.Get_rank()
    nProcs = mpiComm.Get_size()

    first = 0
    last = nElems
    if nProcs > 1:
        nElemsPerProc = nElems // nProcs
        nElemsModProcs = nElems % nProcs
        if iProc < nElemsModProcs:
            first = iProc * (nElemsPerProc+1)
            last = first + nElemsPerProc+1
        else:
            first = iProc * nElemsPerProc + nElemsModProcs
            last = first + nElemsPerProc
    return first, last


@timer
def read_data(file_name, n_samples, n_features, reorder=None, normalize=False):

    first, last = distribute(n_samples, MPI.COMM_WORLD)
    nlocal_samples_padded = last - first
    print('rank:', MPI.COMM_WORLD.Get_rank(), 'first:', first, 'last:', last)

    if reorder is not None:
        X = np.zeros((nlocal_samples_padded, *reorder.shape), dtype=np.float32)
        sample = np.zeros(n_features, dtype=np.float32)
    else:
        X = np.zeros((nlocal_samples_padded, *n_features), dtype=np.float32)

    x_on_disk = np.load(file_name, mmap_mode='r')
    last = min(last, x_on_disk.shape[0])
    nlocal_samples = last - first

    
    for i in range(nlocal_samples):
        if reorder is None:
            X[i,...] = x_on_disk[first+i,...] / 65535
        else:
            sample[...] = x_on_disk[first+i,...] / 65535
            X[i,reorder] = sample.reshape(reorder.shape, order='F')
        if normalize:
            X[i,...] /= np.linalg.norm(X[i,...])

    return X


@timer
def copy_to_multivector(X, dims, batchSize=1):

    n_total = np.prod(dims)
    Xm = pitts_py.MultiVector_float(n_total//dims[-1], dims[-1])
    Xm_view = np.array(Xm, copy=False)
    # move dimension of batch samples to the right
    assert(batchSize == 1)
    #X = X.reshape((X.shape[0]//batchSize, batchSize, *X.shape[1:]), order='C')
    #X = np.moveaxis(X, 1, -1)
    Xm_view[...] = X.reshape(Xm_view.shape, order='F')

    return Xm


@timer
def tensortrain_to_dict(Xtt):
    tt_dict = dict()
    tt_dict['dimensions'] = Xtt.dimensions()
    tt_dict['ranks'] = Xtt.getTTranks()
    tt_dict['cores'] = list()
    for i in range(len(Xtt.dimensions())):
        tt_dict['cores'] += [Xtt.getSubTensor(i)]
    return tt_dict


@timer
def calculate_TT_SVD(Xm, work, dims):
    #Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.005, maxRank=5000, mpiGlobal=True)
    Xtt = pitts_py.fromDense_twoSided(Xm, work, dims, rankTolerance=0.002, maxRank=5000)
    return Xtt


if __name__ == '__main__':

    pitts_py.initialize(True)
    from mpi4py import MPI
    nProcs = MPI.COMM_WORLD.Get_size()
    iProc = MPI.COMM_WORLD.Get_rank()

    file_name = '/scratch/DeepArk_mirror/SANS_CM/stackedPixelData.npy'
    # open file to determine dimensions...
    x = np.load(file_name, mmap_mode='r')
    n_samples = x.shape[0]
    n_features = x.shape[1:]
    x = None

    print("n_samples", n_samples, "n_features", n_features)
    assert(nProcs == 1)
    sample_dims = [n_samples,]
    #reorder_features = hierarchicalOrdering([19,1,1,1,1], [2,2,3,5,5], [2,2,2,5,5], [2,2,2,5,5])
    #feature_dims = [19, 2, 2, 2, 2, 2, 2, 3, 2, 2, 5, 5, 5, 5, 5, 5]
    reorder_features = None
    feature_dims = [19, 5,5,3,2,2, 2,2,2,5,5, 2,2,2,5,5]
    assert(np.prod(sample_dims) < n_samples*2)
    assert(np.prod(sample_dims) >= n_samples)
    assert(np.prod(feature_dims) == np.prod(n_features))
    n_samples_padded = np.prod(sample_dims)

    X = read_data(file_name=file_name,
                  n_samples=n_samples_padded,
                  n_features=n_features,
                  reorder=reorder_features,
                  normalize=False)

    if nProcs > 1:
        local_dims = [1,] + sample_dims[1:] + feature_dims
    else:
        local_dims = sample_dims + feature_dims
    Xm = copy_to_multivector(X, local_dims, batchSize=1)
    del X

    work = pitts_py.MultiVector_float()
    # just for comparison
    pitts_py.copy(Xm, work)
    MPI.COMM_WORLD.Barrier()
    Xtt = calculate_TT_SVD(Xm, work, local_dims)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(Xtt)
        np.savez(file='Xtt_reorder.npz', reorder_features=reorder_features)
        np.savez(file='Xtt_dims.npz', sample_dims=sample_dims, feature_dims=feature_dims)

    xtt_dict = tensortrain_to_dict(Xtt)
    np.savez(file='Xtt%d.npz'%iProc, **xtt_dict)

    pitts_py.finalize(True)
