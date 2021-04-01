#!/usr/bin/env python3

"""Test script for reading some big data and converting it to TT format"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-09-08'

import functools
import timeit
import numpy as np
import cv2
import pitts_py
import sklearn.cluster


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
def read_data(file_template, n_samples, n_features, padWithZero=False, reorder=None, discreteCosineTransform=False):

    first, last = distribute(n_samples, MPI.COMM_WORLD)
    nlocal_samples = last - first
    print('rank:', MPI.COMM_WORLD.Get_rank(), 'first:', first, 'last:', last)

    if reorder is not None:
        X = np.zeros((nlocal_samples, *reorder.shape), dtype=np.float)
    else:
        X = np.zeros((nlocal_samples, *n_features), dtype=np.float)

    for i in range(nlocal_samples):
        iFrame = 30000+first+i+1
        file_name = file_template.format(iFrame)
        img = cv2.imread(file_name) / 255
        if discreteCosineTransform:
            img[:,:,0] = cv2.dct(img[:,:,0])
            img[:,:,1] = cv2.dct(img[:,:,1])
            img[:,:,2] = cv2.dct(img[:,:,2])
        if img is None and padWithZero:
            X[i,...] = 0
            continue
        if img is None:
            raise FileNotFoundError('Could not find: ' + file_name)
        if reorder is not None:
            X[i,reorder] = img.reshape(reorder.shape, order='F')
        else:
            X[i,...] = img

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
    Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.001, maxRank=1000, mpiGlobal=True)
    return Xtt


if __name__ == '__main__':

    pitts_py.initialize(True)
    from mpi4py import MPI
    nProcs = MPI.COMM_WORLD.Get_size()
    iProc = MPI.COMM_WORLD.Get_rank()

    # pad the number of samples so we get a multiple of 2^d*nProcs
    n_samples = (116487 - 30000) // 4
    d = 0
    while 2**d*nProcs < n_samples:
        d = d+1
    sample_dims = [nProcs,] + [2**d,]
    n_features = (192,1024,3)
    #reorder_features = hierarchicalOrdering([2,]*6+[3], [4,]*5, [3,])
    reorder_features = hierarchicalOrdering([192,], [1024,], [3,])
    feature_dims = [2,]*12 + [3*2**4*3]
    assert(np.prod(sample_dims) < n_samples*2)
    assert(np.prod(sample_dims) >= n_samples)
    assert(np.prod(feature_dims) == np.prod(n_features))
    n_samples_padded = np.prod(sample_dims)

    X = read_data(file_template='/scratch/zoel_ml/ATEK_COPY/combustion{:06d}.jpg',
                  n_samples=n_samples_padded,
                  n_features=(192,1024,3),
                  padWithZero=True,
                  reorder=reorder_features,
                  discreteCosineTransform=True)

    local_dims = sample_dims[1:] + feature_dims
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

    xtt_dict = tensortrain_to_dict(Xtt)
    np.savez(file='Xtt%d.npz'%iProc, **xtt_dict)

    pitts_py.finalize(True)
