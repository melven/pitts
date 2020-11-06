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


def timer(func):
    """measure runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_fun(*args, **kwargs):
        wtime = timeit.default_timer()
        value = func(*args, **kwargs)
        wtime = timeit.default_timer() - wtime
        print(func.__name__, "wtime:", wtime)
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
def read_data(file_template, n_samples, n_features, padWithZero=False):

    first, last = distribute(n_samples, MPI.COMM_WORLD)
    nlocal_samples = last - first
    print('rank:', MPI.COMM_WORLD.Get_rank(), 'first:', first, 'last:', last)

    X = np.zeros((nlocal_samples, *n_features), dtype=np.float)

    for i in range(nlocal_samples):
        file_name = file_template.format(first+i+1)
        img = cv2.imread(file_name)
        if img is None and padWithZero:
            X[i,...] = 0
            continue
        if img is None:
            raise FileNotFoundError('Could not find: ' + file_name)
        X[i,...] = img / 255

    return X


@timer
def copy_to_multivector(X, dims):

    n_total = np.prod(dims)
    Xm = pitts_py.MultiVector_float(n_total//dims[-1], dims[-1])
    work = pitts_py.MultiVector_float()
    Xm_view = np.array(Xm, copy=False)
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
    Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.002, maxRank=1000, mpiGlobal=True)
    return Xtt


if __name__ == '__main__':

    pitts_py.initialize(True)
    from mpi4py import MPI

    X = read_data(file_template='/scratch/zoel_ml/ATEK_COPY/combustion{:06d}.jpg',
                  n_samples=2**16,
                  n_features=(192,1024,3),
                  padWithZero=True)

    # check if nProcs is power of 2 (otherwise unsupported
    nProcs = MPI.COMM_WORLD.Get_size()
    iProc = MPI.COMM_WORLD.Get_rank()
    if nProcs & (nProcs-1) != 0:
        raise 'nProcs must be a power of 2'
    log2_nProcs = round(np.log2(nProcs))

    dims=[2,]*(16-log2_nProcs) + [2,]*6 + [3,] + [2,]*6 + [2*2*2*2] + [3,]
    Xm = copy_to_multivector(X, dims)
    del X

    work = pitts_py.MultiVector_float()
    # just for comparison
    pitts_py.copy(Xm, work)
    MPI.COMM_WORLD.Barrier()
    Xtt = calculate_TT_SVD(Xm, work, dims)
    print(Xtt)

    xtt_dict = tensortrain_to_dict(Xtt)
    np.savez(file='Xtt%d.npz'%iProc, **xtt_dict)

    pitts_py.finalize(True)
