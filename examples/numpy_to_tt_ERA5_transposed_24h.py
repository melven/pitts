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
import sys


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
def read_data_to_multivector(file_name, sample_dims, feature_dims):
    features = np.prod(feature_dims)
    samples = np.prod(sample_dims)
    n_total = features * samples
    n_cols = feature_dims[-1]
    n_rows = n_total//n_cols
    reshape_dims = sample_dims + [features//n_cols, n_cols]
    Xm = pitts_py.MultiVector_float(n_rows, n_cols)
    Xm_view = np.array(Xm, copy=False).reshape(reshape_dims, order='F', copy=False)

    jb = 64
    for j in range(0, reshape_dims[0], jb):
        print('progress', int(j/reshape_dims[0]*100+0.5), '%')
        X = np.load(file_name, mmap_mode='r')
        Xview = X[:samples,...].reshape(reshape_dims, order='C', copy=False)
        Xm_view[j:j+jb,...] = Xview[j:j+jb,...]
        del Xview, X
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


@timer
def calculate_TT_SVD(Xm, work, dims):
    Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.001, maxRank=8000)
    return Xtt


if __name__ == '__main__':
    pitts_py.initialize(True)

    #sample_dims = [7*3*3*3*2**6,] + [2,2,2,2]
    #sample_dims = [3**4*2**9,] + [2,]*4
    sample_dims = [47118,] + [2,]*4
    feature_dims = [2,2,2,2,3,5,480]
    dims = sample_dims + feature_dims
    Xm = read_data_to_multivector('/scratch/ERA5_z500/z500_24h_pattern.npy', sample_dims, feature_dims)

    work = pitts_py.MultiVector_float()
    Xtt = calculate_TT_SVD(Xm, work, dims)
    print(Xtt)

    #pitts_py.printPerformanceStatistics()

    ## again for timing
    #for i in range(3):
    #    pitts_py.copy(Xm_ref, Xm)
    #    Xtt = calculate_TT_SVD(Xm, work, dims)
    #    pitts_py.printPerformanceStatistics()

    xtt_dict = tensortrain_to_dict(Xtt)
    np.savez(file='Xtt_ERA5_transposed_24h_.npz', **xtt_dict)


    pitts_py.finalize(True)
