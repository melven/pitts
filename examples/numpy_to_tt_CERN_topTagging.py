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
def read_data_to_multivector(file_name, dims, samples):

    n_total = np.prod(dims)
    n_rows = n_total//dims[-1]
    n_cols = dims[-1]
    Xm = pitts_py.MultiVector_float(n_rows, n_cols)
    Xm_view = np.array(Xm, copy=False)

    for j in range(n_cols):
        print('column', j, 'of', n_cols)
        X = np.load(file_name, mmap_mode='r')
        Xview = X[:samples,...].T.reshape((n_rows, n_cols), order='F', copy=False)
        Xm_view[:,j] = Xview[:,j]
        del Xview, X
    return Xm


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


@timer
def calculate_TT_SVD(Xm, work, dims):
    #Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.4, maxRank=1500)
    Xtt = pitts_py.fromDense_twoSided(Xm, work, dims, rankTolerance=0.01, maxRank=500)
    return Xtt


if __name__ == '__main__':
    pitts_py.initialize(True)

    # real samples 59*3*2**19
    #dims = [7*5*5*2*2*2] + [2,]*16 + [2*2*2*3*59]
    dims = [467*3] + [2,]*16 + [2*2*2*3*59]
    samples = 59*3*(2**19)
    Xm = read_data_to_multivector('/scratch/CERN_ATLAS_top_tagging/npy/const_unit_c_partially_ordered.npy', dims, samples)

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
    np.savez(file='Xtt_CERN_topTagging_unit_partially_ordered.npz', **xtt_dict)


    pitts_py.finalize(True)
