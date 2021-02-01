#!/usr/bin/env python3

""" Use TT-SVD to approximate a large square SVD """

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-12-16'

import numpy as np
import pitts_py
import functools
import timeit
import argparse


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

@timer
def random_square_low_rank_matrix(n, r):
    sigma = np.linspace(1, 0, r, endpoint=False, dtype=np.float64)
    U, _ = np.linalg.qr(np.random.rand(n,r).astype(dtype=np.float64))
    Vt, _ = np.linalg.qr(np.random.rand(n,r).astype(dtype=np.float64))
    M = U @ np.diag(sigma) @ Vt.T
    return M, U, sigma, Vt

@timer
def numpy_svd(M):
    U, S, Vt = np.linalg.svd(M)
    return U, S, Vt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='approximate truncated square SVD of a 2^n x 2^n matrix by TT-SVD')
    parser.add_argument('d', type=int)
    parser.add_argument('r', type=int)
    parser.add_argument('rtt', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    pitts_py.initialize(True)

    # generate square random matrix of given rank
    n = 2**args.d
    M, U_ref, sigma_ref, Vt_ref = random_square_low_rank_matrix(n, args.r)
    print('Singular values:', sigma_ref)

    U, sigma, Vt = numpy_svd(M)
    print('Computed singular values:', sigma[:args.r])
    err = np.linalg.norm(sigma[:args.r] - sigma_ref, ord=np.inf)
    print('Error:', err)

    Xm = pitts_py.MultiVector_double(n*n//2, 2)
    Xm_view = np.array(Xm, copy=False)
    Xm_view[...] = M.reshape((n*n//2, 2), order='F')
    Xm_view = None
    dims = [2,]*args.d*2
    work = pitts_py.MultiVector_double()
    Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.0001, maxRank=args.rtt)
    cores = [Xtt.getSubTensor(i) for i in range(len(dims))]
    for i in range(args.d):
        r1 = cores[i].shape[0]
        # 2 = cores[i].shape[1]
        r2 = cores[i].shape[2]
        coreU, coreS, coreVt = np.linalg.svd(cores[i].reshape((r1*2,r2), order='F'), full_matrices=False)
        cores[i] = coreU.reshape((r1,2,r2), order='F')
        r3 = cores[i+1].shape[2]
        tmp = cores[i+1].reshape((r2,2*r3), order='F')
        if i+1 < args.d:
            tmp = np.diag(coreS) @ coreVt.T @ tmp
        else:
            sigmaTT = coreS
            tmp = coreVt.T @ tmp
        cores[i+1] = tmp.reshape((r2,2,r3), order='F')
    print('TT-SVD computed singular values:', sigmaTT[:args.r])
    err = np.linalg.norm(sigmaTT[:args.r] - sigma_ref, ord=np.inf)
    print('Error:', err)
    UTT = np.ones((1,1), dtype=np.float64)
    for i in range(args.d):
        r1 = cores[i].shape[0]
        # 2 = cores[i].shape[1]
        r2 = cores[i].shape[2]
        UTT = UTT @ cores[i].reshape((r1,2*r2), order='F')
        UTT = UTT.reshape((UTT.size//r2, r2), order='F')
    print('UTT', UTT[:,:args.r].T @ U_ref)
    VTT = np.eye(len(sigmaTT), dtype=np.float64)
    for i in range(args.d, 2*args.d):
        r1 = cores[i].shape[0]
        # 2 = cores[i].shape[1]
        r2 = cores[i].shape[2]
        VTT = VTT @ cores[i].reshape((r1,2*r2), order='F')
        VTT = VTT.reshape((VTT.size//r2, r2), order='F')
    VTT = VTT.reshape((len(sigmaTT),n), order='F')
    print('VTT', VTT[:args.r,:] @ Vt_ref)


    pitts_py.finalize(True)
