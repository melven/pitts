# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# to run this with nvprof: nvprof --openacc-profiling off $(which python) torch_gemm_bench.py 2500000 20 20 100

import functools
import timeit
import argparse
import torch
#from torch.profiler import profile, record_function, ProfilerActivity


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
def gemm_torch(X, M, Y, nIter=1):
    for i in range(nIter):
        torch.matmul(X, M, out=Y)


@timer
def random(shape):
    return torch.rand(*shape,dtype=torch.float64, device=torch.cuda.current_device())


def paddedSize(n):
    # pad to next number divisible by PD/2 but not by PD
    PD = 8 * 16
    if n < 2*PD:
        return n
    return n + (PD + PD//2 - n % PD) % PD;


@timer
def pad(X):
    rows = X.shape[0]
    cols = X.shape[1]
    paddedRows = paddedSize(rows)
    print('rows', rows)
    print('paddedRows', paddedRows)
    tmpX = torch.zeros((paddedRows, cols), dtype=torch.float64, order='F')
    tmpX[:rows,:] = X
    return tmpX[:rows,:]


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for torch gemm')
    parser.add_argument('n', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('nIter', type=int)
    parser.add_argument('--padding', default=False, action='store_true')

    args = parser.parse_args()

    X = random([args.n, args.m])
    M = random([args.m, args.k])
    Y = random([args.n, args.k])
    if args.padding:
        X = pad(X)
        Y = pad(Y)
    gemm_torch(X, M, Y, nIter=args.nIter)
    gemm_torch(X, M, Y, nIter=args.nIter)
    gemm_torch(X, M, Y, nIter=args.nIter)


if __name__ == '__main__':
    main()
