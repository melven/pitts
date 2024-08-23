# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# to run this with nvprof: nvprof --openacc-profiling off $(which python) torch_svd_bench.py 2500000 20 100

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
def svd_torch(X, nIter=1):
    for i in range(nIter):
        # should use dgesdd according to the documentation
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    return U, S, Vt

@timer
def random(shape):
    return torch.rand(*shape,dtype=torch.float64, device=torch.cuda.current_device())


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for torch.linalg.svd')
    parser.add_argument('n', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random([args.n, args.m])
    print('svd_torch: cuda')
    svd_torch(X, nIter=args.nIter)
    svd_torch(X, nIter=args.nIter)

if __name__ == '__main__':
    main()
