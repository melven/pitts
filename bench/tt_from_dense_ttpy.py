# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import timeit
import argparse
import numpy as np
import tt


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
def tt_from_dense(X, max_r, nIter=1):
    for i in range(nIter):
        ttX = tt.vector(X, rmax=max_r)
    return ttX

@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for converting a dense n^d tensor to tensor train format with the ttpy library')
    parser.add_argument('n', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('max_r', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    dims = [args.n]*args.d

    X = random(dims)
    tt_from_dense(X, max_r=args.max_r, nIter=args.nIter)


if __name__ == '__main__':
    main()
