# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import timeit
import argparse
import numpy as np
import scipy as sp
import scipy.linalg


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
def svd_numpy(X, nIter=1):
    for i in range(nIter):
        # should use dgesdd according to the documentation
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt

@timer
def svd_scipy_gesdd(X, nIter=1):
    for i in range(nIter):
        # should use dgesdd according to the documentation
        U, S, Vt = sp.linalg.svd(X, full_matrices=False, lapack_driver='gesvd')
    return U, S, Vt

@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for numpy.linalg.svd')
    parser.add_argument('n', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random([args.n, args.m])
    print('svd_numpy: dgesdd')
    svd_numpy(X, nIter=args.nIter)
    svd_numpy(X, nIter=args.nIter)

    print('svd_scipy_gesdd: dgesvd')
    svd_scipy_gesdd(X, nIter=args.nIter)
    svd_scipy_gesdd(X, nIter=args.nIter)


if __name__ == '__main__':
    main()
