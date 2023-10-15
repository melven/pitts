#!/usr/bin/env python3

# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

""" Compare TSQR runtime with numpy qr runtime """

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-10-03'

import numpy as np
import pitts_py
import functools
import timeit


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
def numpy_qr(X):
    R = np.linalg.qr(X, mode='r')

@timer
def pitts_qr(X):
    R = pitts_py.block_TSQR(X, 20, False)

if __name__ == '__main__':

    pitts_py.initialize(True)

    n = 10000000
    m = 50

    X = pitts_py.MultiVector_double(n, m)
    pitts_py.randomize(X)

    for j in range(1, m+1):
        print("n: ", n, ", m: ", j)
        X.resize(n,j)
        pitts_qr(X)
        numpy_qr(X)

    n = n//10
    m = 100
    print("n: ", n, ", m: ", m)
    X.resize(n, m)
    pitts_py.randomize(X)
    pitts_qr(X)
    numpy_qr(X)

    n = n//10
    m = 200
    print("n: ", n, ", m: ", m)
    X.resize(n, m)
    pitts_py.randomize(X)
    pitts_qr(X)
    numpy_qr(X)

    pitts_py.finalize(True)
