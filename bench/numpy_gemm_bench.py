import functools
import timeit
import argparse
import numpy as np


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
def gemm_numpy(X, M, Y, nIter=1):
    for i in range(nIter):
        np.matmul(X, M, out=Y)


@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')


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
    tmpX = np.zeros((paddedRows, cols), dtype=np.float64, order='F')
    tmpX[:rows,:] = X
    return tmpX[:rows,:]


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for numpy gemm')
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
        y = pad(Y)
    gemm_numpy(X, M, Y, nIter=args.nIter)
    gemm_numpy(X, M, Y, nIter=args.nIter)


if __name__ == '__main__':
    main()
