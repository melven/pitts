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


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for numpy gemm')
    parser.add_argument('n', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random([args.n, args.m])
    M = random([args.m, args.k])
    Y = random([args.n, args.k])
    gemm_numpy(X, M, Y, nIter=args.nIter)
    gemm_numpy(X, M, Y, nIter=args.nIter)


if __name__ == '__main__':
    main()
