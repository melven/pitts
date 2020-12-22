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
def qr_numpy(X, nIter=1):
    for i in range(nIter):
        r = np.linalg.qr(X, mode='r')
    return r

@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for numpy.linalg.qr')
    parser.add_argument('n', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random([args.n, args.m])
    qr_numpy(X, nIter=args.nIter)
    qr_numpy(X, nIter=args.nIter)


if __name__ == '__main__':
    main()
