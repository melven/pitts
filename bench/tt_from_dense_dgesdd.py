import functools
import timeit
import argparse
import numpy as np
from scipy import linalg


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
def tt_from_dense(X, n, d, max_r, nIter=1):
    for i in range(nIter):
        tt_cores = tt_svd_scipy(X, n, d, max_r)
    return tt_cores

@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')

def tt_svd_scipy(X, n, d, max_r, threshold=1.e-8):
    cores = list()
    rank = 1
    for i in range(d-1):
        X = X.reshape((X.size//(rank*n), rank*n), order='F')
        U, S, Vt = linalg.svd(X, full_matrices=False)
        print('S', S)
        new_rank = 1 + np.max(np.where(S / S[0] > threshold))
        new_rank = min(new_rank, max_r)
        print('rank', new_rank)
        cores.append(Vt[:new_rank,:].reshape((new_rank, n, rank), order='F'))
        X = U[:,:new_rank] @ np.diag(S[:new_rank])
        rank = new_rank
    cores.append(X.reshape((1,n,rank), order='F'))
    return cores

def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for converting a dense n^d tensor to tensor train format with the ttpy library')
    parser.add_argument('n', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('max_r', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random((args.n**args.d,))
    tt_from_dense(X, args.n, args.d, max_r=args.max_r, nIter=args.nIter)


if __name__ == '__main__':
    main()
