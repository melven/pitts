import numpy as np
import opt_einsum as oe
import functools
import argparse
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
def contract_opt_einsum(optTensor, envLeft, envRight, envTop, gradOptTensor, nIter=1):
    for i in range(nIter):
        oe.contract('ijb,in,jn,lbn,xn,yn,lcn->cxy', optTensor, envLeft, envRight, envTop, envLeft, envRight, envTop, optimize='optimal', out=gradOptTensor)

@timer
def random(shape):
    return np.random.rand(*shape).astype(dtype=np.float64, order='F')


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for oe.einsum for the TTN local gradient calculation')
    parser.add_argument('b_left', type=int)
    parser.add_argument('b_right', type=int)
    parser.add_argument('b_top', type=int)
    parser.add_argument('n_classes', type=int)
    parser.add_argument('n_samples', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    optTensor = random([args.b_left, args.b_right, args.b_top])
    gradOptTensor = random([args.b_left, args.b_right, args.b_top])
    envLeft = random([args.b_left, args.n_samples])
    envRight = random([args.b_right, args.n_samples])
    envTop = random([args.n_classes, args.b_top, args.n_samples])
    contraction_path = oe.contract_path('ijb,in,jn,lbn,xn,yn,lcn->cxy', optTensor, envLeft, envRight, envTop, envLeft, envRight, envTop, optimize='optimal')
    print('============================== contraction path  ==============================')
    print(contraction_path[1])
    print('============================== actual contraction ==============================')

    contract_opt_einsum(optTensor, envLeft, envRight, envTop, gradOptTensor, nIter=args.nIter)
    contract_opt_einsum(optTensor, envLeft, envRight, envTop, gradOptTensor, nIter=args.nIter)


if __name__ == '__main__':
    main()
