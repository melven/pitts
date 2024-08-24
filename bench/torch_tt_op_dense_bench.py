# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import timeit
import argparse
import torch

#torch.backends.opt_einsum.strategy = 'optimal'


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
def tt_op_apply_torch(A1, A2, A3, X, Y, nIter=1):
    for i in range(nIter):
        Y_ = torch.einsum('ijk,klmn,nop,jmp->ilo', A1, A2, A3, X)


@timer
def random(shape):
    return torch.rand(*shape, dtype=torch.float64)


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='benchmark for 3d tt-op times dense')
    parser.add_argument('n', type=int)
    parser.add_argument('r', type=int)
    parser.add_argument('rOp', type=int)
    parser.add_argument('nIter', type=int)

    args = parser.parse_args()

    X = random([args.r, args.n, args.r])
    A1 = random([args.r, args.r, args.rOp])
    A2 = random([args.rOp, args.n, args.n, args.rOp])
    A3 = random([args.rOp, args.r, args.r])
    Y = random([args.r, args.n, args.r])
    tt_op_apply_torch(A1, A2, A3, X, Y, nIter=args.nIter)
    tt_op_apply_torch(A1, A2, A3, X, Y, nIter=args.nIter)


if __name__ == '__main__':
    main()
