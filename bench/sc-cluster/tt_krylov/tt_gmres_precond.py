#!/usr/bin/env python3

"""
Driver for running different variants of a TT-GMRES algorithm
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-08-09'

import argparse
import pitts_py
from tt_ssor_preconditioner import SSOR_preconditioner
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator
from tt_gmres_pivmgs_leftprecond import tt_gmres_leftprecond
from tt_gmres_pivmgs_rightprecond import tt_gmres_rightprecond
from tt_gmres_pivmgs import tt_gmres_pivmgs
from tt_gmres import tt_gmres


if __name__ == '__main__':
    pitts_py.initialize()

    parser = argparse.ArgumentParser(description="Run TT-GMRES tests for a generated n^d problem (a*I + b*rand(rank_k) + c*Laplace + d*Convection")

    # problem setup: linear operator
    parser.add_argument('-n', type=int, help='individual dimensions n', required=True)
    parser.add_argument('-d', type=int, help='number of dimensions d', required=True)
    parser.add_argument('-I', type=float, help='Coefficient of identity part of the operator', default=0.0)
    parser.add_argument('-Rc', type=float, help='Coefficient of the random part of the operator', default=0.0)
    parser.add_argument('-Rr', type=int, help='TT-ranks of the random part of the operator', default=2)
    parser.add_argument('-L', type=float, help='Coefficient of the Laplace/diffusion part of the operator', default=1.0)
    parser.add_argument('-C', type=float, help='Coefficient of the convection part of the operator', default=0.1)

    # preconditioner setup
    parser.add_argument('--SSOR_omega', type=float, help='SSOR parameter omega', default=1.3)
    parser.add_argument('--SSOR_jacobiIter', type=int, help='SSOR Jaocbi iteration for approximate triangular solves', default=3)

    # problem setup: right-hand side
    parser.add_argument('--rhs_type', type=str, help='Generated right-hand side vector', choices=['rhs_random', 'rhs_ones', 'lhs_random', 'lhs_ones'], default='rhs_random')
    parser.add_argument('--random_rank', type=int, help='TT-ranks of the random right-hand side vector', default=2)

    # GMRES setup
    parser.add_argument('--eps', type=float, help='desired residual tolerance', required=True)
    parser.add_argument('--maxIter', type=int, help='Max. number of GMRES iterations', required=True)
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive tolerance in the GMRES algorithm')


    args = parser.parse_args()
    print('Arguments:', args)

    dims = [args.n]*args.d

    TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOp.setZero()
    if args.I != 0.0:
        TTOpEye = pitts_py.TensorTrainOperator_double(dims, dims)
        pits_py.axpby(args.I, TTOpEye, 1, TTOp)
    if args.Rc != 0.0:
        TTOpRand = pitts_py.TensorTrainOperator_double(dims, dims)
        TTOpRand.setTTranks(args.Rr)
        pitts_py.randomize(TTOpRand)
        pitts_py.normalize(TTOpRand)
        pitts_py.axpby(args.Rc, TTOpRand, 1, TTOp)
    if args.L != 0.0:
        TTOpLaplace = LaplaceOperator(dims)
        pitts_py.axpby(args.L, TTOpLaplace, 1, TTOp)
    if args.C != 0.0:
        TTOpConvection = ConvectionOperator(dims)
        pitts_py.axpby(args.C, TTOpConvection, 1, TTOp)

    preconOp = SSOR_preconditioner(TTOp, args.SSOR_omega, args.SSOR_jacobiIter)

    b = pitts_py.TensorTrain_double(dims)
    nrm_b = 1
    if args.rhs_type == 'rhs_random':
        b.setTTranks(args.random_rank)
        pitts_py.randomize(b)
        pitts_py.normalize(b)
    elif args.rhs_type == 'rhs_ones':
        b.setOnes()
        nrm_b = pitts_py.normalize(b)
    elif args.rhs_type == 'lhs_random':
        x = pitts_py.TensorTrain_double(dims)
        x.setTTranks(args.random_rank)
        pitts_py.randomize(x)
        pitts_py.normalize(x)
        pitts_py.apply(TTOp, x, b)
        nrm_b = pitts_py.normalize(b)
    elif args.rhs_type == 'lhs_ones':
        x = pitts_py.TensorTrain_double(dims)
        x.setOnes()
        pitts_py.apply(TTOp, x, b)
        nrm_b = pitts_py.normalize(b)

    def AOp(x, y, rankTolerance, maxRank):
        pitts_py.apply(TTOp, x, y)
        y_nrm = pitts_py.normalize(y, rankTolerance, maxRank)
        return y_nrm

    r = pitts_py.TensorTrain_double(b.dimensions())

    print('###### no preconditioning ######')
    x, nrm_x = tt_gmres(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    print('###### no preconditioning (PIVMGS) ######')
    x, nrm_x = tt_gmres_pivmgs(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    print('###### no preconditioning (test left) ######')
    x, nrm_x = tt_gmres_leftprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=None, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    print('###### no preconditioning (test right) ######')
    x, nrm_x = tt_gmres_rightprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=None, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    print('###### left preconditioning ######')
    x, nrm_x = tt_gmres_leftprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=preconOp, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    print('###### right preconditioning ######')
    x, nrm_x = tt_gmres_rightprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=preconOp, adaptiveTolerance=args.adaptive)
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Checked resulting residual norm: %g" % (r_nrm / nrm_b) )

    pitts_py.finalize(verbose=False)

