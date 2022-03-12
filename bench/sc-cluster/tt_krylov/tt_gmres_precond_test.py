#!/usr/bin/env python3

"""
Driver for running different variants of a TT-GMRES algorithm
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-08-09'

import argparse
import pitts_py
import numpy as np
from tt_ssor_preconditioner import SSOR_preconditioner
from tt_rank1_preconditioner import TT_Rank1_preconditioner
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator
from tt_gmres_pivmgs_leftprecond import tt_gmres_leftprecond
from tt_gmres_pivmgs_rightprecond import tt_gmres_rightprecond
from tt_gmres_pivmgs import tt_gmres_pivmgs
from tt_gmres_pivmgs_smaller_eps import tt_gmres_pivmgs as tt_gmres_pivmgs_smaller_eps
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
    parser.add_argument('-C', type=float, help='Coefficient of the convection part of the operator, gets scaled by 1/sqrt(d) to obtain a dimension-independent constant convection velocity', default=0.1)

    # preconditioner setup
    parser.add_argument('--preconditioner', type=str, help='type of the preconditionner', choices=['none', 'SSOR', 'TT-rank1'], required=True)
    parser.add_argument('--SSOR_omega', type=float, help='SSOR parameter omega', default=1.3)
    parser.add_argument('--SSOR_jacobiIter', type=int, help='SSOR Jaocbi iteration for approximate triangular solves', default=3)

    # problem setup: right-hand side
    parser.add_argument('--rhs_type', type=str, help='Generated right-hand side vector', choices=['rhs_random', 'rhs_ones', 'lhs_random', 'lhs_ones'], default='rhs_random')
    parser.add_argument('--random_rank', type=int, help='TT-ranks of the random right-hand side vector', default=2)

    # GMRES setup
    parser.add_argument('--eps', type=float, help='desired residual tolerance', required=True)
    parser.add_argument('--maxIter', type=int, help='Max. number of GMRES iterations', required=True)
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive tolerance in the GMRES algorithm')
    parser.add_argument('--variant', type=str, help='(un)preconditioned GMRES variant', choices=['all', 'vanilla', 'no_precond', 'left_precond', 'right_precond', 'smaller_eps'], default='all')


    args = parser.parse_args()
    print('# Arguments:', args)

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
        pitts_py.axpby(args.C/np.sqrt(args.d), TTOpConvection, 1, TTOp)

    if args.preconditioner == 'none':
        preconOp = None
    elif args.preconditioner == 'SSOR':
        preconOp = SSOR_preconditioner(TTOp, args.SSOR_omega, args.SSOR_jacobiIter)
    elif args.preconditioner == 'TT-rank1':
        preconOp = TT_Rank1_preconditioner(TTOp)
    else:
        raise ValueError("Unknown preconditioner type '"+args.preconditioner+'"')

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

    if args.variant in ['all', 'vanilla']:
        print('\n###### no preconditioning ######')
        x, nrm_x = tt_gmres(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, adaptiveTolerance=args.adaptive)
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
        print("# Checked resulting residual norm: %g\n" % (r_nrm / nrm_b) )

    if args.variant in ['all', 'no_precond']:
        print('\n###### no preconditioning (PIVMGS) ######')
        x, nrm_x = tt_gmres_pivmgs(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, adaptiveTolerance=args.adaptive)
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
        print("# Checked resulting residual norm: %g\n" % (r_nrm / nrm_b) )

    if args.variant in ['all', 'left_precond']:
        print('\n###### left preconditioning ######')
        x, nrm_x = tt_gmres_leftprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=preconOp, adaptiveTolerance=args.adaptive)
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
        print("# Checked resulting residual norm: %g\n" % (r_nrm / nrm_b) )

    if args.variant in ['all', 'right_precond']:
        print('\n###### right preconditioning ######')
        x, nrm_x = tt_gmres_rightprecond(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, preconOp=preconOp, adaptiveTolerance=args.adaptive)
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
        print("# Checked resulting residual norm: %g\n" % (r_nrm / nrm_b) )

    if args.variant in ['all', 'smaller_eps']:
        print('\n###### smaller eps (PIVMGS) ######')
        x, nrm_x = tt_gmres_pivmgs_smaller_eps(AOp, b, nrm_b, maxIter=args.maxIter, eps=args.eps, adaptiveTolerance=args.adaptive)
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
        print("# Checked resulting residual norm: %g\n" % (r_nrm / nrm_b) )

    pitts_py.finalize(verbose=False)

