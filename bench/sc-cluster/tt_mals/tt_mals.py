#!/usr/bin/env python3

"""
Driver for running different variants of a TT-MALS algorithm for solving linear systems
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-08-05'

import argparse
import pitts_py
import numpy as np
from tt_rank1_preconditioner import TT_Rank1_preconditioner
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator


if __name__ == '__main__':
    pitts_py.initialize()

    parser = argparse.ArgumentParser(description="Run TT-MALS tests for a generated n^d problem (a*I + b*rand(rank_k) + c*Laplace + d*Convection")

    # problem setup: linear operator
    parser.add_argument('-n', type=int, help='individual dimensions n', required=True)
    parser.add_argument('-d', type=int, help='number of dimensions d', required=True)
    parser.add_argument('-I', type=float, help='Coefficient of identity part of the operator', default=0.0)
    parser.add_argument('-Rc', type=float, help='Coefficient of the random part of the operator', default=0.0)
    parser.add_argument('-Rr', type=int, help='TT-ranks of the random part of the operator', default=2)
    parser.add_argument('-L', type=float, help='Coefficient of the Laplace/diffusion part of the operator', default=1.0)
    parser.add_argument('-C', type=float, help='Coefficient of the convection part of the operator, gets scaled by 1/sqrt(d) to obtain a dimension-independent constant convection velocity', default=0.1)

    # preconditioner setup
    parser.add_argument('--preconditioner', type=str, help='type of the preconditioner', choices=['none', 'TT-rank1'], default='none')

    # problem setup: right-hand side
    parser.add_argument('--rhs_type', type=str, help='Generated right-hand side vector', choices=['rhs_random', 'rhs_ones', 'lhs_random', 'lhs_ones'], default='rhs_ones')
    parser.add_argument('--rhs_random_rank', type=int, help='TT-ranks of the random right-hand side vector', default=2)
    parser.add_argument('--lhs_type', type=str, help='Generated initial guess (left-hand side vector)', choices=['random', 'ones', 'rhs'], default='rhs')
    parser.add_argument('--lhs_random_rank', type=int, help='TT-ranks of the random right-hand side vector', default=2)

    # MALS setup
    parser.add_argument('--nSweeps', type=int, help='max. number of MALS sweeps', required=True)
    parser.add_argument('--eps', type=float, help='desired residual tolerance', required=True)
    parser.add_argument('--nMALS', type=int, help='number of TT cores to combine (standard ALS: 1, standard MALS: 2, only use inner solver: d)', default=2)
    parser.add_argument('--nOverlap', type=int, help='number of TT cores in two consecutive MALS steps (standard ALS: 0, standard MALS: 1)', default=1)
    parser.add_argument('--projection', type=str, help='desired MALS projection', choices=['RitzGalerkin', 'PetrovGalerkin', 'NormalEquations'], default='RitzGalerkin')
    parser.add_argument('--nAMEnEnrichment', type=int, help='additional rank for enriching the basis with directions from the residual (AMEn method)', default=0)
    # inner GMRES setup
    parser.add_argument('--useTTgmres', action='store_true', help='Use TT-GMRES as inner solve (instead of "standard" GMRES)')
    parser.add_argument('--maxRank', type=int, default=150, help='max. TT rank')
    parser.add_argument('--gmresMaxIter', type=int, help='max. number of inner GMRES iterations', default=100)
    parser.add_argument('--gmresRelTol', type=float, help='relative tolerance for the inner GMRES iterations', default=1.e-3)


    args = parser.parse_args()
    print('# Arguments:', args)

    dims = [args.n]*args.d

    TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOp.setZero()
    symmetric = True
    if args.projection == 'RitzGalerkin':
        projection = pitts_py.MALS_projection.RitzGalerkin
    elif args.projection == 'PetrovGalerkin':
        projection = pitts_py.MALS_projection.PetrovGalerkin
    elif args.projection == 'NormalEquations':
        projection = pitts_py.MALS_projection.NormalEquations

    if args.I != 0.0:
        TTOpEye = pitts_py.TensorTrainOperator_double(dims, dims)
        TTOpEye.setEye()
        pits_py.axpby(args.I, TTOpEye, 1, TTOp)
    if args.Rc != 0.0:
        TTOpRand = pitts_py.TensorTrainOperator_double(dims, dims)
        TTOpRand.setTTranks(args.Rr)
        pitts_py.randomize(TTOpRand)
        pitts_py.normalize(TTOpRand)
        pitts_py.axpby(args.Rc, TTOpRand, 1, TTOp)
        symmetric = False
    if args.L != 0.0:
        TTOpLaplace = LaplaceOperator(dims)
        pitts_py.axpby(args.L, TTOpLaplace, 1, TTOp)
    if args.C != 0.0:
        TTOpConvection = ConvectionOperator(dims)
        pitts_py.axpby(args.C/np.sqrt(args.d), TTOpConvection, 1, TTOp)
        symmetric = False

    b = pitts_py.TensorTrain_double(dims)
    if args.rhs_type == 'rhs_random':
        b.setTTranks(args.rhs_random_rank)
        pitts_py.randomize(b)
    elif args.rhs_type == 'rhs_ones':
        b.setOnes()
    elif args.rhs_type == 'lhs_random':
        x = pitts_py.TensorTrain_double(dims)
        x.setTTranks(args.rhs_random_rank)
        pitts_py.randomize(x)
        pitts_py.normalize(x)
        pitts_py.apply(TTOp, x, b)
    elif args.rhs_type == 'lhs_ones':
        x = pitts_py.TensorTrain_double(dims)
        x.setOnes()
        pitts_py.apply(TTOp, x, b)
    nrm_b = pitts_py.normalize(b)

    x = pitts_py.TensorTrain_double(dims)
    if args.lhs_type == 'random':
        x.setTTranks(args.lhs_random_rank)
        pitts_py.randomize(x)
    elif args.lhs_type == 'ones':
        x.setOnes()
    elif args.lhs_type == 'rhs':
        pitts_py.copy(b, x)
    nrm_x = pitts_py.normalize(x)

    if args.preconditioner == 'none':
        precond = None
        TTOp_unprecond = TTOp
        b_unprecond = b
        nrm_b_unprecond = nrm_b
    elif args.preconditioner == 'TT-rank1':
        precond = TT_Rank1_preconditioner(TTOp, twosided=True)

        # directly apply preconditioner, as there is not preconditioner support in MALS (yet)
        # apply to TTOp
        TTOp_unprecond = TTOp
        TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
        TTtmp = pitts_py.TensorTrainOperator_double(dims, dims)
        pitts_py.apply(precond.TTOpL, TTOp_unprecond, TTtmp)
        pitts_py.apply(TTtmp, precond.TTOpR, TTOp)
        nrm_TTOp = pitts_py.normalize(TTOp)

        # apply to b
        b_unprecond = b
        nrm_b_unprecond = nrm_b
        b = pitts_py.TensorTrain_double(dims)
        pitts_py.apply(precond.TTOpL, b_unprecond, b)
        nrm_b *= pitts_py.normalize(b)
        nrm_b *= 1/nrm_TTOp

        if args.lhs_type == 'rhs':
            pitts_py.copy(b, x)
        else:
            # apply inverse to x
            x0_unprecond = x
            x = pitts_py.TensorTrain_double(dims)
            invPrecond = TT_Rank1_preconditioner(precond.TTOpR)
            pitts_py.apply(invPrecond.TTOp, x0_unprecond, x)
            nrm_x *= pitts_py.normalize(x)
    else:
        raise ValueError("Unknown preconditioner type '"+args.preconditioner+'"')


    b_ = b
    b = pitts_py.TensorTrain_double(dims)
    b.setTTranks(b_.getTTranks())
    for i in range(len(dims)):
        b.setSubTensor(i, b_.getSubTensor(i))

    resNorm = pitts_py.solveMALS(
            TTOp, symmetric, projection, b, x,
            nSweeps=args.nSweeps, residualTolerance=args.eps, maxRank=args.maxRank, nMALS=args.nMALS, nOverlap=args.nOverlap,
            useTTgmres=args.useTTgmres, gmresMaxIter=args.gmresMaxIter, gmresRelTol=args.gmresRelTol, nAMEnEnrichment=args.nAMEnEnrichment)
    print("resNorm %g" % resNorm)

    if precond is None:
        x_unprecond = x
    else:
        x_unprecond = pitts_py.TensorTrain_double(dims)
        pitts_py.apply(precond.TTOpR, x, x_unprecond)

        r = pitts_py.TensorTrain_double(b.dimensions())
        pitts_py.apply(TTOp, x, r)
        r_nrm = pitts_py.axpby(1., b, -1., r, 0.)
        print("precond. real resNorm: %g" % r_nrm )

    r = pitts_py.TensorTrain_double(b.dimensions())
    pitts_py.apply(TTOp_unprecond, x_unprecond, r)
    r_nrm = pitts_py.axpby(1., b_unprecond, -nrm_b/nrm_b_unprecond, r, 0.)
    print("Real resNorm: %g" % r_nrm )


    pitts_py.finalize(verbose=True)

