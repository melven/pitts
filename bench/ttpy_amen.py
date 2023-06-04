#!/usr/bin/env python3

"""
Call ttpy amen for comparison for solving a linear system in TT format
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2023-03-28'

import numpy as np
import timeit
import pitts_py
import tt as ttpy
from tt.amen import amen_solve
from tt.core.tools import matvec
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator
from tt_rank1_preconditioner import TT_Rank1_preconditioner


def pitts_tt_to_ttpy(tt):
    nDim = len(tt.dimensions())
    cores = [tt.getSubTensor(iDim) for iDim in range(nDim)]
    return ttpy.vector.from_list(cores)

def pitts_ttOp_to_ttpy(tt):
    nDim = len(tt.row_dimensions())
    cores = [tt.getSubTensor(iDim) for iDim in range(nDim)]
    return ttpy.matrix.from_list(cores)

if __name__ == '__main__':
    pitts_py.initialize()

    np.set_printoptions(linewidth=200)

    dims = [50,]*10
    TTOp = LaplaceOperator(dims)
    symmetric = False
    if not symmetric:
        TTOpConvection = ConvectionOperator(dims)
        pitts_py.axpby(10/np.sqrt(len(dims)), TTOpConvection, 1, TTOp)

    TTb = pitts_py.TensorTrain_double(dims)
    TTb.setTTranks(50)
    pitts_py.randomize(TTb)
    #TTb.setOnes()
    nrm_b = pitts_py.normalize(TTb)

    TTx = pitts_py.TensorTrain_double(dims)
    pitts_py.copy(TTb, TTx)

    precondition = False
    if precondition:
        precond = TT_Rank1_preconditioner(TTOp, twosided=True)

        TTOp_unprecond = TTOp
        TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
        TTtmp = pitts_py.TensorTrainOperator_double(dims, dims)
        pitts_py.apply(precond.TTOpL, TTOp_unprecond, TTtmp)
        pitts_py.apply(TTtmp, precond.TTOpR, TTOp)
        nrm_TTOp = pitts_py.normalize(TTOp)

        # apply to b
        TTb_unprecond = TTb
        nrm_b_unprecond = nrm_b
        TTb = pitts_py.TensorTrain_double(dims)
        pitts_py.apply(precond.TTOpL, TTb_unprecond, TTb)
        nrm_b *= pitts_py.normalize(TTb)
        nrm_b *= 1/nrm_TTOp

        # apply inverse to x
        #TTx0_unprecond = TTx
        #TTx = pitts_py.TensorTrain_double(dims)
        #invPrecond = TT_Rank1_preconditioner(precond.TTOp)
        #pitts_py.apply(invPrecond.TTOp, TTx0_unprecond, TTx)
        #pitts_py.normalize(TTx)

        # preconditioner tries to make the operator closer to identity, so use rhs as initial guess
        #pitts_py.copy(TTb, TTx)

    ttpy_TTOp = pitts_ttOp_to_ttpy(TTOp)
    print(ttpy_TTOp)

    ttpy_TTb = pitts_tt_to_ttpy(TTb)
    print(ttpy_TTb)

    ttpy_TTx = pitts_tt_to_ttpy(TTx)
    print(ttpy_TTx)

    wtime = timeit.default_timer()
    ttpy_TTx = amen_solve(ttpy_TTOp, ttpy_TTb, ttpy_TTx, 1.e-8, nswp=40, local_iters=1, local_restart=80, verb=10, kickrank=50)
    print(ttpy_TTx)
    wtime = timeit.default_timer() - wtime
    print('wtime ttpy AMEN', wtime)
    nrm_r = (ttpy_TTb - matvec(ttpy_TTOp, ttpy_TTx)).norm()
    print('res. norm:', nrm_r, '(abs) ', nrm_r / ttpy_TTb.norm(), '(rel), b norm', ttpy_TTb.norm(), 'x norm', ttpy_TTx.norm())

    pitts_py.clearPerformanceStatistics()
    wtime = timeit.default_timer()
    pitts_py.solveMALS(TTOp, symmetric, pitts_py.MALS_projection.RitzGalerkin, TTb, TTx,
            nSweeps=40, residualTolerance=1.e-8, maxRank=150, useTTgmres=False, gmresMaxIter=80, gmresRelTol=1.e-8, nMALS=1, nOverlap=0, nAMEnEnrichment=50,
            simplifiedAMEn=True)
    #pitts_py.solveGMRES(TTOp, TTb, TTx, maxIter=200, maxRank=150, symmetric=True, absResTol=100, relResTol=1.e-8)
    wtime = timeit.default_timer() - wtime
    print('wtime pitts_py MALS', wtime)
    pitts_py.printPerformanceStatistics()
    TTr = pitts_py.TensorTrain_double(TTb.dimensions())
    pitts_py.apply(TTOp, TTx, TTr)
    r_nrm = pitts_py.axpby(1., TTb, -1., TTr, 0.)
    print("real resNorm: %g" % r_nrm )

    ttpy_TTx_ref = pitts_tt_to_ttpy(TTx)
    nrm_x_ref = ttpy_TTx_ref.norm()
    nrm_err = (ttpy_TTx - ttpy_TTx_ref).norm()
    print('diff TTx pitts_py and ttpy_tt: abs. err', nrm_err, 'rel. err', nrm_err/nrm_x_ref)

    if precondition:
        TTx_unprecond = pitts_py.TensorTrain_double(dims)
        pitts_py.apply(precond.TTOpR, TTx, TTx_unprecond)

        TTr = pitts_py.TensorTrain_double(TTb.dimensions())
        pitts_py.apply(TTOp, TTx, TTr)
        r_nrm = pitts_py.axpby(1., TTb, -1., TTr, 0.)
        print("unprecond. real resNorm: %g" % r_nrm )

    pitts_py.finalize(False)
