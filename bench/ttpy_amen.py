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
    #TTOpConvection = ConvectionOperator(dims)
    #pitts_py.axpby(0.1/np.sqrt(len(dims)), TTOpConvection, 1, TTOp)

    ttpy_TTOp = pitts_ttOp_to_ttpy(TTOp)
    print(ttpy_TTOp)

    TTb = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    TTb.setOnes()

    ttpy_TTb = pitts_tt_to_ttpy(TTb)
    print(ttpy_TTb)

    ttpy_TTx = ttpy_TTb
    nrm_b = ttpy_TTb.norm()
    wtime = timeit.default_timer()
    ttpy_TTx = amen_solve(ttpy_TTOp, ttpy_TTx, ttpy_TTb, 1.e-8, verb=10)
    print(ttpy_TTx)
    wtime = timeit.default_timer() - wtime
    print('wtime ttpy AMEN', wtime)
    nrm_r = (ttpy_TTb - matvec(ttpy_TTOp, ttpy_TTx)).norm()
    print('abs. norm', nrm_r, 'rel. norm', nrm_r/nrm_b)

    TTx = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    TTx.setOnes()
    wtime = timeit.default_timer()
    pitts_py.solveMALS(TTOp, True, pitts_py.MALS_projection.RitzGalerkin, TTb, TTx,
            nSweeps=20, residualTolerance=1.e-8, maxRank=150, useTTgmres=False, gmresMaxIter=40, gmresRelTol=1.e-4, nMALS=1)
    #pitts_py.solveGMRES(TTOp, TTb, TTx, maxIter=200, maxRank=150, symmetric=True, absResTol=100, relResTol=1.e-8)
    wtime = timeit.default_timer() - wtime
    print('wtime pitts_py MALS', wtime)

    ttpy_TTx_ref = pitts_tt_to_ttpy(TTx)
    nrm_x_ref = ttpy_TTx_ref.norm()
    nrm_err = (ttpy_TTx - ttpy_TTx_ref).norm()
    print('diff TTx pitts_py and ttpy_tt: abs. err', nrm_err, 'rel. err', nrm_err/nrm_x_ref)

    pitts_py.finalize(True)
