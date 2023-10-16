#!/usr/bin/env python3

# Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Call scikit_tt MALS for comparison for solving a linear system in TT format
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2023-03-28'

import numpy as np
import timeit
import pitts_py
import scikit_tt
from scikit_tt.solvers.sle import mals
from tt_laplace_operator import LaplaceOperator


def pitts_tt_to_scikit_tt(tt):
    nDim = len(tt.dimensions())
    cores = [tt.getSubTensor(iDim) for iDim in range(nDim)]
    # reshape to 4d tensors
    cores4d = [np.reshape(core, [core.shape[0], core.shape[1], 1, core.shape[2]]) for core in cores]
    return scikit_tt.tensor_train.TT(cores4d)

def pitts_ttOp_to_scikit_tt(tt):
    nDim = len(tt.row_dimensions())
    cores = [tt.getSubTensor(iDim) for iDim in range(nDim)]
    return scikit_tt.tensor_train.TT(cores)

if __name__ == '__main__':
    pitts_py.initialize()

    np.set_printoptions(linewidth=200)

    TTOp = LaplaceOperator([20,]*6)

    scikit_TTOp = pitts_ttOp_to_scikit_tt(TTOp)
    print(scikit_TTOp)

    TTb = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    TTb.setOnes()

    scikit_TTb = pitts_tt_to_scikit_tt(TTb)
    print(scikit_TTb)

    nSweeps = 20
    scikit_TTx = scikit_TTb
    nrm_b = scikit_TTb.norm()
    wtime = timeit.default_timer()
    for sweep in range(nSweeps):
        scikit_TTx = mals(operator=scikit_TTOp, initial_guess=scikit_TTx, right_hand_side=scikit_TTb, repeats=1)
        print(scikit_TTx)
        nrm_r = (scikit_TTb - scikit_TTOp @ scikit_TTx).norm()
        print('Sweep', sweep, 'abs. norm', nrm_r, 'rel. norm', nrm_r/nrm_b)
        if nrm_r / nrm_b < 1.e-8:
            break
    wtime = timeit.default_timer() - wtime
    print('wtime scikit_tt MALS', wtime)

    TTx = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    TTx.setOnes()
    wtime = timeit.default_timer()
    pitts_py.solveMALS(TTOp, True, pitts_py.MALS_projection.RitzGalerkin, TTb, TTx,
            nSweeps=20, residualTolerance=1.e-8, maxRank=150, useTTgmres=True, gmresMaxIter=10, gmresRelTol=1.e-2)
    wtime = timeit.default_timer() - wtime
    print('wtime pitts_py MALS', wtime)

    scikit_TTx_ref = pitts_tt_to_scikit_tt(TTx)
    nrm_x_ref = scikit_TTx_ref.norm()
    nrm_err = (scikit_TTx - scikit_TTx_ref).norm()
    print('diff TTx pitts_py and scikit_tt: abs. err', nrm_err, 'rel. err', nrm_err/nrm_x_ref)

    pitts_py.finalize(False)
