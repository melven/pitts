#!/usr/bin/env python3

# Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Examples for using the TT-MALS linear solver in pitts
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-07-19'

import pitts_py
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator


if __name__ == '__main__':
    pitts_py.initialize()

    #TTOp = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOp.setTTranks(2)
    #pitts_py.randomize(TTOp)
    #pitts_py.normalize(TTOp)
    #TTOpEye = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOpEye.setEye()
    #pitts_py.axpby(1, TTOpEye, 0.1, TTOp)

    TTOp = LaplaceOperator([40,]*8)
    #pitts_py.axpby(0.1, ConvectionOperator([40,]*8), 1, TTOp)

    b = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    #b.setTTranks(3)
    #pitts_py.randomize(b)
    b.setOnes()
    #nrm_b = pitts_py.normalize(b)

    x = pitts_py.TensorTrain_double(TTOp.col_dimensions())
    pitts_py.copy(b, x)

    resNorm = pitts_py.solveMALS(TTOp, True, b, x, nSweeps=4, residualTolerance=1.e-7, maxRank=100, nMALS=8, nOverlap=1, useTTgmres=True, gmresMaxIter=100, gmresRelTol=1.e-9)

    print("resNorm %g" % resNorm)
    r = pitts_py.TensorTrain_double(b.dimensions())
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(1., b, -1., r, 0.)
    print("Real residual norm: %g" % r_nrm )


    pitts_py.finalize(verbose=True)

