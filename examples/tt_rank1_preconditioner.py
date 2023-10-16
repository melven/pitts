#!/usr/bin/env python3

# Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor-train operator preconditioner based on the inverse of the rank-1 approximation of the operator
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-03-10'

import numpy as np
import pitts_py
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator


class TT_Rank1_preconditioner:
    """ Takes the rank-1 approximation of the TT operator and inverts it """

    def __init__(self, TTOp, twosided=False):
        assert(TTOp.row_dimensions() == TTOp.col_dimensions())
        dims = TTOp.row_dimensions()

        self.TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
        if twosided:
            self.TTOpL = pitts_py.TensorTrainOperator_double(dims, dims)
            self.TTOpR = pitts_py.TensorTrainOperator_double(dims, dims)
        pitts_py.copy(TTOp, self.TTOp)
        pitts_py.normalize(self.TTOp, 0, 1)

        assert(self.TTOp.getTTranks() == [1,]*(len(dims)-1))
        for iDim in range(len(dims)):
            subT = self.TTOp.getSubTensor(iDim)
            # matrix inverse
            assert(subT.shape[0] == 1)
            assert(subT.shape[1] == subT.shape[2])
            assert(subT.shape[3] == 1)
            U, S, Vt = np.linalg.svd(subT[0,:,:,0])
            rank = np.sum(S/S[0] > 1.e-8)
            invS = np.ones(len(S))
            if twosided:
                invS[0:rank] = 1/np.sqrt(S[0:rank])
                subT[0,:,:,0] = np.diag(invS) @ U.T
                self.TTOpL.setSubTensor(iDim, subT)
                subT[0,:,:,0] = Vt.T @ np.diag(invS)
                self.TTOpR.setSubTensor(iDim, subT)
            else:
                invS[0:rank] = 1/S[0:rank]
            subT[0,:,:,0] = Vt.T @ np.diag(invS) @ U.T
            self.TTOp.setSubTensor(iDim, subT)
        if twosided:
            self.TTOp = None

    def apply(self, x, y, rankTolerance, maxRank):

        pitts_py.apply(self.TTOp, x, y)
        y_nrm = pitts_py.normalize(y, rankTolerance, maxRank)

        return y_nrm


if __name__ == '__main__':
    pitts_py.initialize()

    TTOp = LaplaceOperator([50,]*4)
    pitts_py.axpby(0.1, ConvectionOperator(TTOp.row_dimensions()), 1, TTOp)

    Rank1InvOp = TT_Rank1_preconditioner(TTOp)

    # perform some checks
    x = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    x.setTTranks(5)
    pitts_py.randomize(x)
    pitts_py.normalize(x)

    # rank(Rank1InvOp * x) = rank(x)
    y = pitts_py.TensorTrain_double(x.dimensions())
    Rank1InvOp.apply(x, y, 0, 1000)
    assert(y.getTTranks() == x.getTTranks())

    pitts_py.finalize()
