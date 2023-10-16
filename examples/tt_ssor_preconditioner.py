#!/usr/bin/env python3

# Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SSOR precondition for a matrix splitting A=L+D+U in tensor-train format
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-07-27'

import numpy as np
import pitts_py
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator


class SSOR_preconditioner:
    def __init__(self, TTop, omega, triangularJacobiIterations=10):
        self.omega = omega
        self.triangularJacobiIterations = triangularJacobiIterations

        assert(TTop.row_dimensions() == TTop.col_dimensions())
        dims = TTop.row_dimensions()

        nDims = len(dims)
        # diagonal part
        self.D = TTop_diag(TTop)
        self.D_nrm = pitts_py.normalize(self.D)

        # lower part
        self.L = TTop_lowerTriangle(TTop)
        pitts_py.axpby(-self.D_nrm, self.D, 1, self.L)

        # upper part
        self.U = TTop_upperTriangle(TTop)
        pitts_py.axpby(-self.D_nrm, self.D, 1, self.U)

        # calculate inverse of D
        D_ranks = self.D.getTTranks()
        assert(D_ranks == [1,]*(nDims-1))
        self.Dinv_nrm = 1/self.D_nrm
        self.Dinv = TTop_diagInverse(self.D)

    def LD_solve(self, b, x, rankTolerance, maxRank):
        """ Approximates x = (D/omega + L)^(-1) b

        Jacobi iteration for approximating a triangular solve (forward-backward substitution is not efficient in TT format)
        """
        Lx = pitts_py.TensorTrain_double(x.dimensions())

        pitts_py.apply(self.Dinv, b, x)
        x_nrm = self.Dinv_nrm * 1/self.omega
        for iter in range(self.triangularJacobiIterations):
            pitts_py.apply(self.L, x, Lx)
            Lx_nrm = pitts_py.axpby(1, b, -x_nrm, Lx, rankTolerance, maxRank)
            pitts_py.apply(self.Dinv, Lx, x)
            x_nrm = Lx_nrm * self.Dinv_nrm * 1/self.omega

        return x_nrm

    def DU_solve(self, b, x, rankTolerance, maxRank):
        """ Approximates x = (D/omega + U)^(-1) b

        Jacobi iteration for approximating a triangular solve (forward-backward substitution is not efficient in TT format)
        """
        Ux = pitts_py.TensorTrain_double(x.dimensions())

        pitts_py.apply(self.Dinv, b, x)
        x_nrm = self.Dinv_nrm * 1/self.omega
        for iter in range(self.triangularJacobiIterations):
            pitts_py.apply(self.U, x, Ux)
            Ux_nrm = pitts_py.axpby(1, b, -x_nrm, Ux, rankTolerance, maxRank)
            pitts_py.apply(self.Dinv, Ux, x)
            x_nrm = Ux_nrm * self.Dinv_nrm * 1/self.omega

        return x_nrm

    def apply(self, x, y, rankTolerance, maxRank):
        tmp = pitts_py.TensorTrain_double(x.dimensions())

        y_nrm = self.DU_solve(x, y, rankTolerance, maxRank)

        pitts_py.apply(self.D, y, tmp)
        tmp_nrm = y_nrm * self.D_nrm * (2-self.omega)/self.omega

        y_nrm = tmp_nrm * self.LD_solve(tmp, y, rankTolerance, maxRank)

        y_nrm = y_nrm * pitts_py.normalize(y, rankTolerance, maxRank)

        return y_nrm


def TTop_diag(TTop):
    D = pitts_py.TensorTrainOperator_double(TTop.row_dimensions(), TTop.col_dimensions())
    pitts_py.copy(TTop, D)
    nDims = len(TTop.row_dimensions())
    for iDim in range(nDims):
        subT = TTop.getSubTensor(iDim)
        # copy diagonal entries
        assert(subT.shape[1] == subT.shape[2])
        n = subT.shape[1]
        newSubT = np.zeros(subT.shape)
        for i in range(n):
            newSubT[:,i,i,:] = subT[:,i,i,:]
        D.setSubTensor(iDim, newSubT)
    return D


def TTop_lowerTriangle(TTop):
    L = pitts_py.TensorTrainOperator_double(TTop.row_dimensions(), TTop.col_dimensions())
    pitts_py.copy(TTop, L)
    nDims = len(TTop.row_dimensions())
    for iDim in range(nDims):
        subT = TTop.getSubTensor(iDim)
        # copy diagonal entries
        assert(subT.shape[1] == subT.shape[2])
        n = subT.shape[1]
        newSubT = np.zeros(subT.shape)
        for i in range(n):
            for j in range(i,n):
                newSubT[:,j,i,:] = subT[:,j,i,:]
        L.setSubTensor(iDim, newSubT)
    return L


def TTop_upperTriangle(TTop):
    U = pitts_py.TensorTrainOperator_double(TTop.row_dimensions(), TTop.col_dimensions())
    pitts_py.copy(TTop, U)
    nDims = len(TTop.row_dimensions())
    for iDim in range(nDims):
        subT = TTop.getSubTensor(iDim)
        # copy diagonal entries
        assert(subT.shape[1] == subT.shape[2])
        n = subT.shape[1]
        newSubT = np.zeros(subT.shape)
        for i in range(n):
            for j in range(i,n):
                newSubT[:,i,j,:] = subT[:,i,j,:]
        U.setSubTensor(iDim, newSubT)
    return U


def TTop_diagInverse(TTop):
    Dinv = pitts_py.TensorTrainOperator_double(TTop.row_dimensions(), TTop.col_dimensions())
    pitts_py.copy(TTop, Dinv)
    nDims = len(TTop.row_dimensions())
    for iDim in range(nDims):
        subT = TTop.getSubTensor(iDim)
        # element-wise inverse
        assert(subT.shape[1] == subT.shape[2])
        n = subT.shape[1]
        for i in range(n):
            subT[:,i,i,:] = 1/subT[:,i,i,:]
        Dinv.setSubTensor(iDim, subT)
    return Dinv


if __name__ == '__main__':
    pitts_py.initialize()

    TTOp = LaplaceOperator([50,]*4)
    pitts_py.axpby(0.1, ConvectionOperator(TTOp.row_dimensions()), 1, TTOp)

    SSORop = SSOR_preconditioner(TTOp, 1.0, 3)

    # perform some checks
    x = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    x.setTTranks(5)
    pitts_py.randomize(x)
    pitts_py.normalize(x)

    # Dinv * D == I
    y = pitts_py.TensorTrain_double(x.dimensions())
    pitts_py.apply(SSORop.D, x, y)
    z = pitts_py.TensorTrain_double(x.dimensions())
    pitts_py.apply(SSORop.Dinv, y, z)
    z_nrm = pitts_py.axpby(-1, x, SSORop.Dinv_nrm * SSORop.D_nrm, z)
    print('Dinv*D*random error:', z_nrm)
    assert(z_nrm < 1.e-8)

    # TTop = L + D + U
    tmpOp = pitts_py.TensorTrainOperator_double(TTOp.row_dimensions(), TTOp.col_dimensions())
    pitts_py.copy(TTOp, tmpOp)
    pitts_py.axpby(-1, SSORop.L, 1, tmpOp)
    pitts_py.axpby(-SSORop.D_nrm, SSORop.D, 1, tmpOp)
    pitts_py.axpby(-1, SSORop.U, 1, tmpOp)
    tmpOp_nrm = pitts_py.normalize(tmpOp)
    print('"A = L + D + U" error', tmpOp_nrm)
    assert(tmpOp_nrm < 1.e-8)

    # SSOR iteration
    b = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    b.setTTranks(3)
    pitts_py.randomize(b)
    pitts_py.normalize(b)

    eps = 1.e-5
    
    x.setZero()
    x_nrm = 0
    for i in range(20):
        # y <- b - Ax
        pitts_py.apply(TTOp, x, y)
        y_nrm = pitts_py.axpby(1, b, -x_nrm, y, rankTolerance=eps)
        print('residual', y_nrm, 'ranks', x.getTTranks())
        # z <- M^(-1) y
        z_nrm = y_nrm * SSORop.apply(y, z, eps, 99999)
        x_nrm = pitts_py.axpby(z_nrm, z, x_nrm, x, rankTolerance=eps)


    pitts_py.finalize()
