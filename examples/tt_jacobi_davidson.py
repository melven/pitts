#!/usr/bin/env python3

"""
Tensor-train Jacobi-Davidson algorithm to calculate a few eigenvalues

Jacobi-Davidson method originally presented in
Fokkema, D. R.; Sleijpen, G. L. G. & Van der Vorst, H. A.: "Jacobi--Davidson Style QR and QZ Algorithms for the Reduction of Matrix Pencils", SIAM Journal on Scientific Computing, 1998
DOI: 10.1137/S1064827596300073
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>',
               'Rebekka-Sarah Hennig <Rebekka-Sarah.Hennig@dlr.de>']
__date__ = '2021-02-18'

import numpy as np
import pitts_py
from tt_laplace_operator import LaplaceOperator
from tt_gmres import tt_gmres




def tt_jacobi_davidson(A, x0, symmetric, eps=1.e-6, maxIter=20, arnoldiIter=5, gmresTol=0.01, gmresIter=10, verbose=True):
    """ Simplistic tensor-train Jacobi-Davidson algorithm """

    assert(x0.dimensions() == A.col_dimensions())

    # create empty search space
    W = list()
    AW = list()
    H = np.zeros((0,0))

    for j in range(maxIter):
        if j == 0:
            # initial search direction
            v = pitts_py.TensorTrain_double(A.col_dimensions())
            pitts_py.copy(x0, v)
            pitts_py.normalize(v, 0.01*eps)
        elif j < arnoldiIter:
            # start with some Arnoldi iterations
            v = r
        else:
            # calculate new search direction
            def JDop(x, y, eps):
                # Jacobi-Davidson operator with projections
                # y = (I - q q^T) (A - sigma I) (I - q q^T) x
                # we only do
                # y = (I - q q^T) (A - sigma I) x
                # because we assume that x is already orthogonal to q
                pitts_py.apply(A, x, y)
                y_nrm = pitts_py.normalize(y, 0.01*eps)
                y_nrm = pitts_py.axpby(-sigma, x, 1, y, 0.01*eps)
                qTy = y_nrm * pitts_py.dot(q, y)
                y_nrm = pitts_py.axpby(-qTy, q, y_nrm, y, 0.01*eps)
                return y_nrm
            v, _ = tt_gmres(JDop, r, r_nrm, eps=gmresTol, maxIter=gmresIter, verbose=False)

        # orthogonalize new search direction wrt. previous vectors
        for iOrtho in range(5):
            max_wTv = 0
            for i in range(j):
                wTv = pitts_py.dot(W[i], v)
                max_wTv = max(max_wTv, abs(wTv))
                if abs(wTv) > 0.01*eps:
                    pitts_py.axpby(-wTv, W[i], 1., v, 0.01*eps)
            if max_wTv < 0.01*eps:
                break
        Av = pitts_py.TensorTrain_double(A.row_dimensions())
        pitts_py.apply(A, v, Av)

        W = W + [v]
        AW = AW + [Av]

        # # calculate orthogonality error
        # WtW = np.zeros((j+1,j+1))
        # for i in range(j+1):
        #     for k in range(j+1):
        #         WtW[i,k] = pitts_py.dot(W[i], W[k])
        # WtW_err = np.linalg.norm(WtW - np.eye(j+1,j+1))
        # print('WtW_err', WtW_err)

        # update H = W^T AW
        H = np.pad(H, ((0,1),(0,1)))
        for i in range(j):
            H[i,-1] = pitts_py.dot(W[i], Av)
            H[-1,i] = H[i,-1] if symmetric else pitts_py.dot(v,AW[i])
        H[-1,-1] = pitts_py.dot(v, Av)

        # compute Schur decomposition H QH = QH RH
        if symmetric:
            (RH,QH) = np.linalg.eigh(H)
            eigIdx = 0
        else:
            (RH,QH) = np.linalg.eig(H)
            eigIdx = np.argmin(RH)

        # compute Ritz value and vector
        sigma = RH[eigIdx]
        q = pitts_py.TensorTrain_double(A.col_dimensions())
        q_nrm = 0
        for i in range(j+1):
            q_nrm = pitts_py.axpby(QH[i,eigIdx], W[i], q_nrm, q, 0.01*eps)

        # calculate residual r = A*q-sigma*q
        r = pitts_py.TensorTrain_double(A.col_dimensions())
        pitts_py.apply(A, q, r)
        r_nrm = pitts_py.axpby(-sigma, q, 1, r, 0.01*eps)
        # explicitly orthogonalize r wrt. q (we have approximation errors)
        qTr = pitts_py.dot(q,r)
        pitts_py.axpby(-qTr, q, 1, r, 0.01*eps)
        if verbose:
            print("TT-JacobiDavidson: Iteration %d, approximated eigenvalue: %g, residual norm: %g (orthog. error %g)" %(j, sigma, r_nrm, qTr))

        # abort if accurate enough
        if r_nrm < eps:
            break

    # return resulting eigenvalue and eigenvector approximation
    return sigma, q



if __name__ == '__main__':
    pitts_py.initialize()

    TTOp = LaplaceOperator([10,]*5)

    x0 = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    x0.setOnes()

    sigma, q = tt_jacobi_davidson(TTOp, x0, symmetric=True, eps=1.e-8)

    r = pitts_py.TensorTrain_double(x0.dimensions())
    pitts_py.apply(TTOp, q, r)
    sigma_ref = pitts_py.dot(q, r)
    r_nrm = pitts_py.axpby(-sigma, q, 1, r)
    print("Residual norm: %g" % r_nrm)
    print("Est. eigenvalue: %g, real Ritz value: %g, error: %g" % (sigma, sigma_ref, np.abs(sigma-sigma_ref)))

    pitts_py.finalize()
