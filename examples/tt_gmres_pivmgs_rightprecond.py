#!/usr/bin/env python3

"""
Tensor-train GMRES algorithm to solve linear systems with pivoted iterated modified Gram-Schmidt

Motivated by
Dolgov, S. V.: "TT-GMRES: solution to a linear system in the structured tensor format",
Russian Journal of Numerical Analysis and Mathematical Modelling, Walter de Gruyter GmbH, 2013
DOI: 10.1515/rnam-2013-0009
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-06-17'

import numpy as np
import pitts_py
from tt_ssor_preconditioner import SSOR_preconditioner
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator
from tt_pivmgs import tt_pivmgs


def tt_gmres_rightprecond(AOp, b, nrm_b, eps=1.e-6, maxIter=20, verbose=True, preconOp=None, adaptiveTolerance=True):
    """ Tensor-train GMRES algorithm without restart """

    def calc_solution():
        x = pitts_py.TensorTrain_double(b.dimensions())
        x.setZero()
        nrm_x = 0
        for i in range(len(y)):
            nrm_x = pitts_py.axpby(y[i], V[i], nrm_x, x, eps)
        if preconOp is not None:
            nrm_z = nrm_x * preconOp.apply(x, z, eps, 9999)
            x = z
            nrm_x = nrm_z
        return x, nrm_x

    def residual_error(x, nrm_x):
        # calculate real residual
        r = pitts_py.TensorTrain_double(b.dimensions())
        r_nrm = nrm_x * AOp(x, r, eps/10, maxRank=9999)
        r_nrm = pitts_py.axpby(nrm_b, b, -r_nrm, r, eps/10, maxRank=9999)
        return r_nrm

    if verbose:
        if preconOp is None:
            print('# "iteration"    "rel LSTQ norm"    "rel residual norm"    "new direction rank"     "new Krylov vector rank"    "solution rank"')
        else:
            print('# "iteration"    "rel LSTQ norm"    "rel residual norm"    "new direction rank"      "precond direction rank"    "new Krylov vector rank"    "solution rank"')

    # assumes b is normalized and nrm_b is the desired rhs norm
    # define initial subspace
    beta = nrm_b
    curr_beta = beta
    V = [b]
    m = maxIter
    H = np.zeros((m + 1, m), order='F')

    if preconOp is not None:
        z = pitts_py.TensorTrain_double(b.dimensions())

    if verbose:
        #print("TT-GMRES: initial residual norm: %g, max. rank: %d" % (beta, np.max(b.getTTranks())))
        if preconOp is None:
            print(0, 1, 1, np.max(b.getTTranks()), np.max(b.getTTranks()), 0)
        else:
            print(0, 1, 1, np.max(b.getTTranks()), np.max(b.getTTranks()), np.max(b.getTTranks()), 0)

    for j in range(m):
        if adaptiveTolerance:
            delta = eps / (curr_beta / beta) / (1.2 * m)
        else:
            delta = eps
        w = pitts_py.TensorTrain_double(b.dimensions())

        if preconOp is not None:
            z_nrm = preconOp.apply(V[j], z, delta, 9999)
        else:
            z = V[j]
            z_nrm = 1
        w_nrm = z_nrm * AOp(z, w, delta, maxRank=9999)# maxRank=(j+2)*rank_b)

        if preconOp is not None:
            rank_z = np.max(z.getTTranks())
        rank_w = np.max(w.getTTranks())

        H[:j+2,j] = w_nrm * tt_pivmgs(V, w, delta, maxRank=9999)

        rank_vj = np.max(w.getTTranks())

        Hj = H[:j+2,:j+1]
        betae = np.zeros(j+2)
        betae[0] = beta
        # solving Hj * y = beta e_1
        y, curr_beta, rank, s = np.linalg.lstsq(Hj, betae, rcond=None)
        curr_beta = np.sqrt(curr_beta[0]) if curr_beta.size > 0 else 0
        if verbose:
            #print("TT-GMRES:   LSTSQ residual norm: %g " % (curr_beta / beta) )
            x, nrm_x = calc_solution()
            r_nrm = residual_error(x, nrm_x)
            rank_x = np.max(x.getTTranks())
            if preconOp is None:
                print(j+1, curr_beta/beta, r_nrm / nrm_b, rank_w, rank_vj, rank_x)
            else:
                print(j+1, curr_beta/beta, r_nrm / nrm_b, rank_w, rank_z, rank_vj, rank_x)
        if curr_beta / beta <= eps:
            break

    if not verbose:
        x, nrm_x = calc_solution()
    return x, nrm_x


if __name__ == '__main__':
    pitts_py.initialize()

    #TTOp = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOp.setTTranks(2)
    #pitts_py.randomize(TTOp)
    #pitts_py.normalize(TTOp)
    #TTOpEye = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOpEye.setEye()
    #pitts_py.axpby(1, TTOpEye, 0.1, TTOp)

    TTOp = LaplaceOperator([80,]*6)
    pitts_py.axpby(0.1, ConvectionOperator([80,]*6), 1, TTOp)

    preconOp = SSOR_preconditioner(TTOp, 1.3, 3)

    b = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    b.setTTranks(3)
    pitts_py.randomize(b)
    nrm_b = pitts_py.normalize(b)

    def AOp(x, y, rankTolerance, maxRank):
        pitts_py.apply(TTOp, x, y)
        y_nrm = pitts_py.normalize(y, rankTolerance, maxRank)
        return y_nrm

    #x, nrm_x = tt_gmres_rightprecond(AOp, b, nrm_b, maxIter=100, eps=1.e-3, preconOp=None)
    x, nrm_x = tt_gmres_rightprecond(AOp, b, nrm_b, maxIter=20, eps=1.e-3, preconOp=preconOp)

    print("nrm_x %g" % nrm_x)

    r = pitts_py.TensorTrain_double(b.dimensions())
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Residual norm: %g" % (r_nrm / nrm_b) )


    pitts_py.finalize(verbose=False)

