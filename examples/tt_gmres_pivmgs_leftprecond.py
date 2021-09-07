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


def tt_gmres_leftprecond(AOp, b, nrm_b, eps=1.e-6, maxIter=20, verbose=True, preconOp=None, adaptiveTolerance=True):
    """ Tensor-train GMRES algorithm without restart """

    def calc_solution():
        x = pitts_py.TensorTrain_double(b.dimensions())
        x.setZero()
        nrm_x = 0
        for i in range(len(y)):
            nrm_x = pitts_py.axpby(y[i], V[i], nrm_x, x, eps)
        return x, nrm_x

    def residual_error(x, nrm_x):
        #print("TT-GMRES: solution max rank %d" % np.max(x.getTTranks()))
        # calculate real residual
        r = pitts_py.TensorTrain_double(b.dimensions())
        r_nrm = nrm_x * AOp(x, r, eps/10, maxRank=9999)
        if preconOp is not None:
            r_nrm = pitts_py.axpby(orig_nrm_b, orig_b, -r_nrm, r, eps/10, maxRank=9999)
            #print("TT-GMRES: real residual norm %g" % (r_nrm/orig_nrm_b) )
        else:
            r_nrm = pitts_py.axpby(nrm_b, b, -r_nrm, r, eps/10, maxRank=9999)
            #print("TT-GMRES: real residual norm %g" % (r_nrm/nrm_b) )
        return r_nrm

    if verbose:
        if preconOp is None:
            print('# "iteration"    "rel LSTQ norm"    "rel residual norm"    "new direction rank"     "new Krylov vector rank"    "solution rank"')
        else:
            print('# "iteration"    "rel LSTQ norm"    "rel residual norm"    "new direction rank"      "precond direction rank"    "new Krylov vector rank"    "solution rank"')

    # assumes b is normalized and nrm_b is the desired rhs norm

    # left-preconditioning, transform RHS
    if preconOp is not None:
        orig_b = b
        orig_nrm_b = nrm_b
        b = pitts_py.TensorTrain_double(orig_b.dimensions())
        nrm_b = nrm_b * preconOp.apply(orig_b, b, eps / 10, 9999)
        nrm_b = nrm_b * pitts_py.normalize(b, eps/10, 9999)

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
            #print("TT-GMRES:   un-preconditioned RHS max. rank: %d" % np.max(orig_b.getTTranks()))
        else:
            print(0, 1, 1, np.max(orig_b.getTTranks()), np.max(b.getTTranks()), np.max(b.getTTranks()), 0)

    for j in range(m):
        if adaptiveTolerance:
            delta = eps / (curr_beta / beta) / (1.2 * m)
        else:
            delta = eps
        w = pitts_py.TensorTrain_double(b.dimensions())

        if preconOp is not None:
            z_nrm = AOp(V[j], z, delta, 9999)#, (j+1)*rank_b)
            w_nrm = z_nrm * preconOp.apply(z, w, delta, 9999)#, (j+2)*rank_b)
        else:
            w_nrm = AOp(V[j], w, delta, 9999)#, (j+2)*rank_b)

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
                print(j+1, curr_beta/beta, r_nrm / orig_nrm_b, rank_w, rank_z, rank_vj, rank_x)
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

    TTOp = LaplaceOperator([40,]*4)
    pitts_py.axpby(0.1, ConvectionOperator([40,]*4), 1, TTOp)

    preconOp = SSOR_preconditioner(TTOp, 1.3, 3)

    b = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    b.setTTranks(1)
    pitts_py.randomize(b)
    nrm_b = pitts_py.normalize(b)

    def AOp(x, y, rankTolerance, maxRank):
        pitts_py.apply(TTOp, x, y)
        y_nrm = pitts_py.normalize(y, rankTolerance, maxRank)
        return y_nrm

    #x, nrm_x = tt_gmres_leftprecond(AOp, b, nrm_b, maxIter=100, eps=1.e-3, preconOp=None)
    x, nrm_x = tt_gmres_leftprecond(AOp, b, nrm_b, maxIter=100, eps=1.e-3, preconOp=preconOp)

    print("nrm_x %g" % nrm_x)

    r = pitts_py.TensorTrain_double(b.dimensions())
    pitts_py.apply(TTOp, x, r)
    r_nrm = pitts_py.axpby(nrm_b, b, -nrm_x, r)
    print("Residual norm: %g" % (r_nrm / nrm_b) )


    pitts_py.finalize(verbose=False)

