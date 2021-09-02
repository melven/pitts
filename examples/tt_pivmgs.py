#!/usr/bin/env python3

"""
Tensor-train pivoted iterated modified Gram-Schmidt ("PIVMGS")
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-06-17'

import numpy as np
import pitts_py


def tt_pivmgs(V, w, rankTolerance, maxRank, verbose=True):
    """
    Tensor-train pivoted iterated modified Gram-Schmidt orthogonalization

    Adds orthogonalizes w wrt V and normalies w, then adds it to the list of directions V.
    Warning: works in-place and does not copy w, so don't access w afterwards anymore!
    """

    w_nrm = pitts_py.normalize(w, rankTolerance, maxRank)
    if len(V) == 0:
        V += [w,]
        return [w_nrm,]

    Vtw = [pitts_py.dot(v,w) for v in V]
    h = [0,]*len(V)
    for ortho in range(3):
        while np.max(np.abs(Vtw)) > rankTolerance:
            pivot = np.argmax(np.abs(Vtw))
            tmp = pitts_py.dot(V[pivot], w)
            h[pivot] += w_nrm*tmp
            if verbose:
                print('# TT-PIVMGS:    ortho', ortho, 'pivot', pivot, 'dot', tmp)
            w_nrm = w_nrm * pitts_py.axpby(-tmp, V[pivot], 1, w, rankTolerance, maxRank)
            Vtw[pivot] = 0

        Vtw = [pitts_py.dot(v,w) for v in V]
        maxErr = np.max(np.abs(Vtw))
        if verbose:
            print('# TT-PIVMGS:    ortho max err', maxErr)
        if maxErr < rankTolerance:
            break

    V += [w]
    return np.array(h + [w_nrm,])


if __name__ == '__main__':
    pitts_py.initialize()

    dims = [40,]*5

    w_ref = pitts_py.TensorTrain_double(dims)
    Vh = pitts_py.TensorTrain_double(dims)
    eps = 1.e-8
    rmax = 999

    V = list()
    for i in range(20):
        w = pitts_py.TensorTrain_double(dims)
        w.setTTranks(2)
        pitts_py.randomize(w)
        pitts_py.copy(w, w_ref)

        h = tt_pivmgs(V, w, eps, rmax)

        # check that all vectors are ortho-normal
        VtV = np.array([[pitts_py.dot(v,w) for v in V] for w in V])
        VtV_error = VtV - np.eye(i+1)
        print('ortho error:', np.max(VtV_error))
        assert(np.all( VtV_error < 10*eps ))

        # check that V*h = w_ref
        Vh.setZero()
        Vh_nrm = 0
        for j in range(i+1):
            Vh_nrm = pitts_py.axpby(h[j], V[j], Vh_nrm, Vh, eps / 100, rmax)
        Vh_error = pitts_py.axpby(-1, w_ref, Vh_nrm, Vh, eps / 100, rmax)
        wNrm = pitts_py.norm2(w_ref)
        print('V*h=w error:', Vh_error/wNrm)
        assert(Vh_error/wNrm < 10*eps)

    pitts_py.finalize()

