#!/usr/bin/env python3

"""
Laplace-operator in tensor-train format
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>',
               'Rebekka-Sarah Hennig <Rebekka-Sarah.Hennig@dlr.de>']
__date__ = '2021-07-29'

import numpy as np
import pitts_py


def LaplaceOperator(dims, mask=None):
    TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOp.setZero()
    TTOp_dummy = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOp_dummy.setEye()
    for iDim in range(len(dims)):
        n_i = dims[iDim]
        h = 1 / (n_i+1)
        eye_i = TTOp_dummy.getSubTensor(iDim)
        tridi_i = np.zeros((n_i,n_i))
        for i in range(n_i):
            if mask is not None and not mask(iDim, i):
                continue
            tridi_i[i,i] = 2. / h**2
            if i > 0:
                tridi_i[i,i-1] = -1. / h**2
            if i+1 < n_i:
                tridi_i[i,i+1] = -1. / h**2
        TTOp_dummy.setSubTensor(iDim, tridi_i.reshape(1,n_i,n_i,1))
        pitts_py.axpby(1, TTOp_dummy, 1, TTOp)
        TTOp_dummy.setSubTensor(iDim, eye_i)
    return TTOp


if __name__ == '__main__':
    pitts_py.initialize()

    np.set_printoptions(linewidth=200)

    TTOp = LaplaceOperator([10,]*5)
    TTOp_nrm = pitts_py.normalize(TTOp)
    print('TTOp norm:', TTOp_nrm, 'rank:', TTOp.getTTranks())

    TTOp = LaplaceOperator([10,]*1, lambda iDim, i: i < 5)
    TTOp_nrm = pitts_py.normalize(TTOp)
    print('TTOp norm:', TTOp_nrm, 'rank:', TTOp.getTTranks())
    print(TTOp.getSubTensor(0).reshape([10,10], order='F'))

    pitts_py.finalize()
