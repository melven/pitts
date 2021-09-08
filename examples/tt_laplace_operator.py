#!/usr/bin/env python3

"""
Laplace-operator in tensor-train format
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>',
               'Rebekka-Sarah Hennig <Rebekka-Sarah.Hennig@dlr.de>']
__date__ = '2021-07-29'

import numpy as np
import pitts_py


def LaplaceOperator(dims):
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
            for j in range(n_i):
                if i == j:
                    tridi_i[i,j] = 2. / h**2
                elif i+1 == j or i-1 == j:
                    tridi_i[i,j] = -1. / h**2
                else:
                    tridi_i[i,j] = 0
        TTOp_dummy.setSubTensor(iDim, tridi_i.reshape(1,n_i,n_i,1))
        pitts_py.axpby(1, TTOp_dummy, 1, TTOp)
        TTOp_dummy.setSubTensor(iDim, eye_i)
    return TTOp


if __name__ == '__main__':
    pitts_py.initialize()

    TTOp = LaplaceOperator([10,]*5)
    TTOp_nrm = pitts_py.normalize(TTOp)
    print('TTOp norm:', TTOp_nrm, 'rank:', TTOp.getTTranks())

    pitts_py.finalize()
