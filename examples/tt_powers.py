#!/usr/bin/env python3

"""
Tensor-train matrix power times vector (just for looking at ranks)
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>',
               'Rebekka-Sarah Hennig <Rebekka-Sarah.Hennig@dlr.de>']
__date__ = '2021-06-01'

import numpy as np
import pitts_py
from tt_laplace_operator import LaplaceOperator


if __name__ == '__main__':
    pitts_py.initialize()

    #TTOp = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOp.setTTranks(5)
    #pitts_py.randomize(TTOp)
    #TTOpEye = pitts_py.TensorTrainOperator_double([2,3,3,2,4,10,7],[2,3,3,2,4,10,7])
    #TTOpEye.setEye()
    #pitts_py.axpby(1, TTOpEye, 0.1, TTOp)

    TTOp = LaplaceOperator([20,]*10)
    print("Operator max. rank: %d" %(np.max(TTOp.getTTranks())))

    x = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    x.setTTranks(5)
    pitts_py.randomize(x)
    y = pitts_py.TensorTrain_double(TTOp.row_dimensions())

    eps = 1.e-8
    for j in range(50):
        pitts_py.apply(TTOp, x, y)
        pitts_py.normalize(y, eps)
        print("A^j x: iteration %d, new vector max. rank: %d" %(j, np.max(y.getTTranks())))
        diff = pitts_py.axpby(-1, y, 1, x)
        print("difference x-y: %g" %(diff))

        # swap x and y
        tmp = x
        x = y
        y = tmp

    pitts_py.finalize()

