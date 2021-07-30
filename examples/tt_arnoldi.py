#!/usr/bin/env python3

"""
Tensor-train matrix arnoldi (just for looking at ranks)
"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>',
               'Rebekka-Sarah Hennig <Rebekka-Sarah.Hennig@dlr.de>']
__date__ = '2021-06-04'

import numpy as np
import pitts_py
from tt_laplace_operator import LaplaceOperator
from tt_convection_operator import ConvectionOperator


if __name__ == '__main__':
    pitts_py.initialize()

    #TTOp = pitts_py.TensorTrainOperator_double([50,]*5, [50,]*5)
    #TTOp.setTTranks(2)
    #pitts_py.randomize(TTOp)
    #pitts_py.normalize(TTOp)
    #TTOpEye = pitts_py.TensorTrainOperator_double([50,]*5, [50,]*5)
    #TTOpEye.setEye()
    #pitts_py.axpby(1, TTOpEye, 0.1, TTOp)

    TTOp = LaplaceOperator([50,]*5)
    pitts_py.axpby(0.1, ConvectionOperator([50,]*5), 1, TTOp)
    print("Operator max. rank: %d" %(np.max(TTOp.getTTranks())))

    x = pitts_py.TensorTrain_double(TTOp.row_dimensions())
    x.setTTranks(1)
    pitts_py.randomize(x)
    d = len(TTOp.row_dimensions())

    eps = 1.e-8
    V = list()
    pitts_py.normalize(x, eps)
    V += [x,]
    r_x = np.max(x.getTTranks())
    print("x: initial max. rank: %d" % r_x)
    for j in range(30):
        w = pitts_py.TensorTrain_double(TTOp.row_dimensions())
        pitts_py.apply(TTOp, V[-1], w)
        for ortho in range(3):
            for vi in V:
                hij = pitts_py.dot(vi, w)
                if np.abs(hij) > eps:
                    pitts_py.axpby(-hij, vi, 1, w, eps/100)
        pitts_py.normalize(w, eps)
        V += [w,]
        r_w = np.max(w.getTTranks())
        print("iteration %d, new vector max. rank: %d (%g)" %(j, r_w, r_w/r_x/(j+2)))

    pitts_py.finalize()

