#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2022 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for the PITTS::TensorTrainOperator python binding"""

__all__ = ['TestTensorTrainOperator']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-07-06'
import pitts_py
import numpy as np
import unittest


class TestTensorTrain_solve(unittest.TestCase):
    """TestCase for pitts_py.solve... functions for linear systems in TT format"""

    def test_solveMALS_identity(self):
        ttOp = pitts_py.TensorTrainOperator_double([5,3,3],[5,3,3])
        ttX = pitts_py.TensorTrain_double([5,3,3])
        ttB = pitts_py.TensorTrain_double([5,3,3])

        ttOp.setEye()
        pitts_py.randomize(ttB)
        pitts_py.randomize(ttX)

        residualNorm = pitts_py.solveMALS(ttOp, True, ttB, ttX, 10)
        self.assertLess(residualNorm, 1.e-5)

        ttR = pitts_py.TensorTrain_double([5,3,3])
        pitts_py.apply(ttOp, ttX, ttR)
        residualNorm_ref = pitts_py.axpby(-1, ttB, 1, ttR)
        self.assertLess(residualNorm_ref, 1.e-5)

    def test_solveGMRES_identity(self):
        ttOp = pitts_py.TensorTrainOperator_double([5,3,3],[5,3,3])
        ttX = pitts_py.TensorTrain_double([5,3,3])
        ttB = pitts_py.TensorTrain_double([5,3,3])

        ttOp.setEye()
        pitts_py.randomize(ttB)
        pitts_py.randomize(ttX)

        residualNorm = pitts_py.solveGMRES(ttOp, ttB, ttX, 10, 1.e-8, 1.e-8, outputPrefix="test: ", verbose=True)
        self.assertLess(residualNorm, 1.e-5)

        ttR = pitts_py.TensorTrain_double([5,3,3])
        pitts_py.apply(ttOp, ttX, ttR)
        residualNorm_ref = pitts_py.axpby(-1, ttB, 1, ttR)
        self.assertLess(residualNorm_ref, 1.e-5)

    def test_solveMALS_identity_withTTgmres(self):
        ttOp = pitts_py.TensorTrainOperator_double([5,3,3],[5,3,3])
        ttX = pitts_py.TensorTrain_double([5,3,3])
        ttB = pitts_py.TensorTrain_double([5,3,3])

        ttOp.setEye()
        pitts_py.randomize(ttB)
        pitts_py.randomize(ttX)

        residualNorm = pitts_py.solveMALS(ttOp, True, ttB, ttX, 10, useTTgmres=True)
        self.assertLess(residualNorm, 1.e-5)

        ttR = pitts_py.TensorTrain_double([5,3,3])
        pitts_py.apply(ttOp, ttX, ttR)
        residualNorm_ref = pitts_py.axpby(-1, ttB, 1, ttR)
        self.assertLess(residualNorm_ref, 1.e-5)



if __name__ == '__main__':
    unittest.main()

