#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for the PITTS::TensorTrainOperator python binding"""

__all__ = ['TestTensorTrainOperator']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2021-02-12'
import pitts_py
import numpy as np
import unittest


class TestTensorTrainOperator(unittest.TestCase):
    """TestCase for pitts_py.TensorTrainOperator_... classes"""

    def test_createTensorTrainOperator_double(self):
        ttOp = pitts_py.TensorTrainOperator_double([3, 4, 5], [2, 3, 4])
        self.assertEqual([3, 4, 5], ttOp.row_dimensions())
        self.assertEqual([2, 3, 4], ttOp.col_dimensions())
        self.assertEqual([1, 1], ttOp.getTTranks())

    def test_createTensorTrainOperator_float(self):
        ttOp = pitts_py.TensorTrainOperator_float([3, 4, 5], [2, 3, 4])
        self.assertEqual([3, 4, 5], ttOp.row_dimensions())
        self.assertEqual([2, 3, 4], ttOp.col_dimensions())
        self.assertEqual([1, 1], ttOp.getTTranks())

    def test_create_dimensionMismatch(self):
        with self.assertRaises(ValueError):
            ttOp = pitts_py.TensorTrainOperator_double([2,4,3],[2,2])

    def test_setGetTTranks(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,2,2,2,2],[2,2,2,2,2])
        self.assertEqual([1,1,1,1], ttOp.getTTranks())
        ttOp.setTTranks([1,3,5,2])
        self.assertEqual([1,3,5,2], ttOp.getTTranks())

    def test_setZero(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,3,4],[3,4,2])
        ttOp.setTTranks([2,4])
        ttOp.setZero()
        self.assertEqual([1,1], ttOp.getTTranks())
        np.testing.assert_array_almost_equal(np.zeros((1,2,3,1)), ttOp.getSubTensor(0))
        np.testing.assert_array_almost_equal(np.zeros((1,3,4,1)), ttOp.getSubTensor(1))
        np.testing.assert_array_almost_equal(np.zeros((1,4,2,1)), ttOp.getSubTensor(2))

    def test_setZero(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,3,4],[3,4,2])
        ttOp.setTTranks([2,4])
        ttOp.setOnes()
        self.assertEqual([1,1], ttOp.getTTranks())
        np.testing.assert_array_almost_equal(np.ones((1,2,3,1)), ttOp.getSubTensor(0))
        np.testing.assert_array_almost_equal(np.ones((1,3,4,1)), ttOp.getSubTensor(1))
        np.testing.assert_array_almost_equal(np.ones((1,4,2,1)), ttOp.getSubTensor(2))

    def test_setEye(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,3,4],[3,4,2])
        ttOp.setTTranks([2,4])
        ttOp.setEye()
        self.assertEqual([1,1], ttOp.getTTranks())
        np.testing.assert_array_almost_equal(np.eye(2,3), ttOp.getSubTensor(0)[0,:,:,0])
        np.testing.assert_array_almost_equal(np.eye(3,4), ttOp.getSubTensor(1)[0,:,:,0])
        np.testing.assert_array_almost_equal(np.eye(4,2), ttOp.getSubTensor(2)[0,:,:,0])

    def test_copy(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,4,3], [4,3,2])
        ttOp2 = pitts_py.TensorTrainOperator_double([2,4,3], [4,3,2])
        ttOp.setOnes()
        pitts_py.copy(ttOp, ttOp2)
        ttOp.setZero()
        np.testing.assert_array_almost_equal(np.ones((1,2,4,1)), ttOp2.getSubTensor(0))
        np.testing.assert_array_almost_equal(np.ones((1,4,3,1)), ttOp2.getSubTensor(1))
        np.testing.assert_array_almost_equal(np.ones((1,3,2,1)), ttOp2.getSubTensor(2))

    def test_copy_dimensionMismatch(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,4,3],[2,2,2])
        ttOp2 = pitts_py.TensorTrainOperator_double([2,4,2],[2,2,2])
        with self.assertRaises(ValueError):
            pitts_py.copy(ttOp, ttOp2)

    def test_randomize(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,4,2],[2,3,3])
        self.assertEqual([1,1], ttOp.getTTranks())

        pitts_py.randomize(ttOp)
        self.assertEqual([1,1], ttOp.getTTranks())

        unique_numbers = np.unique(ttOp.getSubTensor(0))
        unique_numbers = np.append(unique_numbers, np.unique(ttOp.getSubTensor(1)))
        unique_numbers = np.append(unique_numbers, np.unique(ttOp.getSubTensor(2)))
        unique_numbers = np.unique(unique_numbers)
        self.assertEqual(2*2 + 4*3 + 2*3, unique_numbers.size)

        ttOp.setTTranks([2,3])
        pitts_py.randomize(ttOp)
        self.assertEqual([2,3], ttOp.getTTranks())

        unique_numbers = np.append(unique_numbers, np.unique(ttOp.getSubTensor(0)))
        unique_numbers = np.append(unique_numbers, np.unique(ttOp.getSubTensor(1)))
        unique_numbers = np.append(unique_numbers, np.unique(ttOp.getSubTensor(2)))
        self.assertEqual(2*2+4*3+2*3 + 1*2*2*2 + 2*4*3*3 + 3*2*3*1, unique_numbers.size)

    def test_apply_zero(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,3,2],[3,4,1])
        ttX = pitts_py.TensorTrain_double([3,4,1])
        ttY = pitts_py.TensorTrain_double([2,3,2])

        ttOp.setZero()
        pitts_py.randomize(ttX)
        pitts_py.apply(ttOp, ttX, ttY)
        self.assertEqual(0, pitts_py.norm2(ttY))

    def test_apply_identity(self):
        ttOp = pitts_py.TensorTrainOperator_double([5,3,3],[5,3,3])
        ttX = pitts_py.TensorTrain_double([5,3,3])
        ttY = pitts_py.TensorTrain_double([5,3,3])

        ttOp.setEye()
        pitts_py.randomize(ttX)
        pitts_py.applyT(ttOp, ttX, ttY)
        err = pitts_py.axpby(-1, ttX, 1, ttY)
        self.assertLess(err, 1.e-8)

    def test_applyT_zero(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,3,2],[3,4,1])
        ttX = pitts_py.TensorTrain_double([2,3,2])
        ttY = pitts_py.TensorTrain_double([3,4,1])

        ttOp.setZero()
        pitts_py.randomize(ttX)
        pitts_py.applyT(ttOp, ttX, ttY)
        self.assertEqual(0, pitts_py.norm2(ttY))

    def test_applyT_identity(self):
        ttOp = pitts_py.TensorTrainOperator_double([5,3,3],[5,3,3])
        ttX = pitts_py.TensorTrain_double([5,3,3])
        ttY = pitts_py.TensorTrain_double([5,3,3])

        ttOp.setEye()
        pitts_py.randomize(ttX)
        pitts_py.apply(ttOp, ttX, ttY)
        err = pitts_py.axpby(-1, ttX, 1, ttY)
        self.assertLess(err, 1.e-8)

    def test_axpby(self):
        ttOp1 = pitts_py.TensorTrainOperator_double([2,3,4],[3,2,2])
        ttOp1.setTTranks(2)
        ttOp2 = pitts_py.TensorTrainOperator_double([2,3,4],[3,2,2])

        pitts_py.randomize(ttOp1)
        pitts_py.randomize(ttOp2)

        ttOp12 = pitts_py.TensorTrainOperator_double([2,3,4],[3,2,2])
        pitts_py.copy(ttOp2, ttOp12)
        pitts_py.axpby(0.33, ttOp1, -0.97, ttOp12)
        

        ttX = pitts_py.TensorTrain_double([3,2,2])
        ttX.setTTranks(2)
        pitts_py.randomize(ttX)

        ttY = pitts_py.TensorTrain_double([2,3,4])
        pitts_py.apply(ttOp12, ttX, ttY)

        ttY1 = pitts_py.TensorTrain_double([2,3,4])
        pitts_py.apply(ttOp1, ttX, ttY1)

        ttY2 = pitts_py.TensorTrain_double([2,3,4])
        pitts_py.apply(ttOp2, ttX, ttY2)

        ttY12 = pitts_py.TensorTrain_double([2,3,4])
        pitts_py.copy(ttY2, ttY12)
        nrm = pitts_py.axpby(0.33, ttY1, -0.97, ttY12)

        err = pitts_py.axpby(-nrm, ttY12, 1., ttY)
        self.assertLess(err, 1.e-8)

    def test_apply_op(self):
        ttOpA = pitts_py.TensorTrainOperator_double([2,3,2],[3,4,1])
        ttOpB = pitts_py.TensorTrainOperator_double([3,4,1],[3,2,3])
        ttOpC = pitts_py.TensorTrainOperator_double([2,3,2],[3,2,3])
        ttX = pitts_py.TensorTrain_double([3,2,3])
        ttY = pitts_py.TensorTrain_double([3,4,1])
        ttZ = pitts_py.TensorTrain_double([2,3,2])
        ttZ_ref = pitts_py.TensorTrain_double([2,3,2])

        pitts_py.randomize(ttOpA)
        pitts_py.randomize(ttOpB)
        pitts_py.randomize(ttX)

        pitts_py.apply(ttOpA, ttOpB, ttOpC)

        pitts_py.apply(ttOpC, ttX, ttZ)
        
        pitts_py.apply(ttOpB, ttX, ttY)
        pitts_py.apply(ttOpA, ttY, ttZ_ref)

        error = pitts_py.axpby(1., ttZ_ref, -1., ttZ)
        self.assertLess(error, 1.e-8)

    def test_axpby_dimensionMismatch(self):
        ttOp = pitts_py.TensorTrainOperator_double([2,4,3],[2,2,2])
        ttOp2 = pitts_py.TensorTrainOperator_double([2,4,2],[2,2,2])
        with self.assertRaises(ValueError):
            pitts_py.axpby(1., ttOp, 2., ttOp2)

    def test_setSubTensor_invalidShape(self):
        ttOp = pitts_py.TensorTrainOperator_double([3,2,5],[3,3,5])
        with self.assertRaises(IndexError):
            ttOp.setSubTensor(10, np.ones((1,3,5,1)))
        with self.assertRaises(ValueError):
            ttOp.setSubTensor(0, np.ones((1,3,5,1)))
        ttOp.setSubTensor(0, np.ones((1,3,3,1)))
        with self.assertRaises(ValueError):
            ttOp.setSubTensor(1, np.ones((1,3,3,1)))

    def test_setGetSubTensor(self):
        ttOp = pitts_py.TensorTrainOperator_double([3,2,5],[2,3,2])
        ttOp.setTTranks([2,3])
        pitts_py.randomize(ttOp)
        t1_ref = np.random.rand(1,3,2,2)
        t2_ref = np.random.rand(2,2,3,3)
        t3_ref = np.random.rand(3,5,2,1)
        ttOp.setSubTensor(0, t1_ref)
        ttOp.setSubTensor(1, t2_ref)
        ttOp.setSubTensor(2, t3_ref)
        np.testing.assert_array_almost_equal(t1_ref, ttOp.getSubTensor(0))
        np.testing.assert_array_almost_equal(t2_ref, ttOp.getSubTensor(1))
        np.testing.assert_array_almost_equal(t3_ref, ttOp.getSubTensor(2))

    def test_setGetSubTensor_large(self):
        ttOp = pitts_py.TensorTrainOperator_double([10,20,15],[10,15,10])
        ttOp.setTTranks([2,3])
        pitts_py.randomize(ttOp)
        t1_ref = np.random.rand(1,10,10,2)
        t2_ref = np.random.rand(2,20,15,3)
        t3_ref = np.random.rand(3,15,10,1)
        ttOp.setSubTensor(0, t1_ref)
        ttOp.setSubTensor(1, t2_ref)
        ttOp.setSubTensor(2, t3_ref)
        np.testing.assert_array_almost_equal(t1_ref, ttOp.getSubTensor(0))
        np.testing.assert_array_almost_equal(t2_ref, ttOp.getSubTensor(1))
        np.testing.assert_array_almost_equal(t3_ref, ttOp.getSubTensor(2))


if __name__ == '__main__':
    unittest.main()

