#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for the PITTS::TensorTrain python binding"""

__all__ = ['TestTensorTrain']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-06-26'
import pitts_py
import numpy as np
import unittest


class TestTensorTrain(unittest.TestCase):
    """TestCase for pitts_py.TensorTrain_... classes"""

    def test_createTensorTrain_double(self):
        tt = pitts_py.TensorTrain_double([3, 4, 5])
        self.assertEqual([3, 4, 5], tt.dimensions())
        self.assertEqual([1, 1], tt.getTTranks())

    def test_createTensorTrain_float(self):
        tt = pitts_py.TensorTrain_float([3, 4, 5])
        self.assertEqual([3, 4, 5], tt.dimensions())
        self.assertEqual([1, 1], tt.getTTranks())

    def test_set_get_TTranks(self):
        tt = pitts_py.TensorTrain_double([2,2,2,2,2])
        self.assertEqual([1,1,1,1], tt.getTTranks())
        tt.setTTranks([1,3,5,2])
        self.assertEqual([1,3,5,2], tt.getTTranks())

    def test_from_to_dense_float_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float32)
        tt = pitts_py.fromDense(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_float))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_double_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float64)
        tt = pitts_py.fromDense(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_double))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_float_random(self):
        fullTensor_ref = np.random.rand(5, 3, 2, 3).astype(dtype=np.float32)
        tt = pitts_py.fromDense(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_float))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor, decimal=3)

    def test_from_to_dense_double_random(self):
        fullTensor_ref = np.random.rand(5, 3, 2, 3).astype(dtype=np.float64)
        tt = pitts_py.fromDense(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_double))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_setZero(self):
        tt = pitts_py.TensorTrain_double([2,3,4])
        tt.setTTranks([2,4])
        tt.setZero()
        self.assertEqual([1,1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        np.testing.assert_array_almost_equal(np.zeros([2,3,4]), fullTensor)

    def test_setOnes(self):
        tt = pitts_py.TensorTrain_double([2,3,4])
        tt.setTTranks([2,4])
        tt.setOnes()
        self.assertEqual([1,1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        np.testing.assert_array_almost_equal(np.ones([2,3,4]), fullTensor)

    def test_setUnit(self):
        tt = pitts_py.TensorTrain_double([2,3,4])
        tt.setTTranks([2,4])
        tt.setUnit([1,0,3])
        self.assertEqual([1,1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        fullTensor_ref = np.zeros([2,3,4])
        fullTensor_ref[1,0,3] = 1
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_copy(self):
        tt = pitts_py.TensorTrain_double([2,4,3])
        tt2 = pitts_py.TensorTrain_double([2,4,3])
        tt.setUnit([0,1,2])
        pitts_py.copy(tt, tt2)
        tt.setUnit([1,0,0])
        fullTensor = pitts_py.toDense(tt2)
        fullTensor_ref = np.zeros([2,4,3])
        fullTensor_ref[0,1,2] = 1
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_copy_dimensionMismatch(self):
        tt = pitts_py.TensorTrain_double([2,4,3])
        tt2 = pitts_py.TensorTrain_double([2,4,2])
        with self.assertRaises(ValueError):
            pitts_py.copy(tt, tt2)

    def test_randomize(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        self.assertEqual([1,1], tt.getTTranks())

        fullTensor1 = pitts_py.toDense(tt)

        pitts_py.randomize(tt)
        self.assertEqual([1,1], tt.getTTranks())
        fullTensor2 = pitts_py.toDense(tt)

        pitts_py.randomize(tt)
        fullTensor3 = pitts_py.toDense(tt)

        tt.setTTranks([2,3])
        pitts_py.randomize(tt)
        self.assertEqual([2,3], tt.getTTranks())
        fullTensor4 = pitts_py.toDense(tt)

        # check for big enough differences...
        self.assertTrue(np.linalg.norm(fullTensor1 - fullTensor2) > 1.e-4)
        self.assertTrue(np.linalg.norm(fullTensor2 - fullTensor3) > 1.e-4)
        self.assertTrue(np.linalg.norm(fullTensor3 - fullTensor4) > 1.e-4)

    def test_dot(self):
        tt1 = pitts_py.TensorTrain_double([2,5,3])
        tt2 = pitts_py.TensorTrain_double([2,5,3])

        tt1.setUnit([0,1,2])
        tt2.setUnit([0,2,2])

        np.testing.assert_almost_equal(1., pitts_py.dot(tt1, tt1))
        np.testing.assert_almost_equal(0., pitts_py.dot(tt1, tt2))
        np.testing.assert_almost_equal(1., pitts_py.dot(tt2, tt2))

    def test_norm(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        tt.setTTranks([2,2])
        pitts_py.randomize(tt)

        np.testing.assert_almost_equal(np.sqrt(pitts_py.dot(tt,tt)), pitts_py.norm2(tt))

    def test_leftNormalize(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        tt.setTTranks([2,2])
        pitts_py.randomize(tt)

        norm_ref = pitts_py.norm2(tt)
        norm = pitts_py.leftNormalize(tt)
        np.testing.assert_almost_equal(norm_ref, norm)
        np.testing.assert_almost_equal(1., pitts_py.norm2(tt))

    def test_rightNormalize(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        tt.setTTranks([2,2])
        pitts_py.randomize(tt)

        norm_ref = pitts_py.norm2(tt)
        norm = pitts_py.rightNormalize(tt)
        np.testing.assert_almost_equal(norm_ref, norm)
        np.testing.assert_almost_equal(1., pitts_py.norm2(tt))

    def test_normalize(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        tt.setTTranks([2,2])
        pitts_py.randomize(tt)

        norm_ref = pitts_py.norm2(tt)
        norm = pitts_py.normalize(tt)
        np.testing.assert_almost_equal(norm_ref, norm)
        np.testing.assert_almost_equal(1., pitts_py.norm2(tt))

    def test_axpby(self):
        fullTensor1 = np.random.rand(2,4,3)
        fullTensor2 = np.random.rand(2,4,3)

        tt1 = pitts_py.fromDense(fullTensor1)
        tt2 = pitts_py.fromDense(fullTensor2)

        nrm2 = pitts_py.axpby(1.5, tt1, -0.75, tt2)

        result = nrm2 * pitts_py.toDense(tt2)

        np.testing.assert_almost_equal(1.5*fullTensor1 - 0.75*fullTensor2, result)


if __name__ == '__main__':
    unittest.main()

