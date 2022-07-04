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

    def test_setGetTTranks(self):
        tt = pitts_py.TensorTrain_double([2,2,2,2,2])
        self.assertEqual([1,1,1,1], tt.getTTranks())
        tt.setTTranks([1,3,5,2])
        self.assertEqual([1,3,5,2], tt.getTTranks())

    def test_from_to_dense_float_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float32)
        tt = pitts_py.fromDense_classical(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_float))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_tsqr_float_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float32)
        work = pitts_py.MultiVector_float()
        data = pitts_py.MultiVector_float(5*3*2, 3)
        data_view = np.array(data, copy=False)
        data_view[...] = fullTensor_ref.reshape([5*3*2, 3])
        data_view = None
        tt = pitts_py.fromDense(data, work, dimensions=[5, 3, 2, 3])
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_float))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_double_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float64)
        tt = pitts_py.fromDense_classical(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_double))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_tsqr_double_ones(self):
        fullTensor_ref = np.ones([5, 3, 2, 3], dtype=np.float64)
        work = pitts_py.MultiVector_double()
        data = pitts_py.MultiVector_double(5*3*2, 3)
        data_view = np.array(data, copy=False)
        data_view[...] = fullTensor_ref.reshape([5*3*2, 3])
        data_view = None
        tt = pitts_py.fromDense(data, work, dimensions=[5, 3, 2, 3])
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_double))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        self.assertEqual([1, 1, 1], tt.getTTranks())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor)

    def test_from_to_dense_float_random(self):
        fullTensor_ref = np.random.rand(5, 3, 2, 3).astype(dtype=np.float32)
        tt = pitts_py.fromDense_classical(fullTensor_ref)
        self.assertTrue(isinstance(tt, pitts_py.TensorTrain_float))
        self.assertEqual(list(fullTensor_ref.shape), tt.dimensions())
        fullTensor = pitts_py.toDense(tt)
        self.assertEqual(fullTensor_ref.dtype, fullTensor.dtype)
        np.testing.assert_array_almost_equal(fullTensor_ref, fullTensor, decimal=3)

    def test_from_to_dense_double_random(self):
        fullTensor_ref = np.random.rand(5, 3, 2, 3).astype(dtype=np.float64)
        tt = pitts_py.fromDense_classical(fullTensor_ref)
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

        t1 = tt.getSubTensor(0)
        t2 = tt.getSubTensor(1)
        t3 = tt.getSubTensor(2)
        r1 = tt.getTTranks()[0]
        r2 = tt.getTTranks()[1]
        t1_mat = t1.reshape([t1.size//r1, r1])
        t2_mat = t2.reshape([t2.size//r2, r2])
        t3_mat = t3.reshape([t3.size//1, 1])
        np.testing.assert_array_almost_equal(np.eye(r1,r1), np.dot(t1_mat.transpose(), t1_mat))
        np.testing.assert_array_almost_equal(np.eye(r2,r2), np.dot(t2_mat.transpose(), t2_mat))
        np.testing.assert_array_almost_equal(np.eye(1,1), np.dot(t3_mat.transpose(), t3_mat))

    def test_rightNormalize(self):
        tt = pitts_py.TensorTrain_double([2,5,3])
        tt.setTTranks([2,2])
        pitts_py.randomize(tt)

        norm_ref = pitts_py.norm2(tt)
        norm = pitts_py.rightNormalize(tt)
        np.testing.assert_almost_equal(norm_ref, norm)
        np.testing.assert_almost_equal(1., pitts_py.norm2(tt))

        t1 = tt.getSubTensor(0)
        t2 = tt.getSubTensor(1)
        t3 = tt.getSubTensor(2)
        r1 = tt.getTTranks()[0]
        r2 = tt.getTTranks()[1]
        t1_mat = t1.reshape([1, t1.size//1])
        t2_mat = t2.reshape([r1, t2.size//r1])
        t3_mat = t3.reshape([r2, t3.size//r2])
        np.testing.assert_array_almost_equal(np.eye(1,1), np.dot(t1_mat, t1_mat.transpose()))
        np.testing.assert_array_almost_equal(np.eye(r1,r1), np.dot(t2_mat, t2_mat.transpose()))
        np.testing.assert_array_almost_equal(np.eye(2,2), np.dot(t3_mat, t3_mat.transpose()))

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

        tt1 = pitts_py.fromDense_classical(fullTensor1)
        tt2 = pitts_py.fromDense_classical(fullTensor2)

        nrm2 = pitts_py.axpby(1.5, tt1, -0.75, tt2)

        result = nrm2 * pitts_py.toDense(tt2)

        np.testing.assert_almost_equal(1.5*fullTensor1 - 0.75*fullTensor2, result)

    def test_axpby_dimensionMismatch(self):
        tt = pitts_py.TensorTrain_double([2,4,3])
        tt2 = pitts_py.TensorTrain_double([2,4,2])
        with self.assertRaises(ValueError):
            nrm = pitts_py.axpby(1, tt, 2, tt2)

    def test_getSubTensor_zeros(self):
        tt = pitts_py.TensorTrain_double([3,2,5])
        tt.setZero()
        t1 = tt.getSubTensor(0)
        t2 = tt.getSubTensor(1)
        t3 = tt.getSubTensor(2)
        np.testing.assert_array_almost_equal(np.zeros([1,3,1]), t1)
        np.testing.assert_array_almost_equal(np.zeros([1,2,1]), t2)
        np.testing.assert_array_almost_equal(np.zeros([1,5,1]), t3)

    def test_getSubTensor_unit(self):
        tt = pitts_py.TensorTrain_double([3,2,5])
        tt.setUnit([1,0,2])
        t1 = tt.getSubTensor(0)
        t2 = tt.getSubTensor(1)
        t3 = tt.getSubTensor(2)
        np.testing.assert_array_almost_equal([[[0],[1],[0]]], t1)
        np.testing.assert_array_almost_equal([[[1],[0]]], t2)
        np.testing.assert_array_almost_equal([[[0],[0],[1],[0],[0]]], t3)

    def test_setSubTensor_invalidShape(self):
        tt = pitts_py.TensorTrain_double([3,2,5])
        with self.assertRaises(IndexError):
            tt.setSubTensor(10, [[[1],[2],[3]]])
        with self.assertRaises(ValueError):
            tt.setSubTensor(0, [1,2,3])
        tt.setSubTensor(0, [[[1],[2],[3]]])
        with self.assertRaises(ValueError):
            tt.setSubTensor(1, [[[1],[2],[3]]])

    def test_setGetSubTensor(self):
        tt = pitts_py.TensorTrain_double([3,2,5])
        tt.setTTranks([2,3])
        pitts_py.randomize(tt)
        t1_ref = np.array([1,2,3,4,5,6]).reshape([1,3,2])
        t2_ref = np.array([10,11,12,13,14,15,16,17,18,19,20,21]).reshape([2,2,3])
        t3_ref = np.array([101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]).reshape([3,5,1])
        tt.setSubTensor(0, t1_ref)
        tt.setSubTensor(1, t2_ref)
        tt.setSubTensor(2, t3_ref)
        np.testing.assert_array_almost_equal(t1_ref, tt.getSubTensor(0))
        np.testing.assert_array_almost_equal(t2_ref, tt.getSubTensor(1))
        np.testing.assert_array_almost_equal(t3_ref, tt.getSubTensor(2))

    def test_setGetSubTensor_large(self):
        tt = pitts_py.TensorTrain_double([50,100,20])
        tt.setTTranks([2,3])
        pitts_py.randomize(tt)
        t1_ref = np.random.rand(1,50,2)
        t2_ref = np.random.rand(2,100,3)
        t3_ref = np.random.rand(3,20,1)
        tt.setSubTensor(0, t1_ref)
        tt.setSubTensor(1, t2_ref)
        tt.setSubTensor(2, t3_ref)
        np.testing.assert_array_almost_equal(t1_ref, tt.getSubTensor(0))
        np.testing.assert_array_almost_equal(t2_ref, tt.getSubTensor(1))
        np.testing.assert_array_almost_equal(t3_ref, tt.getSubTensor(2))
    
    def test_gramSchmidt(self):
        V = list()
        H = np.zeros([5,5])
        X = np.zeros([5*5*5,5])
        Q = np.zeros([5*5*5,5])
        for i in range(5):
            w = pitts_py.TensorTrain_double([5,5,5])
            w.setTTranks(3);
            pitts_py.randomize(w)

            X[:,i] = pitts_py.toDense(w).reshape([5*5*5], order='F')

            H[0:i+1,i] = pitts_py.gramSchmidt(V, w)

            self.assertEqual(i+1, len(V))
            Q[:,i] = pitts_py.toDense(V[i]).reshape([5*5*5], order='F')
        
        I = np.eye(5,5)
        QtQ = Q.T @ Q
        np.testing.assert_array_almost_equal(I, QtQ)

        QH = Q @ H
        np.testing.assert_array_almost_equal(X, QH)

    
if __name__ == '__main__':
    unittest.main()

