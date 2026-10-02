#!/usr/bin/env python

# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PITTS::applyElementwise (TT-cross) python binding"""

__all__ = ['TestTensorTrainCross']
__authors__ = ['Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2026-09-12'
import pitts_py
import numpy as np
import unittest


def positive_random_tt(dimensions, ranks, seed, dtype=np.float64):
    """random tensor train with all entries positive (cores with entries in (0.5,1.5))"""
    tt = pitts_py.TensorTrain_double(dimensions) if dtype == np.float64 else pitts_py.TensorTrain_float(dimensions)
    tt.setTTranks(ranks)
    rng = np.random.default_rng(seed)
    for i in range(len(dimensions)):
        subT = tt.getSubTensor(i)
        tt.setSubTensor(i, rng.uniform(0.5, 1.5, subT.shape).astype(dtype))
    return tt


class TestTensorTrainCross(unittest.TestCase):
    """TestCase for pitts_py.applyElementwise"""

    def test_square_ufunc(self):
        tt_x = positive_random_tt([4, 3, 5, 3], [2, 3, 2], 42)
        tt_y = pitts_py.TensorTrain_double([4, 3, 5, 3])
        pitts_py.applyElementwise(np.square, tt_x, tt_y, rankTolerance=1.e-8, nSweeps=20)
        np.testing.assert_array_almost_equal(pitts_py.toDense(tt_x)**2, pitts_py.toDense(tt_y))

    def test_inplace_function(self):
        """the function may modify the array in-place and return None"""
        tt_x = positive_random_tt([3, 4, 3], [2, 2], 7)
        tt_y = pitts_py.TensorTrain_double([3, 4, 3])

        def sqrt_inplace(values):
            np.sqrt(values, out=values)

        pitts_py.applyElementwise(sqrt_inplace, tt_x, tt_y, rankTolerance=1.e-8, nSweeps=20)
        np.testing.assert_array_almost_equal(np.sqrt(pitts_py.toDense(tt_x)), pitts_py.toDense(tt_y))

    def test_scalar_function_vectorized(self):
        """scalar functions are supported through numpy.vectorize"""
        tt_x = positive_random_tt([3, 3, 3], [2, 2], 11)
        tt_y = pitts_py.TensorTrain_double([3, 3, 3])
        pitts_py.applyElementwise(np.vectorize(lambda v: v*v), tt_x, tt_y, rankTolerance=1.e-8, nSweeps=20)
        np.testing.assert_array_almost_equal(pitts_py.toDense(tt_x)**2, pitts_py.toDense(tt_y))

    def test_float(self):
        tt_x = positive_random_tt([3, 4, 3], [2, 2], 5, dtype=np.float32)
        tt_y = pitts_py.TensorTrain_float([3, 4, 3])
        seen_dtype = []

        def square(values):
            seen_dtype.append(values.dtype)
            return values**2

        pitts_py.applyElementwise(square, tt_x, tt_y, rankTolerance=1.e-6, nSweeps=20)
        self.assertEqual([np.dtype(np.float32)], list(set(seen_dtype)))
        np.testing.assert_allclose(pitts_py.toDense(tt_x)**2, pitts_py.toDense(tt_y), rtol=1.e-4)

    def test_options(self):
        """options are passed as keyword arguments, e.g. the DMRG-cross variant"""
        tt_x = positive_random_tt([3, 4, 3, 2], [2, 2, 2], 13)
        tt_y = pitts_py.TensorTrain_double([3, 4, 3, 2])
        pitts_py.applyElementwise(np.reciprocal, tt_x, tt_y, nMALS=2, rankTolerance=1.e-8, maxRank=20, nSweeps=20)
        np.testing.assert_array_almost_equal(1/pitts_py.toDense(tt_x), pitts_py.toDense(tt_y))

    def test_wrong_result_size(self):
        tt_x = positive_random_tt([3, 4, 3], [2, 2], 17)
        tt_y = pitts_py.TensorTrain_double([3, 4, 3])
        with self.assertRaises(ValueError):
            pitts_py.applyElementwise(lambda values: values[:-1], tt_x, tt_y)

    def test_exception_in_function(self):
        """an exception in the function is passed on to the caller"""
        tt_x = positive_random_tt([3, 4, 3], [2, 2], 19)
        tt_y = pitts_py.TensorTrain_double([3, 4, 3])

        def failing(values):
            raise RuntimeError("some error")

        with self.assertRaises(RuntimeError):
            pitts_py.applyElementwise(failing, tt_x, tt_y)

    def test_two_inputs_product(self):
        """element-wise product of two tensor trains with a numpy ufunc of two arguments"""
        tt_x1 = positive_random_tt([4, 3, 5, 3], [2, 3, 2], 23)
        tt_x2 = positive_random_tt([4, 3, 5, 3], [3, 1, 2], 29)
        tt_y = pitts_py.TensorTrain_double([4, 3, 5, 3])
        pitts_py.applyElementwise(np.multiply, tt_x1, tt_x2, tt_y, rankTolerance=1.e-8, nSweeps=20)
        np.testing.assert_array_almost_equal(pitts_py.toDense(tt_x1)*pitts_py.toDense(tt_x2), pitts_py.toDense(tt_y))

    def test_two_inputs_order_and_keywords(self):
        """the arrays are passed in the order of the tensors, also as keyword arguments; in-place result in the first one"""
        tt_x1 = positive_random_tt([3, 4, 3], [2, 2], 31, dtype=np.float32)
        tt_x2 = positive_random_tt([3, 4, 3], [2, 1], 37, dtype=np.float32)
        tt_y = pitts_py.TensorTrain_float([3, 4, 3])

        def divide_inplace(a, b):
            self.assertEqual(a.shape, b.shape)
            np.divide(a, b, out=a)

        pitts_py.applyElementwise(divide_inplace, TTx1=tt_x1, TTx2=tt_x2, TTy=tt_y, rankTolerance=1.e-6, nSweeps=20, nMALS=2)
        np.testing.assert_allclose(pitts_py.toDense(tt_x1)/pitts_py.toDense(tt_x2), pitts_py.toDense(tt_y), rtol=1.e-3)

    def test_two_inputs_mismatching_dimensions(self):
        tt_x1 = positive_random_tt([3, 4, 3], [2, 2], 41)
        tt_x2 = positive_random_tt([3, 4, 2], [2, 2], 43)
        tt_y = pitts_py.TensorTrain_double([3, 4, 3])
        with self.assertRaises(ValueError):
            pitts_py.applyElementwise(np.multiply, tt_x1, tt_x2, tt_y)

if __name__ == '__main__':
    unittest.main()
