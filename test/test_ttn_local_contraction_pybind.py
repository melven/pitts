#!/usr/bin/env python

# Copyright (C) 2026 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2026 German Aerospace Center (DLR), Institute of Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PITTS::ttn_local_gradient_contraction python binding"""

__all__ = ['TestTTNlocalGradientContraction']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2026-02-03'
import pitts_py
import numpy as np
from math import sqrt
import unittest


class TestTTNlocalGradientContraction(unittest.TestCase):
    """TestCase for pitts_py.ttn_local_gradient_contraction"""

    def np_einsum_comparison(self, b_left, b_right, b_top, n_samples, n_classes):
        envLeft = pitts_py.MultiVector_double(n_samples, b_left)
        pitts_py.randomize(envLeft)
        envRight = pitts_py.MultiVector_double(n_samples, b_right)
        pitts_py.randomize(envRight)
        envTop = pitts_py.MultiVector_double(n_samples, b_top*n_classes)
        pitts_py.randomize(envTop)
        optTensor = np.random.rand(b_left, b_right, b_top)

        gradOptTensor = pitts_py.ttn_local_gradient_contraction(optTensor, envLeft, envRight, envTop)

        envTop_t3 = np.reshape(envTop, [n_samples, n_classes, b_top], order='F')
        gradOptTensor_ref = np.einsum('ijb,ni,nj,nlb,nx,ny,nlc->xyc', optTensor, envLeft, envRight, envTop_t3, envLeft, envRight, envTop_t3)

        np.testing.assert_array_almost_equal(gradOptTensor_ref, gradOptTensor)

    def test_scalarExample(self):
        self.np_einsum_comparison(1, 1, 1, 1, 1)

    def test_almostScalarExample(self):
        self.np_einsum_comparison(2, 1, 1, 1, 1)
        self.np_einsum_comparison(1, 3, 1, 1, 1)
        self.np_einsum_comparison(1, 1, 4, 1, 1)
        self.np_einsum_comparison(1, 1, 1, 5, 1)
        self.np_einsum_comparison(1, 1, 1, 1, 6)

    def test_smallExample(self):
        self.np_einsum_comparison(10, 11, 12, 13, 15)

    def test_largerExample(self):
        self.np_einsum_comparison(7, 6, 5, 4, 100)


if __name__ == '__main__':
    unittest.main()
