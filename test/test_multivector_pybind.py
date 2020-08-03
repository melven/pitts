#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for the PITTS::MultiVector python binding"""

__all__ = ['TestMultiVector']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-07-16'
import pitts_py
import numpy as np
import unittest


class TestMultiVector(unittest.TestCase):
    """TestCase for pitts_py.MultiVector_... classes"""

    def test_createMultiVector_double(self):
        mv = pitts_py.MultiVector_double(3, 4)
        self.assertEqual(3, mv.rows())
        self.assertEqual(4, mv.cols())

    def test_createMultiVector_float(self):
        mv = pitts_py.MultiVector_float(3, 4)
        self.assertEqual(3, mv.rows())
        self.assertEqual(4, mv.cols())

    def test_resize(self):
        mv = pitts_py.MultiVector_double();
        self.assertEqual(0, mv.rows());
        self.assertEqual(0, mv.cols());
        mv.resize(3, 0)
        self.assertEqual(3, mv.rows());
        self.assertEqual(0, mv.cols());
        mv.resize(4, 7)
        self.assertEqual(4, mv.rows());
        self.assertEqual(7, mv.cols());

    def test_bufferProtocol(self):
        mv = pitts_py.MultiVector_double(3, 4);
        # warning memory was not necessarily initialized!
        mv_view = np.array(mv, copy=False)
        self.assertEqual((3, 4), mv_view.shape)
        mv_view[...] = 1.
        mv_view2 = np.array(mv, copy=False)
        np.testing.assert_array_almost_equal(np.ones([3, 4]), mv_view2)

    def test_copy(self):
        mv = pitts_py.MultiVector_double(20, 5)
        mv_view = np.array(mv, copy=False)
        mv_view[...] = np.random.rand(20, 5)

        mw = pitts_py.MultiVector_double()
        print(mw)
        print(mv)
        pitts_py.copy(mv, mw)
        mw_view = np.array(mw, copy=False)
        np.testing.assert_array_almost_equal(mv_view, mw_view)

    def test_randomize(self):
        mv = pitts_py.MultiVector_double(20, 5)

        pitts_py.randomize(mv)
        self.assertEqual(20, mv.rows())
        self.assertEqual(5, mv.cols())
        mv1 = np.array(mv)

        pitts_py.randomize(mv)
        self.assertEqual(20, mv.rows())
        self.assertEqual(5, mv.cols())
        mv2 = np.array(mv)

        self.assertTrue(np.linalg.norm(mv1 - mv2) > 1.e-4)

    def test_cdist2(self):
        X = pitts_py.MultiVector_double(20,5)
        Y = pitts_py.MultiVector_double(20,3)
        pitts_py.randomize(X)
        pitts_py.randomize(Y)
        dist2 = pitts_py.cdist2(X, Y)
        dist_ref = np.zeros([5,3])
        X = np.array(X)
        Y = np.array(Y)
        for i in range(5):
            for j in range(3):
                dist_ref[i,j] = np.linalg.norm(X[:,i] - Y[:,j])
        dist = np.sqrt(dist2)
        np.testing.assert_array_almost_equal(dist_ref, dist)


if __name__ == '__main__':
    unittest.main()

