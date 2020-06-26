#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for PITTS common functions python binding"""

__all__ = ['TestCommon']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-06-26'
import pitts_py
import unittest


class TestCommon(unittest.TestCase):
    """TestCase for pitts_py.initialize and pitts_py.finalize"""

    def test_example(self):
        pitts_py.initialize()

        # do something interesting...
        tt = pitts_py.TensorTrain_double([5,5,5])
        pitts_py.randomize(tt)

        pitts_py.finalize()


if __name__ == '__main__':
    unittest.main()
