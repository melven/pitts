#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
#                    German Aerospace Center
#
"""Tests for PITTS::QubitSimulator python binding"""

__all__ = ['TestTensorTrain']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-06-26'
import pitts_py
from math import sqrt
import unittest


class TestQubitSimulator(unittest.TestCase):
    """TestCase for pitts_py.QubitSimulator"""

    def test_simpleExample(self):
        qsim = pitts_py.QubitSimulator()

        hadamardGate = [[1/sqrt(2.),1/sqrt(2.)],[1/sqrt(2.),-1/sqrt(2.)]]
        cnotGate = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]

        for i in range(10):
          qsim.allocate_qubit(7)
          qsim.allocate_qubit(9)

          print(qsim.get_classical_value(7))

          qsim.apply_single_qubit_gate(9, hadamardGate)
          qsim.apply_two_qubit_gate(7, 9, cnotGate)

          print(qsim.measure_qubits((7,9)))

          qsim.deallocate_qubit(9)
          qsim.deallocate_qubit(7)

if __name__ == '__main__':
    unittest.main()
