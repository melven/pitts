#!/usr/bin/env python

# Copyright (C) 2022 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PITTS::TensorTrainOperator ITensor helper AutoMPO helper functions"""

__all__ = ['TestTensorTrainOperator_ITensor_AutoMPO']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-01-20'
import pitts_py
import numpy as np
import unittest


class TestTensorTrainOperator_ITensor_AutoMPO(unittest.TestCase):
    """TestCase for generating pitts_py.TensorTrainOperator_... with ITensor::Autompo"""

    def test_createSites_and_AutoMPO(self):
        siteSet = pitts_py.itensor.SpinHalf(7)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

        siteSet = pitts_py.itensor.SpinOne(3)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

        siteSet = pitts_py.itensor.Boson(20)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

        siteSet = pitts_py.itensor.Fermion(13)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

        siteSet = pitts_py.itensor.Electron(17)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

        siteSet = pitts_py.itensor.tJ(7)
        autompo = pitts_py.itensor.AutoMPO(siteSet)

    def test_AutoMPO_accumulate(self):
        siteSet = pitts_py.itensor.SpinOne(10)
        ampo = pitts_py.itensor.AutoMPO(siteSet)
        
        # single site, coef=1
        ampo += 'S+',1

        # single site with coef!=1
        ampo += 3.2,'S-',2

        # two site with coef=1
        ampo += 'S+',3,'S-',2

        # two site with coef!=1
        ampo += 77.7,'Sz',3,'Sz',4

    def test_AutoMPO_with_spinHalfSite(self):
        siteSet = pitts_py.itensor.SpinHalf(7)

        ampo = pitts_py.itensor.AutoMPO(siteSet)
        for j in range(1, 7):
            ampo += 0.5,"S+",j,"S-",j+1
            ampo += 0.5,"S-",j,"S+",j+1
            ampo +=     "Sz",j,"Sz",j+1

        ttOp = pitts_py.itensor.toTTOp(ampo)

        self.assertIsInstance(ttOp, pitts_py.TensorTrainOperator_double)
        self.assertEqual([2,]*7, ttOp.row_dimensions())
        self.assertEqual([2,]*7, ttOp.col_dimensions())
        # we can currently not really test the result but it should have bond-dimension 5
        self.assertEqual([5,]*6, ttOp.getTTranks())


if __name__ == '__main__':
    unittest.main()

