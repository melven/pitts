#!/usr/bin/env python

# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PITTS common functions python binding"""

__all__ = ['TestCommon']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-06-26'
import os
import pitts_py
import threading
import unittest


class TestCommon(unittest.TestCase):
    """TestCase for pitts_py.initialize and pitts_py.finalize"""

    # MPI can only be initialized/finalized once per process, so all tests share one scope
    @classmethod
    def setUpClass(cls):
        pitts_py.initialize()

    @classmethod
    def tearDownClass(cls):
        pitts_py.finalize(False)

    def test_example(self):
        # do something interesting...
        tt = pitts_py.TensorTrain_double([5,5,5])
        pitts_py.randomize(tt)

        pitts_py.clearPerformanceStatistics()
        pitts_py.randomize(tt)
        pitts_py.printPerformanceStatistics()

    def test_large_statistics_to_pipe(self):
        """printPerformanceStatistics must not block when stdout is a pipe drained by a python thread

        That is what jupyter does: ipykernel captures the raw file descriptors and reads them in a
        python thread, which needs the GIL. Writing more than the pipe buffer (64 KiB, easily reached
        by these statistics) while holding the GIL would deadlock the kernel, so the binding has to
        release it. A regression therefore hangs here instead of failing (caught by the ctest timeout).
        """
        # enough different kernel arguments to exceed the pipe buffer
        for n in range(4, 30):
            tt = pitts_py.TensorTrain_double([n, n, n])
            tt.setTTranks(min(n, 8))
            pitts_py.randomize(tt)
            pitts_py.normalize(tt)

        nBytes = 0
        readEnd, writeEnd = os.pipe()
        savedStdout = os.dup(1)
        os.dup2(writeEnd, 1)
        try:
            def drain():
                nonlocal nBytes
                while True:
                    chunk = os.read(readEnd, 4096)
                    if not chunk:
                        break
                    nBytes += len(chunk)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()

            pitts_py.printPerformanceStatistics()
        finally:
            os.dup2(savedStdout, 1)
            os.close(savedStdout)
            os.close(writeEnd)
        reader.join(timeout=10)
        os.close(readEnd)

        self.assertGreater(nBytes, 65536)


if __name__ == '__main__':
    unittest.main()
