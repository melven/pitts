#!/usr/bin/env python

# Copyright (C) 2020 Deutsches Zentrum fuer Luft- und Raumfahrt e.V. (DLR),
# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PITTS common functions python binding"""

__all__ = ['TestCommon']
__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-06-26'
import pitts_py
import subprocess
import sys
import textwrap
import unittest


class TestCommon(unittest.TestCase):
    """TestCase for pitts_py.initialize and pitts_py.finalize"""

    def test_example(self):
        pitts_py.initialize()

        # do something interesting...
        tt = pitts_py.TensorTrain_double([5,5,5])
        pitts_py.randomize(tt)

        pitts_py.clearPerformanceStatistics()
        pitts_py.randomize(tt)
        pitts_py.printPerformanceStatistics()

        pitts_py.finalize(False)

    def test_large_statistics_to_pipe(self):
        """printPerformanceStatistics must not block when stdout is a pipe drained by a python thread

        That is what jupyter does: ipykernel captures the raw file descriptors and reads them in a
        python thread, which needs the GIL. Writing more than the pipe buffer (64 KiB, easily reached
        by these statistics) while holding the GIL would deadlock the kernel, so the binding has to
        release it. Run in a subprocess: a regression hangs instead of failing.
        """
        script = textwrap.dedent("""
            import os, sys, threading
            import pitts_py
            pitts_py.initialize(verbose=False)

            # enough different kernel arguments to exceed the pipe buffer
            for n in range(4, 30):
                tt = pitts_py.TensorTrain_double([n, n, n])
                tt.setTTranks(min(n, 8))
                pitts_py.randomize(tt)
                pitts_py.normalize(tt)

            readEnd, writeEnd = os.pipe()
            savedStdout = os.dup(1)
            os.dup2(writeEnd, 1)

            nBytes = 0
            def drain():
                global nBytes
                while True:
                    chunk = os.read(readEnd, 4096)
                    if not chunk:
                        break
                    nBytes += len(chunk)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()

            pitts_py.printPerformanceStatistics()

            os.dup2(savedStdout, 1)
            os.close(writeEnd)
            reader.join(timeout=10)
            pitts_py.finalize(verbose=False)
            print(nBytes, file=sys.stderr)
            """)
        result = subprocess.run([sys.executable, '-c', script], timeout=120, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertGreater(int(result.stderr.split()[-1]), 65536)


if __name__ == '__main__':
    unittest.main()
