import pitts_py
import numpy as np

pitts_py.initialize(True)
TT1 = pitts_py.TensorTrain_double([100,]*10)
TT2 = pitts_py.TensorTrain_double([100,]*10)
TT2.setTTranks(150)
TT1.setTTranks(20)
pitts_py.randomize(TT1)
pitts_py.randomize(TT2)

tmp = 0
for i in range(10):
    tmp = tmp + pitts_py.axpby(0.01, TT1, 0.9, TT2)

pitts_py.finalize(True)
