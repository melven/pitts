# Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import pitts_py
import numpy as np
from tt_jacobi_davidson import tt_jacobi_davidson
import scipy.sparse.linalg

# temp = tt.rand(n[i]*n[i],d=d[i],r=2)
# print 'norm temp',temp.norm()
# Att = tt.matrix(a=temp)
# Att = Att*(1/Att.norm())
# Att = 0.5*(Att.__add__(Att.T))
# Att = (Att*100).__add__(tt.eye(n[i],d=d[i]))

def transpose(Att):
    Att_t = pitts_py.TensorTrainOperator_double(Att.row_dimensions(), Att.col_dimensions())
    Att_t.setTTranks(Att.getTTranks())
    for iDim in range(len(Att.row_dimensions())):
        subT = Att.getSubTensor(iDim)
        subT_t = np.swapaxes(subT, 1, 2)
        Att_t.setSubTensor(iDim, subT_t)
    return Att_t

def randomOp1(dims, randomOp=None):
    TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
    if randomOp is None:
        TTOp.setTTranks(2)
        pitts_py.randomize(TTOp)
    else:
        pitts_py.copy(randomOp, TTOp)
    TTOpT = transpose(TTOp)
    pitts_py.axpby(1, TTOpT, 1, TTOp)
    pitts_py.normalize(TTOp)
    TTOpEye = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOpEye.setEye()
    eyeNrm = pitts_py.normalize(TTOpEye)
    pitts_py.axpby(eyeNrm, TTOpEye, 0.0125/np.sqrt(2)*eyeNrm, TTOp)
    return TTOp

def randomOp2(dims, randomOp=None):
    TTOp = pitts_py.TensorTrainOperator_double(dims, dims)
    if randomOp is None:
        TTOp.setTTranks(2)
        pitts_py.randomize(TTOp)
    else:
        pitts_py.copy(randomOp, TTOp)
    pitts_py.normalize(TTOp)
    TTOpT = transpose(TTOp)
    pitts_py.axpby(0.5, TTOpT, 0.5, TTOp)
    TTOpEye = pitts_py.TensorTrainOperator_double(dims, dims)
    TTOpEye.setEye()
    pitts_py.axpby(1, TTOpEye, 100, TTOp)
    return TTOp

def toDense(TTOp):
    dims = TTOp.row_dimensions()
    assert(dims == TTOp.col_dimensions())
    N = np.prod(dims)
    M = np.zeros((N,N))

    ei = pitts_py.TensorTrain_double(dims)
    ej = pitts_py.TensorTrain_double(dims)
    for i in range(N):
        #ei.setUnit
        for j in range(N):
            pass

def main():
    pitts_py.initialize()

    dims = [20,]*3
    randomOp = pitts_py.TensorTrainOperator_double(dims, dims)
    randomOp.setTTranks(2)
    pitts_py.randomize(randomOp)

    TTOp1 = randomOp1(dims, randomOp)
    TTOp2 = randomOp2(dims, randomOp)

    TTOpDiff = pitts_py.TensorTrainOperator_double(dims, dims)
    pitts_py.copy(TTOp1, TTOpDiff)
    pitts_py.axpby(-1, TTOp2, 1, TTOpDiff)
    #print('TTOp1 nrm', pitts_py.normalize(TTOp1))
    #print('TTOp2 nrm', pitts_py.normalize(TTOp2))
    print('TTOpDiff nrm', pitts_py.normalize(TTOpDiff))

    x0 = pitts_py.TensorTrain_double(dims)
    x0.setOnes()
    sigma, q = tt_jacobi_davidson(TTOp1, x0, symmetric=True, arnoldiIter=20, eps=0.0001)
    print('sigma', sigma)
    

    # N = np.prod(dims)

    # def apply(x):
    #     x[0] += 1.e-16
    #     yTT = pitts_py.TensorTrain_double(dims)
    #     Xm = pitts_py.MultiVector_double(N//dims[-1], dims[-1])
    #     work = pitts_py.MultiVector_double()
    #     Xm_view = np.array(Xm, copy=False)
    #     #print('x.shape', x.shape)
    #     Xm_view[...] = x.reshape(Xm_view.shape, order='F')
    #     #print('A')
    #     xTT = pitts_py.fromDense(Xm, work, dims, rankTolerance=1.e-8, maxRank=1000)
    #     #print('xTT', xTT)
    #     #print('B')
    #     pitts_py.apply(TTOp1, xTT, yTT)
    #     #print('C')
    #     tmp = pitts_py.toDense(yTT)
    #     #print('D')
    #     return tmp.reshape((N), order='F')
    # 
    # AOp = scipy.sparse.linalg.LinearOperator((N,N), matvec=apply)
    # w, v = scipy.sparse.linalg.eigsh(AOp, k=1, tol=0.0001, which='SA')
    # print('eigs', w)


    pitts_py.finalize(verbose=False)



if __name__ == '__main__':
    main()
