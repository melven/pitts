#!/usr/bin/env python3

"""Test script for reading TT data and analyzing it"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-11-04'

import numpy as np
import pitts_py
from matplotlib import pyplot as plt
import cv2


def dict_to_tensortrain(tt_dict):
    Xtt = pitts_py.TensorTrain_float(tt_dict['dimensions'])
    Xtt.setTTranks(tt_dict['ranks'])
    for i in range(len(Xtt.dimensions())):
        Xtt.setSubTensor(i, tt_dict['cores'][i])
    return Xtt


def multi_index(idx, sample_dims):
    ii = list()
    for d in reversed(sample_dims):
        ii = [idx % d] + ii
        idx = idx // d
    return ii


def eval_sample(Xtt, idx, sample_dims, feature_dims):
    ii = multi_index(idx, sample_dims)
    tmp = np.ones(1, dtype=np.float32)
    d_samples = len(ii)
    d_features = len(Xtt.dimensions()) - d_samples
    print('Xtt', Xtt)
    print('sample_dims', sample_dims)
    print('ii', ii)
    for i in range(d_samples):
        tmp = np.tensordot(tmp, Xtt.getSubTensor(i)[:,ii[i],:], ((0,),(0,)))

    sample_tt = pitts_py.TensorTrain_float(Xtt.dimensions()[d_samples:])
    sample_tt.setTTranks(Xtt.getTTranks()[d_samples:]) 
    tmp = np.tensordot(tmp, Xtt.getSubTensor(d_samples)[:,:,:], ((0,),(0,)))
    tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))
    sample_tt.setSubTensor(0, tmp)
    for i in range(1,d_features):
        j = d_samples + i
        sample_tt.setSubTensor(i, Xtt.getSubTensor(j))

    return sample_tt


if __name__ == '__main__':

    pitts_py.initialize(True)

    nProcs = 8
    d_samples_global = 16
    log2_nProcs = round(np.log2(nProcs))
    d_samples_local = d_samples_global - log2_nProcs
    feature_dims = [192,1024,3]

    iLastFile = None
    for iFrame in range(50000, 50011):

        iFile = iFrame // 2**d_samples_local
        ilocal = iFrame - iFile*2**d_samples_local

        if iLastFile != iFile:
            tt_dict = np.load('Xtt%d.npz'%iFile, allow_pickle=True)
            Xtt = dict_to_tensortrain(tt_dict)
            iLastFile = iFile

        sample_dims = [2,]*d_samples_local
        feature_dims = [192,1024,3]

        sample_tt = eval_sample(Xtt, ilocal, sample_dims, feature_dims)
        sample = pitts_py.toDense(sample_tt)
        sample = sample.reshape(feature_dims, order='F')
        print('min', sample.min(), 'max', sample.max())
        sample = np.maximum(sample, 0)
        sample = np.minimum(sample, 1)
        img = np.zeros(feature_dims, dtype=np.float32)
        img[:,:,0] = sample[:,:,2]
        img[:,:,1] = sample[:,:,1]
        img[:,:,2] = sample[:,:,0]
        plt.imsave('sample%d.png' % iFrame, img)
        img_ref = cv2.imread('/scratch/zoel_ml/ATEK_COPY/combustion{:06d}.jpg'.format(iFrame))
        img_ref = img_ref / 255
        diff = (1 + sample - img_ref)/2
        img[:,:,0] = diff[:,:,2]
        img[:,:,1] = diff[:,:,1]
        img[:,:,2] = diff[:,:,0]
        plt.imsave('diff%d.png' % iFrame, img)

    #n_samples = 100
    #X = np.zeros((n_samples, *feature_dims), dtype=np.float32)
    #for i in range(n_samples):
    #    img = cv2.imread('/scratch/zoel_ml/ATEK_COPY/combustion050{:03d}.jpg'.format(i))
    #    X[i,...] = img / 255
    #dims = [n_samples,] + [2,]*6 + [3,] + [2,]*6 + [2*2*2*2*3,]
    #Xm = pitts_py.MultiVector_float(np.prod(dims[:-1]), dims[-1])
    #work = pitts_py.MultiVector_float(np.prod(dims[:-1]), dims[-1])
    #Xm_view = np.array(Xm, copy=False)
    #Xm_view[...] = X.reshape(Xm_view.shape, order='F')
    #Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.001, maxRank=500, mpiGlobal=True)
    #print(Xtt)
    #Y = pitts_py.toDense(Xtt)
    #print(Y.shape)
    #img0 = Y[0,...].reshape(feature_dims, order='F')
    #img0 = np.maximum(img0, 0)
    #img0 = np.minimum(img0, 1)
    #img0rgb = np.zeros(feature_dims, dtype=np.float32)
    #img0rgb[:,:,0] = img0[:,:,2]
    #img0rgb[:,:,1] = img0[:,:,1]
    #img0rgb[:,:,2] = img0[:,:,0]
    #plt.imsave('sample050000.png', img0rgb)
    #img1 = Y[1,...].reshape(feature_dims, order='F')
    #img1 = np.maximum(img1, 0)
    #img1 = np.minimum(img1, 1)
    #img1rgb = np.zeros(feature_dims, dtype=np.float32)
    #img1rgb[:,:,0] = img1[:,:,2]
    #img1rgb[:,:,1] = img1[:,:,1]
    #img1rgb[:,:,2] = img1[:,:,0]
    #plt.imsave('sample050001.png', img1rgb)

    pitts_py.finalize(True)

