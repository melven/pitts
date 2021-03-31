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


if __name__ == '__main__':

    pitts_py.initialize(True)

    feature_dims = [192,1024,3]

    iFile = 10
    tt_dict = np.load('Xtt%d.npz'%iFile, allow_pickle=True)
    Xtt = dict_to_tensortrain(tt_dict)
    print(Xtt)

    X = pitts_py.toDense(Xtt)
    X = X.reshape([X.shape[0],] + feature_dims, order='F')

    for i in range(X.shape[0]):
        iFrame = i + 30000 + X.shape[0]*iFile
        sample = X[i,:]
        #print('min', sample.min(), 'max', sample.max())
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
        img_error = np.linalg.norm(sample - img_ref)
        real_error = np.linalg.norm(X[i,:] - img_ref)
        print('error img/real:', img_error, real_error)

    pitts_py.finalize(True)

