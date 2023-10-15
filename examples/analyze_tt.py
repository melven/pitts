#!/usr/bin/env python3

# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

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
    reorder = np.load('Xtt_reorder.npz')['reorder_features']
    discreteCosineTransform = True

    X = pitts_py.toDense(Xtt)
    if reorder is not None:
        X = X.reshape([X.shape[0],] + [*reorder.shape], order='F')
    else:
        X = X.reshape([X.shape[0],] + feature_dims, order='F')

    for i in range(X.shape[0]):
        iFrame = i+1 + 30000 + X.shape[0]*iFile
        if reorder is not None:
            sample = X[i,reorder].reshape(feature_dims, order='F')
        else:
            sample = X[i,:]
        if discreteCosineTransform:
            # I get segfaults in idct - this somehow prevents it...
            tmp = np.ones(feature_dims) * sample
            tmp[:,:,0] = cv2.idct(tmp[:,:,0])
            tmp[:,:,1] = cv2.idct(tmp[:,:,1])
            tmp[:,:,2] = cv2.idct(tmp[:,:,2])
            sample = tmp
        #print('min', sample.min(), 'max', sample.max())
        sample = np.maximum(sample, 0)
        sample = np.minimum(sample, 1)
        cv2.imwrite('sample%d.png' % iFrame, sample*255)
        img_ref = cv2.imread('/scratch/zoel_ml/ATEK_COPY/combustion{:06d}.jpg'.format(iFrame))
        img_ref = img_ref / 255
        cv2.imwrite('ref%d.png' % iFrame, img_ref*255)
        diff = (1 + sample - img_ref)/2
        cv2.imwrite('diff%d.png' % iFrame, diff*255)
        img_error = np.linalg.norm(sample - img_ref)
        print('error:', img_error)

    pitts_py.finalize(True)

