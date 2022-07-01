#!/usr/bin/env python3

"""Test script for reading TT data and check its error"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-05-30'

import numpy as np
import pitts_py
from matplotlib import pyplot as plt


def dict_to_tensortrain(tt_dict):
    Xtt = pitts_py.TensorTrain_float(tt_dict['dimensions'])
    Xtt.setTTranks(tt_dict['ranks'])
    for i in range(len(Xtt.dimensions())):
        Xtt.setSubTensor(i, tt_dict['cores'][i])
    return Xtt


if __name__ == '__main__':

    pitts_py.initialize(True)

    tt_dict = np.load('Xtt0.npz', allow_pickle=True)
    Xtt = dict_to_tensortrain(tt_dict)
    print(Xtt)
    reorder = np.load('Xtt_reorder.npz', allow_pickle=True)['reorder_features']
    normalize = False

    file_name = '/scratch/DeepArk_mirror/SANS_CM/stackedPixelData.npy'
    # open file as reference
    X_ref = np.load(file_name, mmap_mode='r')
    n_samples = X_ref.shape[0]
    feature_dims = list(X_ref.shape[1:])

    X = pitts_py.toDense(Xtt)
    if reorder:
        X = X.reshape([X_ref.shape[0],] + [*reorder.shape], order='F')
    else:
        X = X.reshape([X_ref.shape[0],] + feature_dims, order='F')


    if reorder:
        sample = X[10,reorder].reshape(feature_dims, order='F')
    else:
        sample = X[10,:]
    for i in range(sample.shape[-3]):
        print('min', sample.min(), 'max', sample.max())
        plt.imshow(sample[7,i,:,:])
        plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_compressed.png'%i)
        plt.imshow(X_ref[10,7,i,:,:])
        plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_ref.png'%i)
        plt.imshow(sample[7,i,:,:] - X_ref[10,7,i,:,:])
        plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_diff.png'%i)

    pitts_py.finalize(True)

