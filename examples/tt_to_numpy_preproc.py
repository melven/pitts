#!/usr/bin/env python3

"""Test script for reading TT data and check its error"""

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2022-05-30'

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

    tt_dict = np.load('preproc_Xtt0.npz', allow_pickle=True)
    Xtt = dict_to_tensortrain(tt_dict)
    print(Xtt)
    reorder = np.load('preproc_Xtt_reorder.npz', allow_pickle=True)['reorder_features']
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

    fft = False
    if fft:
        nHalf = sample.shape[1] // 2
        sample_fft = np.zeros((sample.shape[0],nHalf+1,sample.shape[2],sample.shape[3]), dtype=np.complex64)
        sample_fft[...,1:,:,:] = sample[...,:nHalf,:,:] + 1.j*sample[...,nHalf:,:,:]

        sample_ref = X_ref[10,...] / 65535
        assert(len(sample_ref.shape) == 4)
        for j in range(sample_ref.shape[0]):
            for k in range(sample_ref.shape[1]):
                sample_ref[j,k,:,:,] = cv2.dct(sample_ref[j,k,:,:])
        sample_fft_ref = np.fft.rfft(sample_ref, axis=-3)

        sample_fft[...,0,:,:] = sample_fft_ref[...,0,:,:]
        #sample_fft[...] = sample_fft_ref[...]

        sample = np.fft.irfft(sample_fft, axis=-3)

    for i in range(sample.shape[-3]):
        img = cv2.idct(sample[7,i,:,:])
        print('min', img.min(), 'max', img.max())
        plt.imshow(img[:,:])
        plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_compressed_dct.png'%i)
        #plt.imshow(X_ref[10,7,i,:,:])
        #plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_ref.png'%i)
        #plt.imshow(sample[7,i,:,:] - X_ref[10,7,i,:,:])
        #plt.savefig('/unsecured_KP/zoel_ml/img10_7_%d_diff.png'%i)

    pitts_py.finalize(True)

