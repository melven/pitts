#!/usr/bin/env python3

# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

""" Example for a spectral clustering algorithm that uses a low-rank approximation of the data """

__authors__ = ['Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>']
__date__ = '2020-09-08'

import numpy as np
import scipy as sp
import pitts_py
import sklearn.cluster
import sklearn.datasets
import functools
import timeit
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.,20.)
from matplotlib import pyplot as plt


def timer(func):
    """measure runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_fun(*args, **kwargs):
        wtime = timeit.default_timer()
        value = func(*args, **kwargs)
        wtime = timeit.default_timer() - wtime
        print(func.__name__, "wtime:", wtime)
        return value
    return wrapper_fun


@timer
def make_data(n_samples, n_features, n_clusters):
    """generate some data with multiple clusters"""
    N = n_samples
    m = n_features
    X1, _ = sklearn.datasets.make_circles(n_samples=N//n_clusters*2, factor=0.5, noise=0.05)
    X1 = X1 * 2
    X1 = np.pad(X1, ((0,0),(0,m-2)))
    X1 = X1 + np.random.rand(*X1.shape) * 1.e-5
    X2, _ = sklearn.datasets.make_blobs(n_samples=N//n_clusters*(n_clusters-2), n_features=m, centers=n_clusters-2, random_state=8)
    X = np.concatenate((X1*5, X2))
    # normalize
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    X = (X - xmin) / (xmax-xmin)
    return X


@timer
def cdist2(X):
    """calculate the pair-wise squared distances of all rows"""
    N = X.shape[0]

    # ||X_i - X_j||_2^2 = ||X_i||_^2 - 2<X_i,X_j> + ||X_j||_2^2
    squareNorms = np.linalg.norm(X,axis=1)**2
    XXt = X @ X.transpose()

    distanceSquared = np.outer(squareNorms, np.ones(N)) - 2 * XXt + np.outer(np.ones(N), squareNorms)
    #distance = np.sqrt(np.maximum(0., distanceSquared))
    return distanceSquared


@timer
def nearestNeighbor_Laplacian(distanceSquared, n_neighbors):
    """calculate a sparse graph Laplacian matrix of nearest neighbors given a matrix with pair-wise distances"""
    N = distanceSquared.shape[0]
    NN = n_neighbors

    # only do partial sorting...
    nearestIdxPerRow = np.argpartition(distanceSquared, NN, axis=1)[:,:NN]
    nearest = sp.sparse.csr_matrix((np.ones(N*NN), nearestIdxPerRow.flatten(), np.arange(0, (N+1)*NN, NN)), shape=(N,N))
    nearest = nearest.maximum(nearest.transpose())
    nearest.setdiag(0)

    rowsum = nearest.sum(axis=0)
    rowsumInvRoot = sp.sparse.diags(np.squeeze(np.asarray(1/np.sqrt(rowsum))), 0)
    nn_laplacian = sp.sparse.eye(N,N) - rowsumInvRoot @ nearest @ rowsumInvRoot
    return nn_laplacian


@timer
def smallestEigenVectors(sparseHermitianMatrix, n_eigs):
    """calculate the eigenvectors of the n smallest eigenvalues given a sparse symmetric postive definite matrix"""
    eigVals, eigVecs = sp.sparse.linalg.eigsh(sparseHermitianMatrix, n_eigs, which='SM')
    return eigVecs


@timer
def kmeansClustering(X, n_clusters):
    """calculate labels based on a k-means clustering of the data"""
    # scale X
    invNorm = 1 / np.linalg.norm(X, axis=1)
    X = np.diag(invNorm) @ X
    _, yest, _ = sklearn.cluster.k_means(X=X, n_clusters=n_clusters)
    return yest


@timer
def nearestNeighborSpectralClustering(X, n_neighbors, n_clusters, debug=False):
    """calculate labels based on a spectral clustering of a nearest neighbor graph Laplacian of the data"""
    distanceSquared = cdist2(X)
    nn_laplacian = nearestNeighbor_Laplacian(distanceSquared, n_neighbors)
    if debug:
        plt.matshow(nn_laplacian.todense() != 0)
        plt.savefig('nn_sparsity.png')
        plt.close()
    nn_V = smallestEigenVectors(nn_laplacian, n_clusters)
    nn_yest = kmeansClustering(nn_V, n_clusters)
    return nn_yest


@timer
def cdist2_TTapprox(X, dims_samples, dims_features, rank_tolerance, max_rank, debug=False):
    """approximate the pair-wise squared distances of all rows using a TT decomposition of the data"""
    # convert to TT format
    dims = dims_samples + dims_features
    n_samples = np.prod(dims_samples)
    n_features = np.prod(dims_features)
    Xm = pitts_py.MultiVector_double(n_samples*n_features//dims[-1], dims[-1])
    work = pitts_py.MultiVector_double()
    Xm_view = np.array(Xm, copy=False)

    # scale by norm -> length 1
    Xnorms = np.linalg.norm(X,axis=1)
    Xm_view[...] = (np.diag(1 / Xnorms) @ X).reshape(Xm_view.shape, order='F')

    Xtt = pitts_py.fromDense(Xm, work, dims, rankTolerance=0.01, maxRank=200)
    if debug:
        X_approx = pitts_py.toDense(Xtt).reshape((n_samples, n_features), order='F') * Xnorms[:, np.newaxis]
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(X_approx[:,0], X_approx[:,1])
        plt.savefig('scatter_01_approx.png')
        plt.close()

        plt.scatter(X[:,1], X[:,2])
        plt.scatter(X_approx[:,1], X_approx[:,2])
        plt.savefig('scatter_12_approx.png')
        plt.close()

        plt.scatter(X[:,3], X[:,4])
        plt.scatter(X_approx[:,3], X_approx[:,4])
        plt.savefig('scatter_34_approx.png')
        plt.close()

    # setup approximated (US) from SVD (U S V^T)
    d_samples = len(dims_samples)
    Xtt_ranks = Xtt.getTTranks()
    k = Xtt_ranks[d_samples-1]
    US_tt = pitts_py.TensorTrain_double(dims_samples + [k,])
    US_tt.setTTranks(Xtt_ranks[:d_samples-1] + [k,])
    for i in range(d_samples):
        US_tt.setSubTensor(i, Xtt.getSubTensor(i))
    # last sub-tensor is just the identity to get an additional direction
    US_tt.setSubTensor(d_samples, np.eye(k,k).reshape((k,k,1)))
    US_approx = pitts_py.toDense(US_tt).reshape((n_samples, k), order='F')
    if debug:
        plt.matshow(US_approx)
        plt.savefig('US_approx.png')
        plt.close()
    US_approx = np.diag(Xnorms) @ US_approx
    return cdist2(US_approx)


@timer
def nearestNeighborSpectralClustering_TT(X, dims_samples, dims_features, rank_tolerance, max_rank, n_neighbors, n_clusters, debug=False):
    """calculate labels based on a spectral clustering of an approximated nearest neighbor graph Laplacian of the data"""
    distanceSquared_approx = cdist2_TTapprox(X, dims_samples, dims_features, rank_tolerance, max_rank, debug=debug)
    nn_laplacian = nearestNeighbor_Laplacian(distanceSquared_approx, n_neighbors)
    if debug:
        plt.matshow(nn_laplacian.todense() != 0)
        plt.savefig('nn_sparsity_approx.png')
        plt.close()
    nn_V = smallestEigenVectors(nn_laplacian, n_clusters)
    nn_yest = kmeansClustering(nn_V, n_clusters)
    return nn_yest


if __name__ == '__main__':

    # TT dimensions (-> kind of QTT format)
    dims_samples = [2**5,] + [2,]*5
    dims_features = [2,]*5 + [2**5,]

    n_samples = np.prod(dims_samples)
    n_features = np.prod(dims_features)

    N = n_samples
    n_clusters = 8

    X = make_data(n_samples, n_features, n_clusters)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig('scatter_01.png')
    plt.close()


    nn_yest = nearestNeighborSpectralClustering(X, n_neighbors=10, n_clusters=n_clusters, debug=False)
    plt.scatter(X[:,0], X[:,1], c=nn_yest)
    plt.savefig('clusters_01.png')
    plt.close()


    pitts_py.initialize(True)

    nn_yest_approx = nearestNeighborSpectralClustering_TT(X, dims_samples, dims_features, rank_tolerance=0.001, max_rank=200, n_neighbors=10, n_clusters=n_clusters-1, debug=False)
    plt.scatter(X[:,0], X[:,1], c=nn_yest_approx)
    plt.savefig('clusters_01_approx.png')
    plt.close()

    pitts_py.finalize(True)
