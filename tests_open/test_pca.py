import numpy as np
import sys
import logging
from h2o4gpu.solvers.pca import PCAH2O
from sklearn.decomposition import PCA

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9):
    np.random.seed(1234)

    X = np.array([[2,4,3], [1,5,7]])
    X = np.transpose(X)
    #X = np.random.rand(m, n)

    print("\n")
    print(X)
    print(np.mean(X, axis=0))

    h2o4gpu_pca = PCAH2O(n_components=k)
    scikit_pca = PCA(n_components=k, svd_solver="full")
    scikit_pca.fit(X)
    h2o4gpu_pca.fit(X)
    print(h2o4gpu_pca.explained_variance)
    print(scikit_pca.explained_variance_)

def test_pca_error_k2(): func(m=5000000, n=10, k=1)