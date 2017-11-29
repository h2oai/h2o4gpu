import numpy as np
import sys
import logging
from h2o4gpu.solvers.pca import PCAH2O
from sklearn.decomposition import PCA

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9):
    np.random.seed(1234)

    X = np.array([[2,4,3], [1,5,7], [3,6,8]])
    X = np.transpose(X)
    print(X.shape)
    #X = np.random.rand(m, n)

    print("\n")
    print(X)
    print(np.mean(X, axis=0))

    h2o4gpu_pca = PCAH2O(n_components=k)
    scikit_pca = PCA(n_components=k, svd_solver="arpack")
    scikit_pca.fit(X)
    h2o4gpu_pca.fit(X)
    print(h2o4gpu_pca.mean_)

    print("Explained variance")
    print(h2o4gpu_pca.explained_variance_)
    print(scikit_pca.explained_variance_)

    print("Explained variance ratio")
    print(h2o4gpu_pca.explained_variance_ratio_)
    print(scikit_pca.explained_variance_ratio_)

    print("Singular values")
    print(h2o4gpu_pca.singular_values_)
    print(scikit_pca.singular_values_)

    print("Components")
    print(h2o4gpu_pca.components_)
    print(scikit_pca.components_)

    print("Num components")
    print(h2o4gpu_pca.n_components)
    print(scikit_pca.n_components)

def test_pca_error_k2(): func(m=5000000, n=10, k=2)