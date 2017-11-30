import numpy as np
import sys
import logging
from h2o4gpu.solvers.pca import PCAH2O
from sklearn.decomposition import PCA

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    print("\n")
    print(X)
    print(np.mean(X, axis=0))

    h2o4gpu_pca = PCAH2O(n_components=k)
    scikit_pca = PCA(n_components=k, svd_solver="arpack")
    scikit_pca.fit(X)
    h2o4gpu_pca.fit(X)

    print("Mean")
    print(h2o4gpu_pca.mean_)
    print(scikit_pca.mean_)
    assert np.allclose(h2o4gpu_pca.mean_, scikit_pca.mean_)

    print("Noise Variance")
    print(h2o4gpu_pca.noise_variance_)
    print(scikit_pca.noise_variance_)
    assert np.allclose(h2o4gpu_pca.noise_variance_, h2o4gpu_pca.noise_variance_)

    print("Explained variance")
    print(h2o4gpu_pca.explained_variance_)
    print(scikit_pca.explained_variance_)
    assert np.allclose(h2o4gpu_pca.explained_variance_, scikit_pca.explained_variance_)

    print("Explained variance ratio")
    print(h2o4gpu_pca.explained_variance_ratio_)
    print(scikit_pca.explained_variance_ratio_)
    assert np.allclose(h2o4gpu_pca.explained_variance_ratio_, scikit_pca.explained_variance_ratio_, .1)

    print("Singular values")
    print(h2o4gpu_pca.singular_values_)
    print(scikit_pca.singular_values_)
    assert np.allclose(h2o4gpu_pca.singular_values_, scikit_pca.singular_values_)

    print("Components")
    print(h2o4gpu_pca.components_)
    print(scikit_pca.components_)
    assert np.allclose(h2o4gpu_pca.components_, scikit_pca.components_, .1)

    print("Num components")
    print(h2o4gpu_pca.n_components)
    print(scikit_pca.n_components)
    assert h2o4gpu_pca.n_components_ == scikit_pca.n_components_

def test_pca_error_k2(): func(m=1000000, n=10, k=2)