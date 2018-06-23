import numpy as np
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from sklearn.datasets import load_iris
from h2o4gpu.solvers import TruncatedSVD

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(k=9, algorithm="cusolver", rtol=1E-3):

    X = load_iris()
    X = X.data

    #Increase row size of matrix
    X = np.concatenate((X, X), axis=0)
    X = np.concatenate((X, X), axis=0)
    X = np.concatenate((X, X), axis=0)
    X = np.concatenate((X, X), axis=0)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)
    X = np.concatenate((X, X), axis=1)

    print("\n")
    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)

    print("\n")
    print("H2O4GPU run")
    h2o4gpu_tsvd_sklearn_wrapper = TruncatedSVD(n_components=k, algorithm=algorithm, tol = 1E-50, n_iter=200, random_state=42, verbose=True)
    h2o4gpu_tsvd_sklearn_wrapper.fit(X)
    print("h2o4gpu tsvd Singular Values")
    print(h2o4gpu_tsvd_sklearn_wrapper.singular_values_)
    print("h2o4gpu tsvd Components (V^T)")
    print(h2o4gpu_tsvd_sklearn_wrapper.components_)
    print("h2o4gpu tsvd Explained Variance")
    print(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_)
    print("h2o4gpu tsvd Explained Variance Ratio")
    print(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_ratio_)

    print("\n")
    print("Sklearn run")
    #Exact scikit impl
    sklearn_tsvd = sklearnsvd(algorithm="arpack", n_components=k, random_state=42)
    sklearn_tsvd.fit(X)
    print("Sklearn Singular Values")
    print(sklearn_tsvd.singular_values_)
    print("Sklearn Components (V^T)")
    print(sklearn_tsvd.components_)
    print("Sklearn Explained Variance")
    print(sklearn_tsvd.explained_variance_)
    print("Sklearn Explained Variance Ratio")
    print(sklearn_tsvd.explained_variance_ratio_)

    #Check singular values
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.singular_values_, sklearn_tsvd.singular_values_, rtol=rtol)

    #Check components for first singular value
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.components_[0], sklearn_tsvd.components_[0], rtol=rtol)

    #Check components for second singular value
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.components_[1], sklearn_tsvd.components_[1], rtol=.7)

    #Check explained variance and explained variance ratio
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_, sklearn_tsvd.explained_variance_, rtol=rtol)
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_ratio_, sklearn_tsvd.explained_variance_ratio_, rtol=rtol)

def test_tsvd_error_k2_cusolver(): func(k=2, algorithm="cusolver")
def test_tsvd_error_k2_power(): func(k=2, algorithm="power")
def test_tsvd_error_k3_cusolver(): func(k=3, algorithm="cusolver")
def test_tsvd_error_k3_power(): func(k=3, algorithm="power")
def test_tsvd_error_k4_cusolver(): func(k=4, algorithm="cusolver")
def test_tsvd_error_k4_power(): func(k=4, algorithm="power")
