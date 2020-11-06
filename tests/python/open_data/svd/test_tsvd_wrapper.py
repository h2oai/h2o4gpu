import numpy as np
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVD

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000, n=10, k=9, algorithm="cusolver", convert_to_float32 = False):
    np.random.seed(1234)

    X = np.random.rand(m, n)
    if convert_to_float32:
        X = X.astype(np.float32)

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
    # Exact scikit impl
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

    rtol = 1E-3
    atol = 1E-5

    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.singular_values_, sklearn_tsvd.singular_values_, rtol=rtol)

    #Check components for first singular value
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.components_[0], sklearn_tsvd.components_[0], rtol=rtol)

    #Check components for second singular value
    #TODO (navdeep) Why does this not match?
    if algorithm != "power":
        assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.components_[1], sklearn_tsvd.components_[1], rtol=.7)

    if algorithm == "power":
        print("Max diff of power components")
        print(str(np.max(h2o4gpu_tsvd_sklearn_wrapper.components_[1]-sklearn_tsvd.components_[1])))

    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_, sklearn_tsvd.explained_variance_, rtol=rtol)
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_ratio_, sklearn_tsvd.explained_variance_ratio_, rtol=rtol)


def test_tsvd_error_k2_cusolver(): func(n=5, k=2, algorithm="cusolver")
def test_tsvd_error_k2_power(): func(n=50, k=2, algorithm="power")
def test_tsvd_error_k2_cusolver_float32(): func(n=5, k=2, algorithm="cusolver", convert_to_float32=True)
def test_tsvd_error_k2_power_float32(): func(n=50, k=2, algorithm="power", convert_to_float32=True)
