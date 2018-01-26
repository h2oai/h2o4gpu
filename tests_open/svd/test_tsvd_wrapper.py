import numpy as np
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVD
import pytest

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000, n=10, k=9, algorithm="cusolver"):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    # Exact scikit impl
    sklearn_tsvd = sklearnsvd(algorithm="arpack", n_components=k, random_state=42)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("Sklearn run through h2o4gpu wrapper")

    h2o4gpu_tsvd_sklearn_wrapper = TruncatedSVD(n_components=k, algorithm=algorithm, random_state=42, verbose=True)
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
    print("sklearn run")
    sklearn_tsvd.fit(X)
    print("Sklearn Singular Values")
    print(sklearn_tsvd.singular_values_)
    print("Sklearn Components (V^T)")
    print(sklearn_tsvd.components_)
    print("Sklearn Explained Variance")
    print(sklearn_tsvd.explained_variance_)
    print("Sklearn Explained Variance Ratio")
    print(sklearn_tsvd.explained_variance_ratio_)

    if algorithm=='arpack':
        rtol = 1E-5
    else:
        rtol = 1E-2
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.singular_values_, sklearn_tsvd.singular_values_, rtol=rtol)
    if algorithm=='arpack':
        rtol = 1E-5
    else:
        rtol = 1E-1
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.components_, sklearn_tsvd.components_, rtol=rtol)
    if algorithm=='arpack':
        rtol = 1E-5
    else:
        rtol = 0.5
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_, sklearn_tsvd.explained_variance_, rtol=rtol)
    assert np.allclose(h2o4gpu_tsvd_sklearn_wrapper.explained_variance_ratio_, sklearn_tsvd.explained_variance_ratio_, rtol=rtol)


def test_tsvd_error_k2_cusolver(): func(n=50, k=2, algorithm="cusolver")
@pytest.mark.skip("Failing")
def test_tsvd_error_k2_power(): func(n=50, k=2, algorithm="power")
