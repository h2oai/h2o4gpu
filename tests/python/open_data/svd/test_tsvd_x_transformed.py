import numpy as np
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVDH2O

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000, n=10, k=9, algorithm = "cusolver", convert_to_float32 = False):
    np.random.seed(1234)

    X = np.random.rand(m, n)
    if convert_to_float32:
        X = X.astype(np.float32)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)

    print("\n")
    print("H2O4GPU run")
    h2o4gpu_tsvd_sklearn_wrapper = TruncatedSVDH2O(n_components=k, algorithm=algorithm, tol = 1e-50, n_iter=2000, random_state=42, verbose=True)
    h2o4gpu_tsvd_sklearn_wrapper.fit(X)
    X_transformed = h2o4gpu_tsvd_sklearn_wrapper.fit_transform(X)
    print("\n")
    print("Sklearn run")
    # Exact scikit impl
    sklearn_tsvd = sklearnsvd(n_components=k, random_state=42)
    sklearn_tsvd.fit(X)
    X_transformed_sklearn = sklearn_tsvd.fit_transform(X)
    print("X transformed cuda")
    print(X_transformed)
    print("X transformed sklearn")
    print(X_transformed_sklearn)
    print("Max absolute diff between X_transformed and X_transformed_sklearn")
    print(str(np.max(np.abs(X_transformed) - np.abs(X_transformed_sklearn))))
    if convert_to_float32:
        if algorithm == "power":
            # TODO: CUDA-10.2 gives less accurate result
            assert np.allclose(X_transformed, X_transformed_sklearn, atol=0.011)
        else:
            assert np.allclose(X_transformed, X_transformed_sklearn, atol = 1.95616e-03)
    else:
        if algorithm=="power":
            assert np.allclose(X_transformed, X_transformed_sklearn, atol=1.8848614999622538e-04)
        else:
            assert np.allclose(X_transformed, X_transformed_sklearn)

def test_tsvd_error_k2_double(): func(n=5, k=3)
def test_tsvd_error_k2_float32(): func(n=5, k=2, convert_to_float32=True)
def test_tsvd_error_k2_double_power(): func(n=5, k=3, algorithm="power")
def test_tsvd_error_k2_float32_power(): func(n=5, k=2, algorithm = "power", convert_to_float32=True)
