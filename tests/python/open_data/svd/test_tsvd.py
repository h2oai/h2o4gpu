import numpy as np
import time
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVDH2O
from scipy.sparse.linalg import svds
from h2o4gpu.utils.extmath import svd_flip

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9, convert_to_float32=False):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        X = X.astype(np.float32)

    # Exact scikit impl
    sklearn_tsvd = sklearnsvd(algorithm="arpack", n_components=k, random_state=42)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu tsvd run")
    start_time = time.time()
    h2o4gpu_tsvd = TruncatedSVDH2O(n_components=k, random_state=42)
    h2o4gpu_tsvd.fit(X)
    end_time = time.time() - start_time
    print("Total time for h2o4gpu tsvd is " + str(end_time))
    print("h2o4gpu tsvd Singular Values")
    print(h2o4gpu_tsvd.singular_values_)
    print("h2o4gpu tsvd Components (V^T)")
    print(h2o4gpu_tsvd.components_)
    print("h2o4gpu tsvd Explained Variance")
    print(h2o4gpu_tsvd.explained_variance_)
    print("h2o4gpu tsvd Explained Variance Ratio")
    print(h2o4gpu_tsvd.explained_variance_ratio_)

    print("\n")
    print("sklearn run")
    start_sk = time.time()
    sklearn_tsvd.fit(X)
    end_sk = time.time() - start_sk
    print("Total time for sklearn is " + str(end_sk))
    print("Sklearn Singular Values")
    print(sklearn_tsvd.singular_values_)
    print("Sklearn Components (V^T)")
    print(sklearn_tsvd.components_)
    print("Sklearn Explained Variance")
    print(sklearn_tsvd.explained_variance_)
    print("Sklearn Explained Variance Ratio")
    print(sklearn_tsvd.explained_variance_ratio_)

    print("\n")
    print("h2o4gpu tsvd U matrix")
    print(h2o4gpu_tsvd.U)
    print("h2o4gpu tsvd V^T")
    print(h2o4gpu_tsvd.components_)
    print("h2o4gpu tsvd Sigma")
    print(h2o4gpu_tsvd.singular_values_)
    print("h2o4gpu tsvd U * Sigma")
    x_tsvd_transformed = h2o4gpu_tsvd.U * h2o4gpu_tsvd.singular_values_
    print(x_tsvd_transformed)
    print("h2o4gpu tsvd Explained Variance")
    print(np.var(x_tsvd_transformed, axis=0))

    U, Sigma, VT = svds(X, k=k, tol=0)
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    print("\n")
    print("Sklearn U matrix")
    print(U)
    print("Sklearn V^T")
    print(VT)
    print("Sklearn Sigma")
    print(Sigma)
    print("Sklearn U * Sigma")
    X_transformed = U * Sigma
    print(X_transformed)
    print("sklearn Explained Variance")
    print(np.var(X_transformed, axis=0))

    print("U shape")
    print(np.shape(h2o4gpu_tsvd.U))
    print(np.shape(U))

    print("Singular Value shape")
    print(np.shape(h2o4gpu_tsvd.singular_values_))
    print(np.shape(sklearn_tsvd.singular_values_))

    print("Components shape")
    print(np.shape(h2o4gpu_tsvd.components_))
    print(np.shape(sklearn_tsvd.components_))

    print("Reconstruction")
    reconstruct_h2o4gpu = h2o4gpu_tsvd.inverse_transform(h2o4gpu_tsvd.fit_transform(X))
    reconstruct_sklearn = sklearn_tsvd.inverse_transform(sklearn_tsvd.fit_transform(X))
    reconstruct_h2o4gpu_manual = np.sum([np.outer(h2o4gpu_tsvd.U[:, i], h2o4gpu_tsvd.components_[i, :]) * si for i, si in enumerate(h2o4gpu_tsvd.singular_values_)], axis=0)
    print("Check inverse_transform() vs manual reconstruction for h2o4gpu")
    rtol=1E-2
    assert np.allclose(reconstruct_h2o4gpu, reconstruct_h2o4gpu_manual, rtol=rtol)
    #reconstruct_sklearn_manual = np.sum([np.outer(U[:, i], sklearn_tsvd.components_[i, :]) * si for i, si in enumerate(sklearn_tsvd.singular_values_)], axis=0)
    print("original X")
    print(X)
    print("h2o4gpu reconstruction")
    print(reconstruct_h2o4gpu)
    print("sklearn reconstruction")
    print(reconstruct_sklearn)
    h2o4gpu_diff = np.subtract(reconstruct_h2o4gpu, X)
    sklearn_diff = np.subtract(reconstruct_sklearn, X)
    print("h2o4gpu diff")
    print(h2o4gpu_diff)
    print("sklearn diff")
    print(sklearn_diff)
    h2o4gpu_max_diff = np.amax(abs(h2o4gpu_diff))
    sklearn_max_diff = np.amax(abs(sklearn_diff))
    print("h2o4gpu max diff")
    print(h2o4gpu_max_diff)
    print("sklearn max diff")
    print(sklearn_max_diff)
    print("h2o4gpu mae")
    h2o4gpu_mae = np.mean(np.abs(h2o4gpu_diff))
    print(h2o4gpu_mae)
    print("sklearn mae")
    sklearn_mae = np.mean(np.abs(sklearn_diff))
    print(sklearn_mae)

    return h2o4gpu_mae, sklearn_mae

def reconstruction_error(m=5000, n=10, k=9, convert_to_float32=False):
    h2o4gpu_mae_list = np.zeros(k, dtype=np.float64)
    sklearn_mae_list = np.zeros(k, dtype=np.float64)
    for i in range(1,k+1):
        h2o4gpu_mae_list[i-1] = func(m, n, i, convert_to_float32=convert_to_float32)[0]
        sklearn_mae_list[i-1] = func(m, n, i, convert_to_float32=convert_to_float32)[1]
    print("H2O4GPU MAE across k")
    print(h2o4gpu_mae_list)
    #Sort in descending order and check error goes down as k increases
    h2o4gpu_mae_list_sorted = np.sort(h2o4gpu_mae_list)[::-1]
    assert np.array_equal(h2o4gpu_mae_list, h2o4gpu_mae_list_sorted)
    print("Sklearn MAE across k")
    print(sklearn_mae_list)
    assert np.allclose(h2o4gpu_mae_list, sklearn_mae_list, 1e-3, 1e-3)
    # np.savetxt('h2o4gpu_k'+ str(k) + '_' + str(m) + '_by_' + str(n) + '_.csv', h2o4gpu_mae_list, delimiter=',')
    # np.savetxt('sklearn_k'+ str(k) + '_' + str(m) + '_by_' + str(n) + '_.csv', sklearn_mae_list, delimiter=',')

def test_tsvd_error_k2(): reconstruction_error(n=50, k=5)
def test_tsvd_error_k5(): reconstruction_error(n=100, k=7)
def test_tsvd_error_k2_float32(): reconstruction_error(n=50, k=5, convert_to_float32= True)
def test_tsvd_error_k5_float32(): reconstruction_error(n=100, k=7, convert_to_float32=True)
