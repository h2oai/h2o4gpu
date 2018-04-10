import numpy as np
import sys
import logging
from h2o4gpu.solvers.pca import PCA
from h2o4gpu.solvers.pca import PCAH2O
from h2o4gpu.decomposition import PCASklearn
from h2o4gpu.util import gpu

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9, change_gpu_id=False, use_wrappper=False, convert_to_float32=False):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        X = X.astype(np.float32)
    print("X dtype")
    print(X.dtype)
    gpu_id = 0

    if change_gpu_id:
        total_gpu, total_mem, gpu_type = gpu.get_gpu_info()

        if(total_gpu > 0):
            gpu_id = 1 #Use second gpu

    print("\n")
    print(X)
    print(np.mean(X, axis=0))

    if use_wrappper:
        h2o4gpu_pca = PCA(n_components=k, gpu_id = gpu_id)
    else:
        h2o4gpu_pca = PCAH2O(n_components=k, gpu_id=gpu_id)

    scikit_pca = PCASklearn(n_components=k, svd_solver="arpack")
    scikit_pca.fit(X)
    h2o4gpu_pca.fit(X)

    print("Mean")
    print(h2o4gpu_pca.mean_)
    print(scikit_pca.mean_)
    if convert_to_float32:
        assert np.allclose(h2o4gpu_pca.mean_, scikit_pca.mean_, 1e-4)
    else:
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
def test_pca_error_k2_gpuid(): func(m=1000000, n=10, k=2, change_gpu_id=True)
def test_pca_error_k2_wrapper(): func(m=1000000, n=10, k=2, use_wrappper=True)
def test_pca_error_k2_gpuid_wrapper(): func(m=1000000, n=10, k=2, change_gpu_id=True, use_wrappper=True)
def test_pca_error_k2_float(): func(m=1000000, n=10, k=2, convert_to_float32 = True)
def test_pca_error_k2_gpuid_float(): func(m=1000000, n=10, k=2, change_gpu_id=True, convert_to_float32 = True)
def test_pca_error_k2_wrapper_float(): func(m=1000000, n=10, k=2, use_wrappper=True, convert_to_float32 = True)
def test_pca_error_k2_gpuid_wrapper_float(): func(m=1000000, n=10, k=2, change_gpu_id=True, use_wrappper=True, convert_to_float32 = True)
