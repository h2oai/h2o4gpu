import numpy as np
import time
import sys
import logging
from h2o4gpu.solvers import TruncatedSVDH2O
from h2o4gpu.util import gpu

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000, n=10, k=9, convert_to_float32=False):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        X = X.astype(np.float32)
    gpu_id = 0

    total_gpu, total_mem, gpu_type = gpu.get_gpu_info()

    if(total_gpu > 1): #More than one gpu?
        gpu_id = 1 #Use second gpu

    print("\n")
    print("SVD on gpu id -> " + str(gpu_id))
    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu tsvd run")
    start_time = time.time()
    h2o4gpu_tsvd = TruncatedSVDH2O(n_components=k, gpu_id=gpu_id)
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

def test_tsvd_gpu_id(): func(k=5)
def test_tsvd_gpu_id_float32(): func(k=5, convert_to_float32=True)