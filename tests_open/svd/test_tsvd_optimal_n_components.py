import numpy as np
import time
import sys
import logging
from h2o4gpu.solvers import TruncatedSVDH2O
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers.decomposition_utils.utils import find_optimal_n_components

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000, n=10, convert_to_float32=False):
    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        X = X.astype(np.float32)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")

    print("h2o4gpu tsvd run")
    start_time = time.time()
    h2o4gpu_tsvd = TruncatedSVDH2O(n_components=9, random_state=42)
    h2o4gpu_tsvd.fit(X)
    end_time = time.time() - start_time
    print("Total time for h2o4gpu tsvd is " + str(end_time))

    print("\n")
    print("sklearn tsvd run")
    start_time_sklearn = time.time()
    sklearn_tsvd = sklearnsvd(algorithm="arpack", n_components=9, random_state=42)
    sklearn_tsvd.fit(X)
    end_time_sklearn = time.time() - start_time_sklearn
    print("Total time for sklearn tsvd is " + str(end_time_sklearn))

    h2o4gpu_tsvd_var_ratios = h2o4gpu_tsvd.explained_variance_ratio_
    sklearn_tsvd_var_ratios = sklearn_tsvd.explained_variance_ratio_
    optimal_variance = .80
    optimal_k_h2o4gpu = find_optimal_n_components(h2o4gpu_tsvd_var_ratios, optimal_variance)
    optimal_k_sklearn = find_optimal_n_components(sklearn_tsvd_var_ratios, optimal_variance)
    print("\n")
    print("Finding optimal k to account for %s percent of the variance in h2o4gpu tsvd" % str(optimal_variance*100))
    print("Optimal k for h2o4gpu impl-> " + str(optimal_k_h2o4gpu))
    print("\n")
    print("Finding optimal k to account for %s percent of the variance in sklearn tsvd" % str(optimal_variance*100))
    print("Optimal k for sklearn impl -> " + str(optimal_k_sklearn))
    assert optimal_k_h2o4gpu == optimal_k_sklearn, "Optimal k between h2o4gpu and skearn should match"

def test_tsvd_double(): func()
def test_tsvd_float32(): func(convert_to_float32=True)