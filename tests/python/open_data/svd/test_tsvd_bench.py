import numpy as np
import time
import sys
import logging
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVDH2O


print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9, convert_to_float32=False):
    import os
    if os.getenv("CHECKPERFORMANCE") is not None:
        pass
    else:
        m=int(m/10) # reduce system memory requirements for basic tests, otherwise some tests eat too much system memory

    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        X = X.astype(np.float32)

    # Exact scikit impl
    sklearn_tsvd = sklearnsvd(algorithm="randomized", n_components=k, random_state=42)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu tsvd run")
    start_time = time.time()
    h2o4gpu_tsvd = TruncatedSVDH2O(n_components=k)
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

    # Override run_h2o False default if environ exists
    import os
    end_sk = time.time() - time.time()
    if os.getenv("CHECKPERFORMANCE") is not None:
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

    return end_time, end_sk

def run_bench(m=5000000, n=10, k=9, convert_to_float32=False):
    results = func(m, n, k, convert_to_float32=convert_to_float32)
    import os
    if os.getenv("CHECKPERFORMANCE") is not None:
        assert results[0] <= results[1], "h2o4gpu tsvd is not faster than sklearn for m = %s and n = %s" % (m,n)
    # filename = 'bench_results.csv'
    #
    # if os.path.exists(filename):
    #     append_write = 'a'  # append if already exists
    # else:
    #     append_write = 'w'  # make a new file if not
    #
    # timings = open(filename, append_write)
    # timings.write(str(results[0])+","+str(results[1])+'\n')
    # timings.close()


#Timings from ARPACK
#1M x 1000, k = 10 -> gpu took 27 sec, sklearn took 198 sec (7x speed up)
#100K x 10K, k = 100 -> gpu took 80 sec,  sklearn took 738 sec (9x speed up)
#5M x 100, k =5 -> gpu took 14 sec, sklearn took 57 sec (4x speed up)
#5M x 10 -> gpu took 3 sec, sklearn took 2 sec (a little bit slower)
# these currently take about 40-20GB system memory when doing sklearn tests
def test_tsvd_error_k2(): run_bench(m=5000000, n=10, k=2)
def test_tsvd_error_k5(): run_bench(m=5000000, n=100, k=5)
def test_tsvd_error_k10(): run_bench(m=1000000, n=1000, k=10)
def test_tsvd_error_k100(): run_bench(m=100000, n=10000, k=100)
def test_tsvd_error_k2_float32(): run_bench(m=5000000, n=10, k=2, convert_to_float32=True)
def test_tsvd_error_k5_float32(): run_bench(m=5000000, n=100, k=5, convert_to_float32=True)
def test_tsvd_error_k10_float32(): run_bench(m=1000000, n=1000, k=10, convert_to_float32=True)
def test_tsvd_error_k100_float32(): run_bench(m=100000, n=10000, k=100, convert_to_float32=True)
