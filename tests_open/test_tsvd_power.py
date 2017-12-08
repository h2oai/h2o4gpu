import numpy as np
import time
import sys
import logging
from h2o4gpu.solvers import TruncatedSVDH2O
import csv
print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=2000, n = 20, k = 5):
    np.random.seed(1234)

    X = np.random.rand(m,n)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("\n")

    start_time_cusolver = time.time()
    print("CUSOLVER")
    h2o4gpu_tsvd_cusolver = TruncatedSVDH2O(n_components=k, algorithm="cusolver")
    h2o4gpu_tsvd_cusolver.fit(X)
    end_time_cusolver = time.time() - start_time_cusolver
    print("Took cusolver " + str(end_time_cusolver) + " seconds")

    start_time_power = time.time()
    print("POWER")
    h2o4gpu_tsvd_power = TruncatedSVDH2O(n_components=k, algorithm="power", tol = 0.0)
    h2o4gpu_tsvd_power.fit(X)
    end_time_power = time.time() - start_time_power
    print("Took power method " + str(end_time_power) + " seconds")

    #Benchmarks
    ########################################################################
    # dim = str(m) + "by" + str(n)
    # import csv
    # with open('tsvd_bench.csv', 'a', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     csvwriter.writerow(['cusolver', str(end_time_cusolver), dim, str(k)])
    #     csvwriter.writerow(['power', str(end_time_power), dim, str(k)])
    #     csvfile.close()
    #########################################################################

    print("h2o4gpu cusolver components")
    print(h2o4gpu_tsvd_cusolver.components_)
    print("h2o4gpu cusolver singular values")
    print(h2o4gpu_tsvd_cusolver.singular_values_)
    print("h2o4gpu tsvd cusolver Explained Variance")
    print(h2o4gpu_tsvd_cusolver.explained_variance_)
    print("h2o4gpu tsvd cusolver Explained Variance Ratio")
    print(h2o4gpu_tsvd_cusolver.explained_variance_ratio_)

    print("h2o4gpu power components")
    print(h2o4gpu_tsvd_power.components_)
    print("h2o4gpu power singular values")
    print(h2o4gpu_tsvd_power.singular_values_)
    print("h2o4gpu tsvd power Explained Variance")
    print(h2o4gpu_tsvd_power.explained_variance_)
    print("h2o4gpu tsvd power Explained Variance Ratio")
    print(h2o4gpu_tsvd_power.explained_variance_ratio_)

    print("Checking singular values")
    assert np.allclose(h2o4gpu_tsvd_cusolver.singular_values_, h2o4gpu_tsvd_power.singular_values_, .001)

    print("Checking explained variance")
    assert np.allclose(h2o4gpu_tsvd_cusolver.explained_variance_, h2o4gpu_tsvd_power.explained_variance_, .001)

    print("Checking explained variance ratio")
    assert np.allclose(h2o4gpu_tsvd_cusolver.explained_variance_ratio_, h2o4gpu_tsvd_power.explained_variance_ratio_, .001)

def test_tsvd_power_k7(): func(k=7)
def test_tsvd_power_k6(): func(k=6)
def test_tsvd_power_k5(): func(k=5)
def test_tsvd_power_k4(): func(k=4)
def test_tsvd_power_k3(): func(k=3)
