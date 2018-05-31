import numpy as np
import time
import sys
import logging
import csv
from h2o4gpu.solvers import TruncatedSVDH2O
print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func_bench(m=2000, n = 20, k = 5):
    np.random.seed(1234)

    X = np.random.rand(m,n)

    #Warm start
    W = np.random.rand(1000,5)
    print('Cusolver Warm Start')
    h2o4gpu_tsvd_cusolver = TruncatedSVDH2O(n_components=3, algorithm="cusolver", random_state=42)
    h2o4gpu_tsvd_cusolver.fit(W)
    print('Power Warm Start')
    h2o4gpu_tsvd_power = TruncatedSVDH2O(n_components=3, algorithm="power", tol = 1e-5, n_iter=100, random_state=42, verbose=True)
    h2o4gpu_tsvd_power.fit(W)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix with k=" + str(k))
    print("\n")

    cusolver_sum_time = 0
    power_sum_time = 0
    for i in range(5):
        start_time_cusolver = time.time()
        print("CUSOLVER Bencmark on iteration " + str(i))
        h2o4gpu_tsvd_cusolver.n_components = k
        h2o4gpu_tsvd_cusolver.fit(X)
        end_time_cusolver = time.time() - start_time_cusolver
        cusolver_sum_time +=end_time_cusolver
        print("Took cusolver " + str(end_time_cusolver) + " seconds on iteration " + str(i))

        print("Sleep before Power on iteration " + str(i))
        time.sleep(5)

        start_time_power = time.time()
        print("POWER Bencmark on iteration " + str(i))
        h2o4gpu_tsvd_power.n_components = k
        h2o4gpu_tsvd_power.fit(X)
        end_time_power = time.time() - start_time_power
        power_sum_time += end_time_power
        print("Took power method " + str(end_time_power) + " seconds on iteration " + str(i))

    #Benchmarks
    ########################################################################
    dim = str(m) + "by" + str(n)
    with open('power_cusolver_avg_run.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['cusolver', str(cusolver_sum_time/5), dim, str(k)])
        csvwriter.writerow(['power', str(power_sum_time/5), dim, str(k)])
        csvfile.close()
    #########################################################################

def func(m=2000, n = 20, k = 5):
    np.random.seed(1234)

    X = np.random.rand(m,n)

    print("SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("\n")

    start_time_cusolver = time.time()
    print("CUSOLVER")
    h2o4gpu_tsvd_cusolver = TruncatedSVDH2O(n_components=k, algorithm="cusolver", random_state=42)
    h2o4gpu_tsvd_cusolver.fit(X)
    end_time_cusolver = time.time() - start_time_cusolver
    print("Took cusolver " + str(end_time_cusolver) + " seconds")

    start_time_power = time.time()
    print("POWER")
    h2o4gpu_tsvd_power = TruncatedSVDH2O(n_components=k, algorithm="power", tol = 1E-50, n_iter=2000, random_state=42, verbose=True)
    h2o4gpu_tsvd_power.fit(X)
    end_time_power = time.time() - start_time_power
    print("Took power method " + str(end_time_power) + " seconds")

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
    rtol = 1E-5
    assert np.allclose(h2o4gpu_tsvd_cusolver.singular_values_, h2o4gpu_tsvd_power.singular_values_, rtol=rtol)

    print("Checking explained variance")
    rtol = 1E-3
    assert np.allclose(h2o4gpu_tsvd_cusolver.explained_variance_, h2o4gpu_tsvd_power.explained_variance_, rtol=rtol)

    print("Checking explained variance ratio")
    assert np.allclose(h2o4gpu_tsvd_cusolver.explained_variance_ratio_, h2o4gpu_tsvd_power.explained_variance_ratio_, rtol=rtol)

def test_tsvd_power_k7(): func(k=7)
def test_tsvd_power_k6(): func(k=6)
def test_tsvd_power_k5(): func(k=5)
def test_tsvd_power_k4(): func(k=4)
def test_tsvd_power_k3(): func(k=3)