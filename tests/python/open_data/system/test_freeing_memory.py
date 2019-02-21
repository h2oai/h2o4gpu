import numpy as np
import h2o4gpu
import random
import time
import h2o4gpu.solvers.utils
import pytest

def run(model, m, n, s):
    X = np.random.uniform(-100, 100, size = (m, n))
    coefs = np.random.randn(n)
    const_coef = np.random.randn(1)

    zero_coef_loc = random.sample(range(n), s)
    coefs[zero_coef_loc] = 0

    y = np.dot(X, coefs) + const_coef
    model.fit(X, y)

def run_stress(model):
    runs = 4
    n = 2048
    m_ratio = 2
    s_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    m = int(m_ratio * n)
    for s_ratio in s_ratios:
        s = int(s_ratio * n)
        for i in range(runs):
            run(model, m, n, s)

def test_ridge(): run_stress(h2o4gpu.Ridge())
def test_lasso(): run_stress(h2o4gpu.Lasso())
# TODO: find out why it hangs
@pytest.mark.skip(reason="It Hangs")
def test_elasticnet(): run_stress(h2o4gpu.ElasticNet())
def test_pca(): run_stress(h2o4gpu.PCA())
def test_truncatedsvd(): run_stress(h2o4gpu.TruncatedSVD())

if __name__ == "__main__":
    test_ridge()
    test_lasso()
    #test_elasticnet()
    test_pca()
    test_truncatedsvd()