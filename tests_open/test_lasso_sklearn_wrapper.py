import time
import sys
import os
import numpy as np
import logging
import pandas as pd

print(sys.path)

from h2o4gpu.util.testing_utils import find_file, run_glm
import h2o4gpu
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.DEBUG)


# Function to check fall back to sklearn
def test_fit_simple_backupsklearn(backend='auto'):
    df = pd.read_csv("./open_data/simple.txt", delim_whitespace=True)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    Solver = h2o4gpu.Lasso

    enet = Solver(glm_stop_early=False, backend=backend)
    print("h2o4gpu fit()")
    enet.fit(X, y)
    print("h2o4gpu predict()")
    print(enet.predict(X))
    print("h2o4gpu score()")
    print(enet.score(X,y))

    enet_wrapper = Solver(positive=True, random_state=1234, backend=backend)
    print("h2o4gpu scikit wrapper fit()")
    enet_wrapper.fit(X, y)
    print("h2o4gpu scikit wrapper predict()")
    print(enet_wrapper.predict(X))
    print("h2o4gpu scikit wrapper score()")
    print(enet_wrapper.score(X, y))

    from h2o4gpu.linear_model.coordinate_descent import LassoSklearn
    enet_sk = LassoSklearn(positive=True, random_state=1234)
    print("Scikit fit()")
    enet_sk.fit(X, y)
    print("Scikit predict()")
    print(enet_sk.predict(X))
    print("Scikit score()")
    print(enet_sk.score(X, y))

    enet_sk_coef = csr_matrix(enet_sk.coef_, dtype=np.float32).toarray()
    enet_sk_sparse_coef = csr_matrix(enet_sk.sparse_coef_, dtype=np.float32).toarray()

    if backend!='h2o4gpu':
        print(enet_sk.coef_)
        print(enet_sk.sparse_coef_)

        print(enet_sk_coef)
        print(enet_sk_sparse_coef)

        print(enet_wrapper.coef_)
        print(enet_wrapper.sparse_coef_)

        print(enet_sk.intercept_)
        print(enet_wrapper.intercept_)

        print(enet_sk.n_iter_)
        print(enet_wrapper.n_iter_)

        print(enet_wrapper.time_prepare)
        print(enet_wrapper.time_upload_data)
        print(enet_wrapper.time_fitonly)

        assert np.allclose(enet_wrapper.coef_, enet_sk_coef)
        assert np.allclose(enet_wrapper.intercept_, enet_sk.intercept_)
        assert np.allclose(enet_wrapper.n_iter_, enet_sk.n_iter_)


def test_sklearn_lasso(): test_fit_simple_backupsklearn()
def test_sklearn_lasso2(): test_fit_simple_backupsklearn(backend='sklearn')
def test_sklearn_lasso3(): test_fit_simple_backupsklearn(backend='h2o4gpu')


if __name__ == '__main__':
    test_sklearn_lasso()
    test_sklearn_lasso2()
    test_sklearn_lasso3()
