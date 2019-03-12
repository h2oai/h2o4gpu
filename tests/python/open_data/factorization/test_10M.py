import numpy as np
import scipy
import scipy.sparse
import h2o4gpu
from sklearn.metrics import mean_squared_error
from math import sqrt


def _load_train_test():
    R_csc_data = np.fromfile(
        'open_data/factorization/R_train_csc.data.bin', dtype=np.float32)
    R_csc_indices = np.fromfile(
        'open_data/factorization/R_train_csc.indices.bin', dtype=np.int32)
    R_csc_indptr = np.fromfile(
        'open_data/factorization/R_train_csc.indptr.bin', dtype=np.int32)

    R_test_coo_col = np.fromfile(
        'open_data/factorization/R_test_coo.col.bin', dtype=np.int32)
    R_test_coo_row = np.fromfile(
        'open_data/factorization/R_test_coo.row.bin', dtype=np.int32)
    R_test_coo_data = np.fromfile(
        'open_data/factorization/R_test_coo.data.bin', dtype=np.float32)
    X = scipy.sparse.csc_matrix((R_csc_data, R_csc_indices, R_csc_indptr))
    X_test = scipy.sparse.coo_matrix(
        (R_test_coo_data, (R_test_coo_row, R_test_coo_col)), shape=X.shape)
    return X, X_test


def test_factorization_memory_leak():
    for i in range(100):
        X, _ = _load_train_test()
        factorization = h2o4gpu.solvers.FactorizationH2O(10, 0.1, max_iter=1)
        factorization.fit(X)


def test_factorization_fit_predict():
    X, X_test = _load_train_test()
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        50, 0.1, max_iter=10)
    factorization.fit(X, scores=scores)
    X_pred = factorization.predict(X.tocoo())
    not_nan = ~np.isnan(X_pred.data)
    assert np.allclose(sqrt(mean_squared_error(
        X.data[not_nan], X_pred.data[not_nan])), scores[-1][0])


def test_early_stop():
    X, X_test = _load_train_test()
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        50, 0.01, max_iter=10000)
    factorization.fit(X, scores=scores, X_test=X_test,
                      early_stopping_rounds=10, verbose=True)
    best = factorization.best_iteration
    for i in range(best, best + 10, 1):
        assert scores[best][1] <= scores[i][1]


def test_multi_batches():
    X, X_test = _load_train_test()
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        90, 0.1, max_iter=40)
    factorization.fit(X, scores=scores, X_BATCHES=2, THETA_BATCHES=2)
    X_pred = factorization.predict(X.tocoo())
    not_nan = ~np.isnan(X_pred.data)
    assert np.allclose(sqrt(mean_squared_error(
        X.data[not_nan], X_pred.data[not_nan])), scores[-1][0])


if __name__ == '__main__':
    test_early_stop()
    test_factorization_fit_predict()
    test_multi_batches()
