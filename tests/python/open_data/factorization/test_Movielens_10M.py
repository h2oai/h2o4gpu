import numpy as np
import scipy
import scipy.sparse
import h2o4gpu
from sklearn.metrics import mean_squared_error


def _load_train_test():
    # preprocessed http://files.grouplens.org/datasets/movielens/ml-10m-README.html
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
        factorization = h2o4gpu.solvers.FactorizationH2O(20, 0.1, max_iter=5)
        factorization.fit(X)


def factorization_fit_predict(F, BATCHES=1):
    X, X_test = _load_train_test()
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        F, 0.001, max_iter=10)
    factorization.fit(X, scores=scores, X_BATCHES=BATCHES,
                      THETA_BATCHES=BATCHES, verbose=True)
    X_pred = factorization.predict(X.tocoo())
    not_nan = ~np.isnan(X_pred.data)
    assert np.count_nonzero(not_nan) == 9000048
    assert np.allclose(np.sqrt(mean_squared_error(
        X.data[not_nan], X_pred.data[not_nan])), scores[-1][0])
    last = np.inf
    for score, _ in scores:
        assert score < 1.0
        assert last >= score
        last = score


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


def test_factorization_fit_predict_10(): factorization_fit_predict(10)


def test_factorization_fit_predict_20(): factorization_fit_predict(20)


def test_factorization_fit_predict_30(): factorization_fit_predict(30)


def test_factorization_fit_predict_40(): factorization_fit_predict(40)


def test_factorization_fit_predict_50(): factorization_fit_predict(50)


def test_factorization_fit_predict_60(): factorization_fit_predict(60)


def test_factorization_fit_predict_70(): factorization_fit_predict(70)


def test_factorization_fit_predict_80(): factorization_fit_predict(80)


def test_factorization_fit_predict_40_2_batches(): factorization_fit_predict(40, 2)


def test_factorization_fit_predict_50_2_batches(): factorization_fit_predict(50, 2)


def test_factorization_fit_predict_60_2_batches(): factorization_fit_predict(60, 2)


def test_factorization_fit_predict_70_2_batches(): factorization_fit_predict(70, 2)


def test_factorization_fit_predict_80_2_batches(): factorization_fit_predict(80, 2)


def test_factorization_fit_predict_90_2_batches(
): factorization_fit_predict(90, BATCHES=2)


def test_factorization_fit_predict_100_2_batches(
): factorization_fit_predict(100, BATCHES=2)


def test_factorization_fit_predict_110_2_batches(
): factorization_fit_predict(110, BATCHES=2)


def test_factorization_fit_predict_100_3_batches(
): factorization_fit_predict(100, BATCHES=3)


def test_factorization_fit_predict_110_3_batches(
): factorization_fit_predict(110, BATCHES=3)


if __name__ == '__main__':
    test_factorization_fit_predict_30()
    # test_factorization_memory_leak()
    # test_early_stop()
    # test_factorization_fit_predict_70()
    # test_factorization_fit_predict_100_3_batches()
