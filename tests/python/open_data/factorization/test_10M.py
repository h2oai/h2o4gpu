import numpy as np
import scipy
import scipy.sparse
import h2o4gpu


def test_factorization():
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

    factorization = h2o4gpu.solvers.FactorizationH2O(
        50, 0.1, max_iter=10, double_precision=False)
    X = scipy.sparse.csc_matrix((R_csc_data, R_csc_indices, R_csc_indptr))
    X_test = scipy.sparse.coo_matrix(
        (R_test_coo_data, (R_test_coo_row, R_test_coo_col)), shape=X.shape)
    factorization.fit(X, X_test)


if __name__ == '__main__':
    test_factorization()
