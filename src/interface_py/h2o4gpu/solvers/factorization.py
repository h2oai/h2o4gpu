# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
Matrix factorization solver.

:copyright: 2017-2019 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np
import scipy
import scipy.sparse


def _get_sparse_matrixes(X):
    '''Create csc, csr and coo sparse matrix from any of the above

    Arguments:
        X {array-like, csc, csr or coo sparse matrix}

    Returns:
        csc, csr, coo
    '''

    X_coo = X_csc = X_csr = None
    if scipy.sparse.isspmatrix_coo(X):
        X_coo = X
        X_csr = X_coo.tocsr(True)
        X_csc = X_coo.tocsc(True)
    elif scipy.sparse.isspmatrix_csr(X):
        X_csr = X
        X_csc = X_csr.tocoo(True)
        X_coo = X_csr.tocsc(True)
    elif scipy.sparse.isspmatrix_csc(X):
        X_csc = X
        X_csr = X_csc.tocsr(True)
        X_coo = X_csc.tocoo(True)
    else:
        assert False, "only coo, csc and csr sparse matrixes are supported"
    return X_csc, X_csr, X_coo


class FactorizationH2O(object):
    '''Matrix Factorization on GPU with Alternating Least Square (ALS) algorithm.

    Factors a sparse rating matrix X (m by n, with N_z non-zero elements)
    into a m-by-f and a f-by-n matrices.

    Parameters
    ----------
    f int
        decomposition size
    lambda_ float
        lambda regularization
    max_iter int, default: 100
        number of training iterations
    double_precision bool, default: False
        use double presition, not yet supported
    thetaT {array-like} shape (n, f),  default: None
        initial theta matrix
    XT {array-like} shape (m, f), default: None
        initial XT matrix

    Attributes
    ----------
    XT {array-like} shape (m, f)
        XT matrix contains User's features
    thetaT {array-like} shape (n, f)
        transposed theta matrix, item's features

    Warnings
    --------
    Matrixes ``XT`` and ``thetaT`` may contain nan elements. This is because in some datasets,
    there are users or items with no ratings in training set. That results in solutions of
    a system of linear equations becomes nan. Such elements can be easily removed with numpy
    functions like numpy.nan_to_num, but existence of them may be usefull for troubleshooting
    perposes.

    '''

    def __init__(self, f, lambda_, max_iter=100, double_precision=False, thetaT=None, XT=None):
        assert not double_precision, 'double precision is not yet supported'
        assert f % 10 == 0, 'f has to be a multiple of 10'
        self.f = f
        self.lambda_ = lambda_
        self.double_precision = double_precision
        self.dtype = np.float64 if self.double_precision else np.float32
        self.thetaT = thetaT
        self.XT = XT
        self.max_iter = max_iter

    def _load_lib(self):
        from ..libs.lib_utils import GPUlib

        gpu_lib = GPUlib().get(1)
        return gpu_lib

    def fit(self, X, y=None, X_test=None, X_BATCHES=1, THETA_BATCHES=1, early_stopping_rounds=None, verbose=False, scores=None):
        #pylint: disable=unused-argument
        '''Learn model from rating matrix X.

        Parameters
        ----------
        X {array-like, sparse matrix}, shape (m, n)
            Data matrix to be decomposed.
        y None
            Ignored
        X_test {array-like, coo sparse matrix}, shape (m, n)
            Data matrix for cross validation.
        X_BATCHES int, default: 1
            Batches to split XT, increase this parameter in case out of memory error.
        THETA_BATCHES int, default: 1
            Batches to split theta, increase this parameter in case out of memory error.
        early_stopping_rounds int, default: None
            Activates early stopping. Cross validation error needs to decrease
            at least every <early_stopping_rounds> round(s) to continue training. Requires <X_test>.
            Returns the model from the last iteration (not the best one). If early stopping occurs,
            the model will have three additional fields: best_cv_score, best_train_score and best_iteration.
        verbose bool, default: False
            Prints training and validation score(if applicable) on each iteration.
        scores {list}
            List of tuples with train, cv score for every iteration.

        Returns
        -------
        self : returns an instance of self.

        '''

        csc_X, csr_X, coo_X = _get_sparse_matrixes(X)

        if early_stopping_rounds is not None:
            assert X_test is not None, 'X_test is mandatory with early stopping'
        if X_test is not None:
            assert scipy.sparse.isspmatrix_coo(
                X_test), 'X_test must be a coo sparse scipy matrix'
            assert X.shape == X_test.shape
            assert X_test.dtype == self.dtype

        assert X.dtype == self.dtype

        coo_X_test = X_test

        lib = self._load_lib()
        if self.double_precision:
            make_data = lib.make_factorization_data_double
            run_step = lib.run_factorization_step_double
            factorization_score = lib.factorization_score_double
            copy_fecatorization_result = lib.copy_fecatorization_result_double
            free_data = lib.free_data_double
        else:
            make_data = lib.make_factorization_data_float
            run_step = lib.run_factorization_step_float
            factorization_score = lib.factorization_score_float
            copy_fecatorization_result = lib.copy_fecatorization_result_float
            free_data = lib.free_data_float

        m = coo_X.shape[0]
        n = coo_X.shape[1]
        nnz = csc_X.nnz
        if coo_X_test is None:
            nnz_test = 0
        else:
            nnz_test = coo_X_test.nnz

        if self.thetaT is None:
            self.thetaT = np.random.rand(n, self.f).astype(self.dtype)
        else:
            assert self.thetaT.dtype == self.dtype

        if self.XT is None:
            self.XT = np.random.rand(m, self.f).astype(self.dtype)
        else:
            assert self.XT.dtype == self.dtype

        csrRowIndexDevicePtr = None
        csrColIndexDevicePtr = None
        csrValDevicePtr = None
        cscRowIndexDevicePtr = None
        cscColIndexDevicePtr = None
        cscValDevicePtr = None
        cooRowIndexDevicePtr = None
        cooColIndexDevicePtr = None
        cooValDevicePtr = None
        thetaTDevice = None
        XTDevice = None
        cooRowIndexTestDevicePtr = None
        cooColIndexTestDevicePtr = None
        cooValTestDevicePtr = None

        status, csrRowIndexDevicePtr, csrColIndexDevicePtr, csrValDevicePtr, \
            cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr, \
            cooRowIndexDevicePtr, cooColIndexDevicePtr, cooValDevicePtr, \
            thetaTDevice, XTDevice, cooRowIndexTestDevicePtr, \
            cooColIndexTestDevicePtr, cooValTestDevicePtr = make_data(  # pylint: disable=W0212
                m, n, self.f, nnz, nnz_test, csr_X.indptr, csr_X.indices, csr_X.data,
                csc_X.indices, csc_X.indptr, csc_X.data,
                coo_X.row, coo_X.col, coo_X.data,
                self.thetaT, self.XT, coo_X_test.row if coo_X_test is not None else None,
                coo_X_test.col if coo_X_test is not None else None, coo_X_test.data if coo_X_test is not None else None,
                csrRowIndexDevicePtr, csrColIndexDevicePtr, csrValDevicePtr, cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr,
                cooRowIndexDevicePtr, cooColIndexDevicePtr, cooValDevicePtr,
                thetaTDevice, XTDevice, cooRowIndexTestDevicePtr,
                cooColIndexTestDevicePtr, cooValTestDevicePtr)

        assert status == 0, 'Failure uploading the data'

        self.best_train_score = np.inf
        self.best_cv_score = np.inf
        self.best_iteration = -1
        cv_score = train_score = np.inf

        for i in range(self.max_iter):
            status = run_step(m,
                              n,
                              self.f,
                              nnz,
                              self.lambda_,
                              csrRowIndexDevicePtr,
                              csrColIndexDevicePtr,
                              csrValDevicePtr,
                              cscRowIndexDevicePtr,
                              cscColIndexDevicePtr,
                              cscValDevicePtr,
                              thetaTDevice,
                              XTDevice,
                              X_BATCHES,
                              THETA_BATCHES)
            if verbose or scores is not None:
                result = factorization_score(m,
                                             n,
                                             self.f,
                                             nnz,
                                             self.lambda_,
                                             thetaTDevice,
                                             XTDevice,
                                             cooRowIndexDevicePtr,
                                             cooColIndexDevicePtr,
                                             cooValDevicePtr)
                train_score = result[0]
            if X_test is not None and (verbose or early_stopping_rounds is not None or scores is not None):
                result = factorization_score(m,
                                             n,
                                             self.f,
                                             nnz_test,
                                             self.lambda_,
                                             thetaTDevice,
                                             XTDevice,
                                             cooRowIndexTestDevicePtr,
                                             cooColIndexTestDevicePtr,
                                             cooValTestDevicePtr)
                cv_score = result[0]
            if verbose:
                print("iteration {0} train: {1} cv: {2}".format(
                    i, train_score, cv_score))
            if scores is not None:
                scores.append((train_score, cv_score))

            if early_stopping_rounds is not None:
                if self.best_cv_score > cv_score:
                    self.best_cv_score = cv_score
                    self.best_train_score = train_score
                    self.best_iteration = i
                if (i - self.best_iteration) > early_stopping_rounds:
                    if verbose:
                        print('best iteration:{0} train: {1} cv: {2}'.format(
                            self.best_iteration, self.best_train_score, self.best_cv_score))
                    break

        lib.free_data_int(csrRowIndexDevicePtr)
        lib.free_data_int(csrColIndexDevicePtr)
        free_data(csrValDevicePtr)
        lib.free_data_int(cscRowIndexDevicePtr)
        lib.free_data_int(cscColIndexDevicePtr)
        free_data(cscValDevicePtr)
        lib.free_data_int(cooRowIndexDevicePtr)
        lib.free_data_int(cooColIndexDevicePtr)
        free_data(cooValDevicePtr)
        lib.free_data_int(cooRowIndexTestDevicePtr)
        lib.free_data_int(cooColIndexTestDevicePtr)
        free_data(cooValTestDevicePtr)

        copy_fecatorization_result(self.XT, XTDevice, m * self.f)
        copy_fecatorization_result(self.thetaT, thetaTDevice, n * self.f)

        free_data(thetaTDevice)
        free_data(XTDevice)

        return self

    def predict(self, X):
        '''Predict none zero elements of coo sparse matrix X according to the fitted model.

        Parameters
        ----------
            X {array-like, sparse coo matrix} shape (m, n)
                Data matrix in coo format. Values are ignored.

        Returns
        -------
            {array-like, sparse coo matrix} shape (m, n)
                Predicted values.

        '''

        assert self.XT is not None and self.thetaT is not None, 'tranform is invoked on an unfitted model'
        assert scipy.sparse.isspmatrix_coo(
            X), 'convert X to coo sparse matrix'
        assert X.dtype == self.dtype
        a = np.take(self.XT, X.row, axis=0)
        b = np.take(self.thetaT, X.col, axis=0)
        val = np.sum(a * b, axis=1)
        return scipy.sparse.coo_matrix((val, (X.row, X.col)), shape=X.shape)
