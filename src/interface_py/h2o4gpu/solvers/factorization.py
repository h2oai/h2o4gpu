# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
KMeans clustering solver.

:copyright: 2017-2019 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np
import scipy
import scipy.sparse


class FactorizationH2O(object):
    '''[summary]

    Arguments:
        object {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''

    def __init__(self, f, lambda_, max_iter=100, double_precision=False, thetaT=None, XT=None):
        self.max_iter = max_iter
        assert f % 10 == 0, 'f has to be a multiple of 10'
        self.f = f
        self.lambda_ = lambda_
        self.double_precision = double_precision
        self.thetaT = thetaT
        self.XT = XT

    def _load_lib(self):
        from ..libs.lib_utils import GPUlib

        gpu_lib = GPUlib().get(1)
        return gpu_lib

    def fit(self, X, X_test=None, X_BATCHES=1, THETA_BATCHES=1, early_stopping_rounds=None, verbose=False):
        '''[summary]

        Arguments:
            X {[type]} -- [description]

        Keyword Arguments:
            X_test {[type]} -- [description] (default: {None})
            X_BATCHES {int} -- [description] (default: {1})
            THETA_BATCHES {int} -- [description] (default: {1})
            early_stopping_rounds {[type]} -- [description] (default: {None})
            verbose {bool} -- [description] (default: {False})
        '''

        if early_stopping_rounds is not None:
            assert X_test is not None, 'X_test is mandatory with early stopping'
        assert scipy.sparse.isspmatrix_csc(
            X), 'X must be a csc sparse scipy matrix'
        if X_test is not None:
            assert scipy.sparse.isspmatrix_coo(
                X_test), 'X_test must be a coo sparse scipy matrix'
            assert X.shape == X_test.shape

        dtype = np.float64 if self.double_precision else np.float32

        assert X.dtype == dtype
        assert X_test.dtype == dtype

        csc_X = X
        csr_X = csc_X.tocsr(True)
        coo_X = csc_X.tocoo(True)

        coo_X_test = X_test

        lib = self._load_lib()
        if self.double_precision:
            make_data = lib.make_factorization_data_double
            run_step = lib.run_factorization_step_double
            factorization_score = lib.factorization_score_double
        else:
            make_data = lib.make_factorization_data_float
            run_step = lib.run_factorization_step_float
            factorization_score = lib.factorization_score_float

        m = coo_X.shape[0]
        n = coo_X.shape[1]
        nnz = csc_X.nnz
        nnz_test = coo_X_test.nnz

        if self.thetaT is None:
            thetaT = np.random.rand(n, self.f).astype(dtype)
        else:
            thetaT = self.thetaT
            assert thetaT.dtype == dtype

        if self.XT is None:
            XT = np.random.rand(m, self.f).astype(dtype)
        else:
            XT = self.XT
            XT.dtype = dtype

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
                thetaT, XT, coo_X_test.row,
                coo_X_test.col, coo_X_test.data, csrRowIndexDevicePtr, csrColIndexDevicePtr,
                csrValDevicePtr, cscRowIndexDevicePtr, cscColIndexDevicePtr, cscValDevicePtr,
                cooRowIndexDevicePtr, cooColIndexDevicePtr, cooValDevicePtr,
                thetaTDevice, XTDevice, cooRowIndexTestDevicePtr,
                cooColIndexTestDevicePtr, cooValTestDevicePtr)

        assert status == 0, 'Failure uploading the data'

        best_CV = np.inf
        best_Iter = -1
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

            if early_stopping_rounds is not None:
                if best_CV > cv_score:
                    best_CV = cv_score
                    best_Iter = i
                if (i - best_Iter) > early_stopping_rounds:
                    break
