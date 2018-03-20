# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
try:
    __import__('daal')
except ImportError:
    import platform
    print("Daal is not supported. Architecture detected {}".format(platform.architecture()))
else:
    import os
    import time
    import numpy as np
    import logging
    from daal.algorithms import svd
    from daal.data_management import HomogenNumericTable
    from h2o4gpu.solvers.daal_solver.daal_data import getNumpyArray
    from h2o4gpu.solvers import TruncatedSVD
    from numpy.ma.testutils import assert_array_almost_equal

    logging.basicConfig(level=logging.DEBUG)

    def test_svd_simple():
        indata = np.array([[1,2],[3,4],[5,6],[7,8]])
        dataSource = HomogenNumericTable(indata)
        _in_rows, in_columns = indata.shape

        algorithm = svd.Batch(method=svd.defaultDense,
                              leftSingularMatrix=svd.requiredInPackedForm,
                              rightSingularMatrix=svd.requiredInPackedForm)

        algorithm.input.set(svd.data, dataSource)
        result = algorithm.compute()

        sigma = getNumpyArray(result.get(svd.singularValues))
        U = getNumpyArray(result.get(svd.leftSingularMatrix))
        V = getNumpyArray(result.get(svd.rightSingularMatrix))

        assert sigma.shape[1] == in_columns
        assert indata.shape == U.shape
        assert in_columns == V.shape[0] == V.shape[1]

        assert_array_almost_equal(np.array([[14.269, 0.6268]]), sigma, decimal=4)

        assert_array_almost_equal(np.array([[-0.152, -0.823],
                                        [-0.350, -0.421], 
                                        [-0.547, -0.020],
                                        [-0.745, 0.381 ]]),
                                        U, decimal=3)

        assert_array_almost_equal(np.array([[-0.641, -0.767],
                                            [0.767, -0.641 ]]),
                                            V, decimal=3)

    def test_svd_simple_check():
        indata = np.array([[1,3,4],[5,6,9],[1,2,3],[7,6,8]])
        dataSource = HomogenNumericTable(indata)

        algorithm = svd.Batch()
        algorithm.input.set(svd.data, dataSource)
        result = algorithm.compute()

        sigma = getNumpyArray(result.get(svd.singularValues))
        U = getNumpyArray(result.get(svd.leftSingularMatrix))
        V = getNumpyArray(result.get(svd.rightSingularMatrix))

        # create diagonal matrix of Singular values
        _rows, cols = sigma.shape
        d_sigma = sigma.reshape(cols,)
        outdata = np.dot(U, np.dot(np.diag(d_sigma), V))

        assert_array_almost_equal(outdata, indata)

    def get_random_array(rows=10, columns=9):
        x = np.random.rand(rows, columns)
        return x

    def test_svd_daal_vs_sklearn(rows=1000, columns=1000):
        indata = get_random_array(rows, columns)
        daal_input = HomogenNumericTable(indata)
        algorithm = svd.Batch()
        algorithm.input.set(svd.data, daal_input)

        start_sklearn = time.time()
        _U, s, _Vh = np.linalg.svd(indata, full_matrices=False)
        end_sklearn = time.time()

        start_daal = time.time()
        result = algorithm.compute()
        end_daal = time.time()

        if os.getenv("CHECKPERFORMANCE") is not None:
            assert(end_daal-start_daal <= end_sklearn-start_sklearn)

        sigma = getNumpyArray(result.get(svd.singularValues))
        _rows, cols = sigma.shape
        d_sigma = sigma.reshape(cols,)

        assert_array_almost_equal(d_sigma, s)

        print("SVD for matrix[{}][{}]".format(rows, columns))
        print("+ Sklearn SVD: {}".format(end_sklearn - start_sklearn))
        print("+ Sklearn Daal: {}".format(end_daal - start_daal))

    def test_tsvd_wrapper(rows = 100, cols = 100, k = 100):
        indata = get_random_array(rows, cols)
        start_sklearn = time.time()
        h2o4gpu_tsvd_sklearn = TruncatedSVD(n_components=k,
                                verbose=True,
                                backend='sklearn')
        h2o4gpu_tsvd_sklearn.fit(indata)
        end_sklearn = time.time()

        start_daal = time.time()
        h2o4gpu_tsvd_daal = TruncatedSVD(n_components=k,
                                         verbose=True,
                                         backend='daal')
        h2o4gpu_tsvd_daal.fit(indata)
        end_daal = time.time()

        print("H2o4GPU tsvd for backend=sklearn: {} seconds taken".format(end_sklearn-start_sklearn))
        print("H2o4GPU tsvd for backend=daal: {} seconds taken".format(end_daal-start_daal))

        sklearn_sigma = h2o4gpu_tsvd_sklearn.singular_values_
        daal_sigma = h2o4gpu_tsvd_daal.singular_values_
        print("H2o4GPU tsvd Sklearn Singular values: {}".format(sklearn_sigma))
        print("H2o4GPU tsvd Daal Singular values:    {}".format(daal_sigma))

        if os.getenv("CHECKPERFORMANCE") is not None:
            assert(end_daal-start_daal <= end_sklearn-start_sklearn)

    def test_svd():
        test_svd_simple()
        test_svd_simple_check()
    def test_svd_benchmark(): 
        test_svd_daal_vs_sklearn(20,20)
        #test_svd_daal_vs_sklearn(1000,1000)
        #test_svd_daal_vs_sklearn(2000,2000)
        #test_svd_daal_vs_sklearn(100000,2)
    def test_svd_wrapper():
        test_tsvd_wrapper(10, 10, 10)

    if __name__ == '__main__':
        print("SVD Testing")
        test_svd()
        test_svd_benchmark()
        test_svd_wrapper()
    
