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
    from numpy.random import RandomState
    import sys
    import time
    import os
    import numpy as np
    import h2o4gpu
    import logging
    from daal.data_management import HomogenNumericTable
    from daal.algorithms.linear_regression import training as linear_training
    from daal.algorithms.linear_regression import prediction as linear_prediction
    from h2o4gpu.solvers.daal_solver.daal_data import getNumpyArray, getNumpyShape
    from numpy.linalg.tests.test_linalg import assert_almost_equal
    from numpy.ma.testutils import assert_array_almost_equal

    logging.basicConfig(level=logging.DEBUG)

    seeded = RandomState(42)

    from h2o4gpu.solvers.linear_regression import LinearRegression
    from h2o4gpu.solvers import DLR

    def test_fit_linear_regression_daal_vs_sklearn():

        trainData = seeded.rand(200,10)
        trainDependentVariables = seeded.rand(200,2)

        testData = seeded.rand(50,10)
        testDependentVariables = seeded.rand(50,2)

        solver_daal = LinearRegression(
            fit_intercept=True,
            normalize=False,
            verbose=True,
            backend='daal')

        start_daal = time.time()
        solver_daal.fit(trainData, trainDependentVariables)
        end_daal = time.time()

        solver_sk = LinearRegression(normalize=True)
        start_sklearn = time.time()
        solver_sk.fit(trainData, trainDependentVariables)
        end_sklearn = time.time()

        print("TEST FIT Sklearn vs Daal")
        print("Time taken in daal: {}".format(end_daal-start_daal))
        print("Time taken in sklearn: {}".format(end_sklearn-start_sklearn))
        print("DONE.")

        if os.getenv("CHECKPERFORMANCE") is not None:
            assert end_daal - start_daal <= end_sklearn - start_sklearn

        #DLR.print_table(trained.getBeta(), "Linear Regression coefficients:")
        #prediction = solver.predict(testData)
        #DLR.print_table(prediction, "Linear Regression prediction results: (first 10 rows):", 10)
    def test_linear_regression_simple():

        # calculate beta coefficients
        x = np.array([0.,2.,3.]).reshape(3,1)
        y = x

        nt_x = nt_y = HomogenNumericTable(x)

        lr_alg = linear_training.Batch()
        lr_alg.input.set(linear_training.data, nt_x)
        lr_alg.input.set(linear_training.dependentVariables, nt_y)
        result = lr_alg.compute()
        model = result.get(linear_training.model)
        beta_coeff = model.getBeta()
        np_beta_coeff = getNumpyArray(beta_coeff)

        res_beta_coeff = np.array([0,1]).reshape(1,2)

        assert_almost_equal(res_beta_coeff, np_beta_coeff)

        # predict
        lr_alg_predict = linear_prediction.Batch()
        lr_alg_predict.input.setModel(linear_prediction.model, model)
        lr_alg_predict.input.setTable(linear_prediction.data, nt_x)
        result = lr_alg_predict.compute()
        np_predict = getNumpyArray(result.get(linear_prediction.prediction))
        assert_array_almost_equal(x, np_predict)

    def get_random_array(rows=10, columns=9):
        x = np.random.rand(rows, columns)
        y = np.random.rand(rows, 1)

        return (x,y)

    def test_overfitting(rows=10, columns=9):
        '''
        overfitting - more features than data points
        for n(number of observation) > p (number of variables)
        in this case, the least squares estimates tend to have low variance,
        and hence performs well on test observations
        for n <= p, in this case a lot of variability in the least squares fit
        for n << p, no longer a unique least squares coefficient estimate, the variance
        is infinite so the method cannot be used at all.
        for the last second cases, one has to use ridgit regression, lasso, or 
        reduct dimension (subset selection, e.g. scikit does this approach)
        '''
        assert rows > columns, "More features than data points in linear regression!"

    def get_daal_prediction(x=np.array([1,2,3]), y=np.array([1,2,3])):
        ntX = HomogenNumericTable(x)
        ntY = HomogenNumericTable(y)

        lr_train = linear_training.Batch()
        lr_train.input.set(linear_training.data, ntX)
        lr_train.input.set(linear_training.dependentVariables, ntY)
        result = lr_train.compute()
        model = result.get(linear_training.model)

        lr_predict = linear_prediction.Batch()
        lr_predict.input.setModel(linear_prediction.model, model)
        lr_predict.input.setTable(linear_prediction.data, ntX)
        result = lr_predict.compute()

        np_predicted = getNumpyArray(result.get(linear_prediction.prediction))
        # assert the same as the initial dependent variable
        assert_array_almost_equal(y, np_predicted)
        return np_predicted

    def get_scikit_prediction(x=np.array([1,2,3]), y=np.array([1,2,3])):

        from sklearn.linear_model.base import LinearRegression as ScikitLinearRegression

        regression = ScikitLinearRegression()
        regression.fit(x, y)

        return regression.predict(x)

    def test_linear_regression_against_scikit(rows=10, columns=9):
        '''
        Test prediction daal against scikit
        Test for overfitting
        :param rows:
        :param columns:
        '''
        inout = get_random_array(rows, columns)
        test_overfitting(rows, columns)
        x = inout[0]
        y = inout[1]
        daal_predicted = get_daal_prediction(x, y)
        scik_predicted = get_scikit_prediction(x, y)

        assert_array_almost_equal(daal_predicted, scik_predicted)

    def test_coeff_size(rows=10, columns=9):
        '''
        number of beta coefficients (with intercept flag on)
        is the same number as size of data sample
        '''
        inout = get_random_array(rows, columns)
        test_overfitting(rows, columns)
        x = inout[0]
        y = inout[1]

        ntX = HomogenNumericTable(x)
        ntY = HomogenNumericTable(y)

        lr_train = linear_training.Batch()
        lr_train.input.set(linear_training.data, ntX)
        lr_train.input.set(linear_training.dependentVariables, ntY)
        result = lr_train.compute()
        model = result.get(linear_training.model)
        beta_coeff = model.getBeta()
        np_beta = getNumpyArray(beta_coeff)

        assert y.transpose().shape == np_beta.shape, "Dependent variable size must have\
            the same size as Beta coefficient"

    def test_intercept_flag(rows=10, columns=9):
        inout = get_random_array(rows, columns)
        test_overfitting(rows, columns)
        x = inout[0]
        y = inout[1]

        ntX = HomogenNumericTable(x)
        ntY = HomogenNumericTable(y)

        lr_train = linear_training.Batch()
        lr_train.input.set(linear_training.data, ntX)
        lr_train.input.set(linear_training.dependentVariables, ntY)
        result = lr_train.compute()
        model = result.get(linear_training.model)
        beta_coeff = model.getBeta()
        np_beta = getNumpyArray(beta_coeff)
        daal_intercept = np_beta[0,0]

        from sklearn.linear_model.base import LinearRegression as ScikitLinearRegression
        regression = ScikitLinearRegression()
        regression.fit(x, y)

        scikit_intercept = regression.intercept_
        assert_array_almost_equal(scikit_intercept, [daal_intercept])


    def test_linear_regression_normalized(): test_fit_linear_regression_daal_vs_sklearn()
    def test_linear_regression(): test_linear_regression_simple()
    def test_linear_regression_param_3_2(): test_linear_regression_against_scikit(rows=3, columns=2)
    def test_linear_regression_with_sc(): test_linear_regression_against_scikit()
    def test_beta(): 
        test_coeff_size(rows=10, columns=9)
        test_intercept_flag(rows=10, columns=9)

    if __name__ == '__main__':
        test_linear_regression_normalized()
        test_linear_regression()
        test_linear_regression_param()
        test_beta()
