# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from numpy.random import RandomState
import sys
import time
import os
import numpy as np
import h2o4gpu
import logging
from daal.data_management import HomogenNumericTable
from daal.algorithms.ridge_regression import training as ridge_training
from daal.algorithms.ridge_regression import prediction as ridge_prediction
from h2o4gpu.solvers.daal_solver.daal_data import getNumpyArray, getNumpyShape
from numpy.linalg.tests.test_linalg import assert_almost_equal
from numpy.ma.testutils import assert_array_almost_equal
from sklearn.linear_model import Ridge as ScikitRidgeRegression

logging.basicConfig(level=logging.DEBUG)

seeded = RandomState(42)

from h2o4gpu.solvers.ridge import Ridge
from h2o4gpu.solvers import DRR

def test_fit_ridge_regression_daal_vs_sklearn():

    trainData = seeded.rand(200,10)
    trainDependentVariables = seeded.rand(200,2)

    testData = seeded.rand(50,10)
    testDependentVariables = seeded.rand(50,2)

    solver_daal = Ridge(
        fit_intercept=True,
        normalize=False,
        verbose=True,
        backend='daal')

    start_daal = time.time()
    solver_daal.fit(trainData, trainDependentVariables)
    end_daal = time.time()

    solver_sk = ScikitRidgeRegression(alpha=1.0)
    start_sklearn = time.time()
    solver_sk.fit(trainData, trainDependentVariables)
    end_sklearn = time.time()
    
    print("TEST FIT Sklearn vs Daal of Ridge Regression")
    print("Time taken in daal: {}".format(end_daal-start_daal))
    print("Time taken in sklearn: {}".format(end_sklearn-start_sklearn))
    print("DONE.")

    if os.getenv("CHECKPERFORMANCE") is not None:
        assert end_daal - start_daal <= end_sklearn - start_sklearn

def test_ridge_regression_simple():

    # calculate beta coefficients
    x = np.array([0.,2.,3.]).reshape(3,1)
    y = x

    nt_x = nt_y = HomogenNumericTable(x)

    ridge_training_algorithm = ridge_training.Batch()
    # set input values
    ridge_training_algorithm.input.set(ridge_training.data, nt_x)
    ridge_training_algorithm.input.set(ridge_training.dependentVariables,
                                        nt_y)
    # check if intercept flag is set
    #ridge_training_algorithm.parameter.interceptFlag = True \
    #    if 'intercept' in self.parameters else True
    # set parameter
    alpha = 1.0
    alpha_nt = HomogenNumericTable(np.array([alpha], ndmin=2))
    ridge_training_algorithm.parameter.ridgeParameters = alpha_nt
    # calculate
    res = ridge_training_algorithm.compute()
    # return trained model
    model = res.get(ridge_training.model)
    beta_coeff = model.getBeta()
    np_beta_coeff = getNumpyArray(beta_coeff)

    res_beta_coeff = np.array([0.294118, 0.823529]).reshape(1,2)

    assert_array_almost_equal(res_beta_coeff, np_beta_coeff)

    # predict
    ridge_prediction_algorithm = ridge_prediction.Batch_Float64DefaultDense()
    ridge_prediction_algorithm.input.setModel(ridge_prediction.model, model)
    ridge_prediction_algorithm.input.setTable(ridge_prediction.data, nt_x)

    result = ridge_prediction_algorithm.compute()
    np_predict = getNumpyArray(result.get(ridge_prediction.prediction))
    assert_array_almost_equal(x, np_predict, decimal=0)

def get_random_array(rows=10, columns=9):

    x = np.random.rand(rows, columns)
    y = np.random.rand(rows, 1)

    return (x,y)

# remark: we do not need test for overfitting, Ridge Regression helps here

def get_daal_prediction(x=np.arange(10).reshape(10,1), y=np.arange(10).reshape(10,1)):

    ntX = HomogenNumericTable(x)
    ntY = HomogenNumericTable(y)

    ridge_training_algorithm = ridge_training.Batch()
    ridge_training_algorithm.input.set(ridge_training.data, ntX)
    ridge_training_algorithm.input.set(ridge_training.dependentVariables, ntY)

    # set parameter
    alpha = 0.0
    alpha_nt = HomogenNumericTable(np.array([alpha], ndmin=2))
    ridge_training_algorithm.parameter.ridgeParameters = alpha_nt

    result = ridge_training_algorithm.compute()
    model = result.get(ridge_training.model)

    ridge_prediction_algorithm = ridge_prediction.Batch()
    ridge_prediction_algorithm.input.setModel(ridge_prediction.model, model)
    ridge_prediction_algorithm.input.setTable(ridge_prediction.data, ntX)
    result = ridge_prediction_algorithm.compute()

    np_predicted = getNumpyArray(result.get(ridge_prediction.prediction))
    # assert the same as the initial dependent variable
    assert_array_almost_equal(y, np_predicted)
    return np_predicted

def get_scikit_prediction(x=np.arange(10).reshape(10,1), y=np.arange(10).reshape(10,1)):

    regression = ScikitRidgeRegression(alpha=0.0)
    regression.fit(x, y)

    return regression.predict(x)

def test_ridge_regression_against_scikit(rows=10, columns=9):
    '''
    Test prediction daal against scikit
    :param rows:
    :param columns:
    '''
    inout = get_random_array(rows, columns)
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
    x = inout[0]
    y = inout[1]

    ntX = HomogenNumericTable(x)
    ntY = HomogenNumericTable(y)

    ridge_training_algorithm = ridge_training.Batch()
    ridge_training_algorithm.input.set(ridge_training.data, ntX)
    ridge_training_algorithm.input.set(ridge_training.dependentVariables, ntY)

    # set parameter
    alpha = 1.0
    alpha_nt = HomogenNumericTable(np.array([alpha], ndmin=2))
    ridge_training_algorithm.parameter.ridgeParameters = alpha_nt

    result = ridge_training_algorithm.compute()
    model = result.get(ridge_training.model)
    beta_coeff = model.getBeta()
    np_beta = getNumpyArray(beta_coeff)

    assert y.transpose().shape == np_beta.shape, "Dependent variable size must have\
        the same size as Beta coefficient"

def test_intercept_flag(rows=10, columns=9):
    inout = get_random_array(rows, columns)
    x = inout[0]
    y = inout[1]

    ntX = HomogenNumericTable(x)
    ntY = HomogenNumericTable(y)

    ridge_training_algorithm = ridge_training.Batch()
    ridge_training_algorithm.input.set(ridge_training.data, ntX)
    ridge_training_algorithm.input.set(ridge_training.dependentVariables, ntY)

    # set parameter
    alpha = 1.0
    alpha_nt = HomogenNumericTable(np.array([alpha], ndmin=2))
    ridge_training_algorithm.parameter.ridgeParameters = alpha_nt

    result = ridge_training_algorithm.compute()

    model = result.get(ridge_training.model)
    beta_coeff = model.getBeta()
    np_beta = getNumpyArray(beta_coeff)
    daal_intercept = np_beta[0,0]

    regression = ScikitRidgeRegression(alpha=1.0, fit_intercept=True)
    regression.fit(x, y)

    scikit_intercept = regression.intercept_
    assert_array_almost_equal(scikit_intercept, [daal_intercept])

def test_ridge_regression_normalized(): test_fit_ridge_regression_daal_vs_sklearn()
def test_ridge_regression(): test_ridge_regression_simple()
def test_ridge_regression_param_3_2(): test_ridge_regression_against_scikit(rows=3, columns=2)
def test_ridge_regression_with_sc(): test_ridge_regression_against_scikit()
def test_beta(): 
    test_coeff_size(rows=10, columns=9)
    test_intercept_flag(rows=10, columns=9)
