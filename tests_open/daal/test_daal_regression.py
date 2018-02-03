# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

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
        normalize=True,
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
        kmeans_sk = skKMeans(n_init=1, n_clusters=centers, algorithm='full', n_jobs=-1)
        start_sk = time.time()
        kmeans_sk.fit(X)
        end_sk = time.time()
        #assert end_daal - start_daal <= end_sklearn - start_sklearn

    #DLR.print_table(trained.getBeta(), "Linear Regression coefficients:")
    #prediction = solver.predict(testData)
    #DLR.print_table(prediction, "Linear Regression prediction results: (first 10 rows):", 10)

def test_linear_regression_normalized(): test_fit_linear_regression()

if __name__ == '__main__':
    test_linear_regression_normalized()
