import numpy as np
from numpy.random import RandomState
import daal_solver
import sys

seeded = RandomState(42)

from h2o4gpu.solvers.linear_regression import LinearRegression
from h2o4gpu.solvers import DLR 
   
def test_fit_linear_regression():
    
    trainData = seeded.rand(200,10)
    trainDependentVariables = seeded.rand(200,2)
    
    testData = seeded.rand(50,10)
    testDependentVariables = seeded.rand(50,2)
    
    solver = LinearRegression(
            fit_intercept=True, 
            normalize=False,
            verbose=True,
            backend='daal')
    trained = solver.fit(trainData, trainDependentVariables)
    DLR.print_table(trained.getBeta(), "Linear Regression coefficients:")
    prediction = solver.predict(testData)
    DLR.print_table(prediction, "Linear Regression prediction results: (first 10 rows):", 10)

test_fit_linear_regression()