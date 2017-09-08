# -*- encoding: utf-8 -*-
"""
GLM solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
import logging
import numpy as np
from h2o4gpu.solvers.linear_regression import LinearRegression

#
print(sys.path)
#
logging.basicConfig(level=logging.DEBUG)


#
def func():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    #
    a = 2.0
    b = 10.0
    y = a * X + b
    #
    tol = 1E-3
    lm = LinearRegression(tol = tol, glm_stop_early=False)
    lm.fit(X, y)
    #
    print('Linear Regression')
    test0 = np.array([15.0])
    result1 = lm.predict(test0)
    result2 = lm.predict(np.array([15.0]))
    test = np.array([15.0]).astype(np.float32)  # pass in data that is already float32
    result3 = lm.predict(test)

    print('Predicted:', result1)
    print('Predicted:', result2)
    print('Predicted:', result3)
    print('Coefficients:', lm.X)
    #
    # Assert coefficients are within a reasonable range for various prediction values
    assert np.fabs(lm.X[0][0] - a) < 1.1*tol
    assert np.fabs(lm.X[0][1] - b) < 1.1*tol

    assert np.fabs(result1 - 40.0) < 1.1*tol
    assert np.fabs(result2 - 40.0) < 1.1*tol
    assert np.fabs(result3 - 40.0) < 1.1*tol


#

def func2():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #
    a = 2
    b = 10
    y = a * X + b
    #
    tol = 1e-3
    lm = LinearRegression(tol=tol, glm_stop_early=False)
    lm.fit(X, y)
    #
    print('Linear Regression')
    test0 = np.array([15])
    result1 = lm.predict(test0)
    result2 = lm.predict(np.array([15]))
    test = np.array([15]).astype(np.int32)  # pass in data that is already int32
    result3 = lm.predict(test)

    print('Predicted:', result1)
    print('Predicted:', result2)
    print('Predicted:', result3)
    print('Coefficients:', lm.X)
    #
    # Assert coefficients are within a reasonable range for various prediction values
    # 2.0*tol because reverts to float32
    assert np.fabs(lm.X[0][0] - a) < 2.0*tol
    assert np.fabs(lm.X[0][1] - b) < 2.0*tol

    assert np.fabs(result1 - 40) < 2.0*tol
    assert np.fabs(result2 - 40) < 2.0*tol
    assert np.fabs(result3 - 40) < 2.0*tol


#
#
def test_glm_np_input(): func()


def test_glm_np_input_integer(): func2()


#
#
if __name__ == '__main__':
    test_glm_np_input()
    test_glm_np_input_integer()
