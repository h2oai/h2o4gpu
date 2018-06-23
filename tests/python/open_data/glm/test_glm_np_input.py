# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
import logging
import numpy as np
from h2o4gpu.solvers.elastic_net import ElasticNetH2O

def isclose(a,b,reltol):
    norm = np.fabs(a) + np.fabs(b) + 1E-30
    diff = (a-b)
    reldiff = np.fabs(diff)/norm
    if reldiff<=reltol:
        return True
    else:
        return False


#
print(sys.path)
#
logging.basicConfig(level=logging.DEBUG)

#Set up model
tol = 1E-3
n_threads = None
n_gpus = -1
fit_intercept = True
n_folds = 1
tol_seek_factor = 1E-1
glm_stop_early = False
lambda_stop_early=False
glm_stop_early_error_fraction = 1.0
max_iter = 5000
verbose = 0
store_full_path = 0
lambda_max = 0.0
alpha_max = 0.0
alpha_min = 0.0
n_alphas=1
n_lambdas=1
lambda_min_ratio=0.0
family = "elasticnet"
order=None
lm = ElasticNetH2O(
    n_threads=n_threads,
    n_gpus=n_gpus,
    fit_intercept=fit_intercept,
    lambda_min_ratio=lambda_min_ratio,
    n_lambdas=n_lambdas,
    n_folds=n_folds,
    n_alphas=n_alphas,
    tol=tol,
    tol_seek_factor=tol_seek_factor,
    lambda_stop_early=lambda_stop_early,
    glm_stop_early=glm_stop_early,
    glm_stop_early_error_fraction=glm_stop_early_error_fraction,
    max_iter=max_iter,
    verbose=verbose,
    family=family,
    lambda_max=lambda_max,
    alpha_max=alpha_max,
    alpha_min=alpha_min,
    store_full_path=store_full_path,
    order=order
)
def func(model):
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    #
    a = 2.0
    b = 10.0
    y = a * X + b
    #
    lm = model.fit(X, y)
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
    assert isclose(lm.X[0][0], a, tol)
    assert isclose(lm.X[0][1], b, tol)

    assert isclose(result1, 40.0, tol)
    assert isclose(result2, 40.0, tol)
    assert isclose(result3, 40.0, tol)

def func2(model):
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #
    a = 2
    b = 10
    y = a * X + b
    #
    lm = model
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
    assert isclose(lm.X[0][0] , a, tol)
    assert isclose(lm.X[0][1] , b, tol)

    assert isclose(result1, 40, tol)
    assert isclose(result2, 40, tol)
    assert isclose(result3, 40, tol)


# test for higher precision linear fit (exact values)
# No early stopping and low tolerance
# TODO: Succeeds as individual test, but during "make test"
# it fails as not being exactly same.
# Since generally don't expect to be machine accurate,
# relax this test.  May indicate non-reproducibility.
def func3(model):
    X = np.array([1,2,3,4,5,6,7,8,9,10])    
    X = X.astype(np.float32)
    
    a = 2
    b = 10
    y = a * X + b

    model.tol=1E-7
    model.tol_seek_factor=1E-2
    lm = model
    lm.fit(X, y)

    print("predict1=%21.15g" % lm.predict(np.array([15.0])))
    print("predict2=%21.15g" % lm.predict(np.array([16.0])))
    assert isclose(lm.predict(np.array([15.0])), 40.0, tol)
    assert isclose(lm.predict(np.array([16.0])), 42.0, tol)

def test_glm_np_input(): func(model=lm)

def test_glm_np_input_integer(): func2(model=lm)

def test_glm_np_exact(): func3(model=lm)

if __name__ == '__main__':
    test_glm_np_input()
    test_glm_np_input_integer()
    test_glm_np_exact()
