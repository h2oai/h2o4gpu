# -*- encoding: utf-8 -*-
"""
ElasticNet_h2o4gpu solver tests using Iris dataset.
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

import time
import sys
import random
import numpy as np
import logging
import h2o4gpu

print(sys.path)

logging.basicConfig(level=logging.DEBUG)


def func():
    tol = 1e-3

    # generating the data
    X = np.random.randn(1000, 20)
    coefs = np.random.randn(20)
    const_coef = np.random.randn(1)

    # index of sparse coefficients
    zero_coef_loc = random.sample(range(20), 4)
    coefs[zero_coef_loc] = 0

    y = np.dot(X, coefs) + const_coef

    start = time.time()
    lasso = h2o4gpu.Lasso(tol=tol)
    lasso_model = lasso.fit(X, y)
    print('time to train:', time.time() - start)
    print('original coeffs', coefs)
    print('Lasso coefficients:', lasso_model.X)
    print(const_coef)

    zero_coef_index = np.where(lasso_model.X[0] == 0)
    print(zero_coef_index)
    print(zero_coef_loc)

    check_true = (zero_coef_index == np.sort(zero_coef_loc)).all()
    assert check_true == True
    assert np.fabs(lasso_model.X[0][-1] - const_coef) < 2 * tol

import pytest
@pytest.mark.skip("Not a rigorous test as can have many more coefficients set as zero as one designed.")
def test_lasso_sparsity(): func()


if __name__ == '__main__':
    test_lasso_sparsity()
