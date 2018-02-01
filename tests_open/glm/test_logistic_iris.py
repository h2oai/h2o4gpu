# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Iris dataset.
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import numpy as np
import logging
import pandas as pd
from h2o4gpu import datasets
import h2o4gpu

print(sys.path)

logging.basicConfig(level=logging.DEBUG)


def func():
   
    # data prep
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # removing the third class, making it a binary problem
    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)

    # splitting into train and valid frame
    X_test = X[np.r_[40:50,90:100]]
    y_test = y[np.r_[40:50,90:100]]
    X = X[np.r_[:40,50:90]]
    y = y[np.r_[:40,50:90]]

    classification = True

    logreg = h2o4gpu.LogisticRegression(penalty="l1")
    lr = h2o4gpu.ElasticNetH2O(
        n_threads = None,
        n_alphas = 1,
        n_lambdas = 1,
        n_folds = 1,
        lambda_max = 1.0,
        lambda_min_ratio = 1.0,
        lambda_stop_early = False,
        store_full_path = 0,
        alphas = None,
        lambdas = None,
        family = 'logistic',
        alpha_max = 1.0,
        alpha_min = 1.0)
    
    model = logreg.fit(X, y)
    mm = lr.fit(X, y)

    y_pred = model.predict(X_test)
    print(y_pred)
    y_p = mm.predict(X_test)
    print(y_p)
    print(y_pred, np.round(y_pred))

    # TO-DO: change the assertion once the logic to convert probabilities to classes is implemented
    assert (y_test == np.round(y_pred)).all() == True
    assert (y_pred == y_p).all() == True

def test_logistic_iris(): func()



if __name__ == '__main__':
    test_logistic_iris()
