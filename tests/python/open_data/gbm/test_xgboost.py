# -*- encoding: utf-8 -*-
"""
XGBoost solver tests using Kaggle datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import logging

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def fun():
    import xgboost as xgb
    import numpy as np
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    import time

    # Fetch dataset using sklearn
    cov = fetch_covtype()
    X = cov.data
    y = cov.target

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                        random_state=42)

    # Specify sufficient boosting iterations to reach a minimum
    num_round = 10

    # Leave most parameters as default
    param = {'objective': 'multi:softmax', # Specify multiclass classification
             'num_class': 8, # Number of possible output classes
             'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
             }

    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)

    gpu_res = {} # Store accuracy result
    tmp = time.time()
    # Train model
    xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

    # Repeat for CPU algorithm
    tmp = time.time()
    param['tree_method'] = 'hist'
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
    print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))



def test_xgboost_covtype(): fun()


if __name__ == '__main__':
    test_xgboost_covtype()
