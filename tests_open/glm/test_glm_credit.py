# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import numpy as np
import logging
import pandas as pd
print(sys.path)

from h2o4gpu.util.testing_utils import find_file, run_glm

logging.basicConfig(level=logging.DEBUG)


def func(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, verbose=0,family="elasticnet", print_all_errors=False, tolerance=.001):
    name = str(sys._getframe().f_code.co_name)
    name = str(sys._getframe(1).f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    print("Reading Data")
    df = pd.read_csv("./open_data/creditcard.csv")
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    t1 = time.time()

    logloss_train, logloss_test = run_glm(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                validFraction=validFraction, verbose=verbose,family=family,print_all_errors=print_all_errors,tolerance=tolerance, name=name)


    # check logloss
    print(logloss_train[0, 0])
    print(logloss_train[0, 1])
    print(logloss_train[0, 2])
    print(logloss_test[0, 2])
    sys.stdout.flush()

    #Train only
    if validFraction==0.0 and nFolds == 0:
        assert logloss_train[0, 0] < .47
        assert logloss_train[1,0] < .47
        assert logloss_train[2,0] < .47
        assert logloss_test[0, 0] < .47
        assert logloss_test[1, 0] < .47
        assert logloss_test[2, 0] < .47

    #Train + nfolds
    if validFraction==0.0 and nFolds > 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 1] < .44
        assert logloss_train[1, 0] < .48
        assert logloss_train[1, 1] < .48
        assert logloss_train[2, 0] < .48
        assert logloss_train[2, 1] < .44
        assert logloss_test[0, 0] < .48
        assert logloss_test[0, 1] < .44
        assert logloss_test[1, 0] < .48
        assert logloss_test[1, 1] < .48
        assert logloss_test[2, 0] < .48
        assert logloss_test[2, 1] < .44

    #Train + validation
    if validFraction > 0.0 and nFolds == 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 2] < .44
        assert logloss_train[1, 0] < .48
        assert logloss_train[1, 2] < .44
        assert logloss_train[2, 0] < .48
        assert logloss_train[2, 2] < .44
        assert logloss_test[0, 0] < .48
        assert logloss_test[0, 2] < .44
        assert logloss_test[1, 0] < .48
        assert logloss_test[1, 2] < .44
        assert logloss_test[2, 0] < .48
        assert logloss_test[2, 2] < .44

    #Train + validation + nfolds
    if validFraction > 0.0 and nFolds > 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 1] < .48
        assert logloss_train[0, 2] < .44
        assert logloss_train[1, 0] < .48
        assert logloss_train[1, 1] < .48
        assert logloss_train[1, 2] < .44
        assert logloss_train[2, 0] < .48
        assert logloss_train[2, 1] < .48
        assert logloss_train[2, 2] < .44
        assert logloss_test[0, 0] < .48
        assert logloss_test[0, 1] < .48
        assert logloss_test[0, 2] < .44
        assert logloss_test[1, 0] < .48
        assert logloss_test[1, 1] < .48
        assert logloss_test[1, 2] < .44
        assert logloss_test[2, 0] < .48
        assert logloss_test[2, 1] < .48
        assert logloss_test[2, 2] < .44
    

    sys.stdout.flush()

    print('/n Total execution time:%d' % (time.time() - t1))

    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))

    print("DONE.")
    sys.stdout.flush()

def test_glm_credit_gpu_quick_train(): func(nGPUs=1, nFolds=0, nLambdas=5, nAlphas=3, validFraction=0.0,verbose=0,family="logistic",print_all_errors=False,tolerance=.009)
def test_glm_credit_gpu_quick_train_5fold(): func(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.0,verbose=0,family="logistic",print_all_errors=False,tolerance=.009)
def test_glm_credit_gpu_quick_train_valid_nofold(): func(nGPUs=1, nFolds=0, nLambdas=5, nAlphas=3, validFraction=0.2,verbose=0,family="logistic",print_all_errors=False,tolerance=.009)
def test_glm_credit_gpu_quick_train_valid_5fold(): func(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.2,verbose=0,family="logistic",print_all_errors=False,tolerance=.009)


if __name__ == '__main__':
    test_glm_credit_gpu_quick_train()
    test_glm_credit_gpu_quick_train_5fold()
    test_glm_credit_gpu_quick_train_valid_nofold()
    test_glm_credit_gpu_quick_train_valid_5fold()