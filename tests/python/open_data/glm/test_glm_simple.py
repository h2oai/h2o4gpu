# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import numpy as np
import pandas as pd
import logging

print(sys.path)

from h2o4gpu.util.testing_utils import find_file, run_glm

logging.basicConfig(level=logging.DEBUG)


def fun(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, choosealphalambda=None):
    name = str(sys._getframe().f_code.co_name)
    name = str(sys._getframe(1).f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    #    pipes = startfunnel(os.path.join(os.getcwd(), "tmp/"), name)

    print("Reading Data")
    # from numpy.random import randn
    #  m=1000
    #  n=100
    #  A=randn(m,n)
    #  x_true=(randn(n)/n)*float64(randn(n)<0.8)
    #  b=A.dot(x_true)+0.5*randn(m)

    df = pd.read_csv("./open_data/simple.txt", sep=" ", header=None)
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    if choosealphalambda == 1 or choosealphalambda == 3:
        alphas = [1E-1, 0.3, 0.5, 1.0]
    else:
        alphas = None
    if choosealphalambda == 2 or choosealphalambda == 3:
        lambdas = [2, 1E-2, 1E-3, 1E-5]
    else:
        lambdas = None

    t1 = time.time()
    rmse_train, rmse_test = run_glm(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                                    validFraction=validFraction, verbose=0, name=name, alphas=alphas, lambdas=lambdas, tolerance=0.3)

    # check rmse
    print(rmse_train[0, 0])
    print(rmse_train[0, 1])
    print(rmse_train[0, 2])
    print(rmse_test[0, 2])
    sys.stdout.flush()

    if validFraction == 0.0:
        if nLambdas > 50:
            if nFolds == 1:
                assert rmse_train[0, 0] < 0.04
                assert rmse_train[0, 1] < 0.04
                assert rmse_train[0, 2] < 0.04
                assert rmse_test[0, 2] < 0.04

                assert rmse_train[-1, 0] < 0.05
                assert rmse_train[-1, 1] < 0.05
                assert rmse_train[-1, 2] < 0.05
                assert rmse_test[-1, 2] < 0.05
            else:
                assert rmse_train[0, 0] < 0.37
                assert rmse_train[0, 1] < 0.22
                assert rmse_train[0, 2] < 0.4
                assert rmse_test[0, 2] < 0.4

                assert rmse_train[-1, 0] < 0.37
                assert rmse_train[-1, 1] < 0.77
                assert rmse_train[-1, 2] < 0.4
                assert rmse_test[-1, 2] < 0.4
        else:
            if nFolds == 1:
                assert rmse_train[0, 0] < 0.11
                assert rmse_train[0, 1] < 0.12
                assert rmse_train[0, 2] < 0.1
                assert rmse_test[0, 2] < 0.1

                assert rmse_train[-1, 0] < 0.11
                assert rmse_train[-1, 1] < 0.12
                assert rmse_train[-1, 2] < 0.1
                assert rmse_test[-1, 2] < 0.1
            else:
                assert rmse_train[0, 0] < 0.37
                assert rmse_train[0, 1] < 0.22
                assert rmse_train[0, 2] < 0.4
                assert rmse_test[0, 2] < 0.4

                assert rmse_train[-1, 0] < 0.37
                assert rmse_train[-1, 1] < 0.24
                assert rmse_train[-1, 2] < 0.4
                assert rmse_test[-1, 2] < 0.4
    else:
        if nLambdas > 50:
            if nFolds == 1:
                assert rmse_train[0, 0] < 0.4
                assert rmse_train[0, 1] < 0.4
                assert rmse_train[0, 2] < 0.51
                assert rmse_test[0, 2] < 0.51

                assert rmse_train[-1, 0] < 0.51
                assert rmse_train[-1, 1] < 0.51
                assert rmse_train[-1, 2] < 0.51
                assert rmse_test[-1, 2] < 0.51
            else:
                assert rmse_train[0, 0] < 0.51
                assert rmse_train[0, 1] < 0.51
                assert rmse_train[0, 2] < 2
                assert rmse_test[0, 2] < 2

                assert rmse_train[-1, 0] < 0.54
                assert rmse_train[-1, 1] < 0.54
                assert rmse_train[-1, 2] < 2.2
                assert rmse_test[-1, 2] < 2.2
        else:
            if nFolds == 1:
                assert rmse_train[0, 0] < 0.4
                assert rmse_train[0, 1] < 0.4
                assert rmse_train[0, 2] < 2
                assert rmse_test[0, 2] < 2

                assert rmse_train[-1, 0] < 0.51
                assert rmse_train[-1, 1] < 0.51
                assert rmse_train[-1, 2] < 2
                assert rmse_test[-1, 2] < 2
            else:
                assert rmse_train[0, 0] < 0.45
                assert rmse_train[0, 1] < 0.3
                assert rmse_train[0, 2] < 2
                assert rmse_test[0, 2] < 2

                assert rmse_train[-1, 0] < 0.45
                assert rmse_train[-1, 1] < 0.3
                assert rmse_train[-1, 2] < 2
                assert rmse_test[-1, 2] < 2

    print('/n Total execution time:%d' % (time.time() - t1))

    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))
    #    endfunnel(pipes)
    print("DONE.")
    sys.stdout.flush()

# for now don't test folds with simple because h2o-3 can't handle it
# for small data sets
altfold = 1

def test_glm_simple_gpu_fold1_quick_0(): fun(1, 1, 5, 3, validFraction=0)


def test_glm_simple_gpu_fold1_0(): fun(1, 1, 100, 8, validFraction=0)


def test_glm_simple_gpu_fold2_0(): fun(1, altfold, 100, 3, validFraction=0)


def test_glm_simple_gpu_fold1_quick(): fun(1, 1, 5, 3, validFraction=0.2)


def test_glm_simple_gpu_fold1(): fun(1, 1, 100, 8, validFraction=0.2)


def test_glm_simple_gpu_fold2(): fun(1, altfold, 100, 3, validFraction=0.2)


def test_glm_simple_gpu2_fold1_quick(): fun(2, 1, 5, 3, validFraction=0.2)


def test_glm_simple_gpu2_fold1(): fun(2, 1, 100, 8, validFraction=0.2)


def test_glm_simple_gpu2_fold2(): fun(3, altfold, 100, 3, validFraction=0.2)


def test_glm_simple_cpu_fold1_quick(): fun(0, 1, 5, 3, validFraction=0.2)


def test_glm_simple_cpu_fold1(): fun(0, 1, 100, 8, validFraction=0.2)


def test_glm_simple_cpu_fold2(): fun(0, altfold, 100, 3, validFraction=0.2)


def test_glm_simple_gpu_choosealphalambda1(): fun(1, 1, 5, 3, validFraction=0, choosealphalambda=1)
def test_glm_simple_gpu_choosealphalambda2(): fun(1, 1, 5, 3, validFraction=0, choosealphalambda=2)
def test_glm_simple_gpu_choosealphalambda3(): fun(1, 1, 5, 3, validFraction=0, choosealphalambda=3)


if __name__ == '__main__':
    test_glm_simple_gpu_fold1_quick_0()
    test_glm_simple_gpu_fold1_0()
    test_glm_simple_gpu_fold2_0()

    test_glm_simple_gpu_fold1_quick()
    test_glm_simple_gpu_fold1()
    test_glm_simple_gpu_fold2()

    test_glm_simple_gpu2_fold1_quick()
    test_glm_simple_gpu2_fold1()
    test_glm_simple_gpu2_fold2()

    test_glm_simple_cpu_fold1_quick()
    test_glm_simple_cpu_fold1()
    test_glm_simple_cpu_fold2()

    test_glm_simple_gpu_choosealphalambda1()
    test_glm_simple_gpu_choosealphalambda2()
    test_glm_simple_gpu_choosealphalambda3()
