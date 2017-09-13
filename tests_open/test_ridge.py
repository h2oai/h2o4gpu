import time
import sys
import os
import numpy as np
import pandas as pd
import logging

print(sys.path)

from h2o4gpu.util.testing_utils import find_file, run_glm

logging.basicConfig(level=logging.DEBUG)

def fun(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2):
    name = str(sys._getframe().f_code.co_name)
    name = sys._getframe(1).f_code.co_name
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()


    print("Reading Data")
    #
    df = pd.read_csv("./open_data/simple.txt", delim_whitespace=True)
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    t1 = time.time()
    rmse_train, rmse_test = run_glm(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                                        validFraction=validFraction, verbose=0, name=name, solver="ridge")

    # check rmse
    print(rmse_train[0, 0])
    print(rmse_train[0, 1])
    print(rmse_train[0, 2])
    print(rmse_test[0, 2])
    sys.stdout.flush()

    if validFraction==0.0:
        assert rmse_train[0, 0] < 54000
        assert rmse_train[0, 1] < 54000
        assert rmse_train[0, 2] < 54000
        assert rmse_test[0, 2] < 54000
    else:
        if nLambdas>20:
            assert rmse_train[0, 0] < 50000
            assert rmse_train[0, 1] < 50000
            assert rmse_train[0, 2] < 50000
            assert rmse_test[0, 2] < 50000
        else:
            assert rmse_train[0, 0] < 59000
            assert rmse_train[0, 1] < 59000
            assert rmse_train[0, 2] < 59000
            assert rmse_test[0, 2] < 59000

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

def test_glmridge_ipums_gpu_fold1_quick_0(): fun(nGPUs=1, nFolds=1, nLambdas=3, nAlphas=3, validFraction=0)


def test_glmridge_ipums_gpu_fold1_0(): fun(nGPUs=1, nFolds=1, nLambdas=20, nAlphas=3, validFraction=0)


def test_glmridge_ipums_gpu_fold5_0(): fun(nGPUs=1, nFolds=altfold, nLambdas=20, nAlphas=3, validFraction=0)


def test_glmridge_ipums_gpu_fold1_quick(): fun(nGPUs=1, nFolds=1, nLambdas=5, nAlphas=3, validFraction=0.2)


def test_glmridge_ipums_gpu_fold1(): fun(nGPUs=1, nFolds=1, nLambdas=20, nAlphas=3, validFraction=0.2)


def test_glmridge_ipums_gpu_fold5(): fun(nGPUs=1, nFolds=altfold, nLambdas=20, nAlphas=3, validFraction=0.2)


def test_glmridge_ipums_gpu2_fold1_quick(): fun(nGPUs=2, nFolds=1, nLambdas=3, nAlphas=3, validFraction=0.2)


def test_glmridge_ipums_gpu2_fold1(): fun(nGPUs=2, nFolds=1, nLambdas=20, nAlphas=3, validFraction=0.2)


if __name__ == '__main__':
    test_glmridge_ipums_gpu_fold1_quick_0()
    test_glmridge_ipums_gpu_fold1_0()
    test_glmridge_ipums_gpu_fold5_0()

    test_glmridge_ipums_gpu_fold1_quick()
    test_glmridge_ipums_gpu_fold1()
    test_glmridge_ipums_gpu_fold5()

    test_glmridge_ipums_gpu2_fold1_quick()
    test_glmridge_ipums_gpu2_fold1()