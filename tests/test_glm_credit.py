import time
import sys
import os
import numpy as np
import logging
import feather

print(sys.path)

try:
    from utils import find_file, runglm, elastic_net
except:
    from tests.utils import find_file, runglm, elastic_net

logging.basicConfig(level=logging.DEBUG)


def func(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, verbose=0,family="elasticnet", print_all_errors=False, tolerance=.001):
    name = str(sys._getframe().f_code.co_name)
    name = str(sys._getframe(1).f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    print("Reading Data")
    df = feather.read_dataframe("./data/credit.feather")
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    t1 = time.time()

    elastic_net(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                validFraction=validFraction, verbose=verbose,family=family,print_all_errors=print_all_errors,tolerance=tolerance, name=name)

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