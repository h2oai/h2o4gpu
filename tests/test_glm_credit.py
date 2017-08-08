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


def fun(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2):
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    print("Reading Data")
    df = feather.read_dataframe("./data/credit.feather")
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    t1 = time.time()
    logloss_train, logloss_test = elastic_net(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                                        validFraction=validFraction, verbose=0,family="logistic")

    # check logloss
    print(logloss_train[0, 0])
    print(logloss_train[0, 1])
    print(logloss_train[0, 2])
    print(logloss_test[0, 2])
    sys.stdout.flush()
    
    if validFraction==0.0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 1] < .48
        assert logloss_train[0, 2] < .48
        assert logloss_test[0, 2] < .48
            
    print('/n Total execution time:%d' % (time.time() - t1))

    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))
    #    endfunnel(pipes)
    print("DONE.")
    sys.stdout.flush()


def test_glm_credit_gpu_fold5_quick_train(): fun(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.0)
def test_glm_credit_gpu_fold5_quick_valid(): fun(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.2)


if __name__ == '__main__':
    test_glm_credit_gpu_fold5_quick_train()
    test_glm_credit_gpu_fold5_quick_valid()
