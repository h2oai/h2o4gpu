import time
import sys
import os
import numpy as np
import logging
import pandas as pd

print(sys.path)

from h2o4gpu.util.testing_utils import find_file, run_glm
import h2o4gpu
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.DEBUG)


def func(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, verbose=0, family="elasticnet",
         print_all_errors=False, tolerance=.1):
    name = str(sys._getframe().f_code.co_name)
    name = str(sys._getframe(1).f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    print("Reading Data")
    df = pd.read_csv("../open_data/creditcard.csv")
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')

    t1 = time.time()

    logloss_train, logloss_test = run_glm(X, y, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                                          validFraction=validFraction, verbose=verbose,print_all_errors=print_all_errors,
                                          tolerance=tolerance, name=name, solver="logistic")

    # check logloss
    print(logloss_train[0, 0])
    print(logloss_train[0, 1])
    print(logloss_train[0, 2])
    print(logloss_test[0, 2])
    sys.stdout.flush()

    # Train only
    if validFraction == 0.0 and nFolds == 0:
        assert logloss_train[0, 0] < .47
        assert logloss_test[0, 0] < .47

    # Train + nfolds
    if validFraction == 0.0 and nFolds > 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 1] < .44

    # Train + validation
    if validFraction > 0.0 and nFolds == 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 2] < .44
        assert logloss_test[0, 0] < .48
        assert logloss_test[0, 2] < .44

    # Train + validation + nfolds
    if validFraction > 0.0 and nFolds > 0:
        assert logloss_train[0, 0] < .48
        assert logloss_train[0, 1] < .48
        assert logloss_train[0, 2] < .44
        assert logloss_test[0, 0] < .48
        assert logloss_test[0, 1] < .48
        assert logloss_test[0, 2] < .44

    sys.stdout.flush()

    print('/n Total execution time:%d' % (time.time() - t1))

    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))

    print("DONE.")
    sys.stdout.flush()

#Function to check fall back to sklearn
def test_fit_credit_backupsklearn():
    df = pd.read_csv("./open_data/creditcard.csv")
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    Solver = h2o4gpu.LogisticRegression
    enet = Solver(dual=True, max_iter=100, tol=1E-4)
    print("h2o4gpu scikit wrapper fit()")
    print(enet.fit(X, y))
    print("h2o4gpu scikit wrapper predict()")
    print(enet.predict(X))
    print("h2o4gpu scikit wrapper predict_proba()")
    print(enet.predict_proba(X))
    print("h2o4gpu scikit wrapper predict_log_proba()")
    print(enet.predict_log_proba(X))
    print("h2o4gpu scikit wrapper score()")
    print(enet.score(X,y))
    print("h2o4gpu scikit wrapper decision_function()")
    print(enet.decision_function(X))
    print("h2o4gpu scikit wrapper densify()")
    print(enet.densify())
    print("h2o4gpu scikit wrapper sparsify")
    print(enet.sparsify())
    
    from sklearn.linear_model.logistic import  LogisticRegression
    enet_sk = LogisticRegression(dual=True, max_iter=100, tol=1E-4)
    print("Scikit fit()")
    print(enet_sk.fit(X, y))
    print("Scikit predict()")
    print(enet_sk.predict(X))
    print("Scikit predict_proba()")
    print(enet_sk.predict_proba(X))
    print("Scikit predict_log_proba()")
    print(enet_sk.predict_log_proba(X))
    print("Scikit score()")
    print(enet_sk.score(X,y))
    print("Scikit decision_function()")
    print(enet_sk.decision_function(X))
    print("Scikit densify()")
    print(enet_sk.densify())
    print("Sciki sparsify")
    print(enet_sk.sparsify())

    enet_sk_coef = csr_matrix(enet_sk.coef_, dtype=np.float32).toarray()
    print(enet_sk.coef_)
    print(enet_sk_coef)
    print(enet.coef_)
    print("Coeffs should match")
    assert np.allclose(enet.coef_, enet_sk_coef, rtol = 1e-2, atol=1e-2)
    


# def test_glmlogistic_credit_gpu_quick_train(): func(nGPUs=1, nFolds=0, nLambdas=5, nAlphas=3, validFraction=0.0, verbose=0,
#                                             family="logistic", print_all_errors=False, tolerance=.1)
#
#
# def test_glmlogistic_credit_gpu_quick_train_5fold(): func(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.0,
#                                                   verbose=0, family="logistic", print_all_errors=False, tolerance=.1)
#
#
# def test_glmlogistic_credit_gpu_quick_train_valid_nofold(): func(nGPUs=1, nFolds=0, nLambdas=5, nAlphas=3, validFraction=0.2,
#                                                          verbose=0, family="logistic", print_all_errors=False,
#                                                          tolerance=.1)
#
#
# def test_glmlogistic_credit_gpu_quick_train_valid_5fold(): func(nGPUs=1, nFolds=5, nLambdas=5, nAlphas=3, validFraction=0.2,
#                                                         verbose=0, family="logistic", print_all_errors=False,
#                                                         tolerance=.1)
#def test_sklearn_logit(): test_fit_credit_backupsklearn()


if __name__ == '__main__':
    # test_glmlogistic_credit_gpu_quick_train()
    # test_glmlogistic_credit_gpu_quick_train_5fold()
    # test_glmlogistic_credit_gpu_quick_train_valid_nofold()
    # test_glmlogistic_credit_gpu_quick_train_valid_5fold()
    test_fit_credit_backupsklearn()