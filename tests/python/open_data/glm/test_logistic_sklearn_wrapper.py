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

#Function to check fall back to sklearn
def test_fit_credit_backupsklearn():
    df = pd.read_csv("./open_data/creditcard.csv")
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    Solver = h2o4gpu.LogisticRegression

    enet_h2o4gpu = Solver(glm_stop_early=False)
    print("h2o4gpu fit()")
    enet_h2o4gpu.fit(X, y)
    print("h2o4gpu predict()")
    print(enet_h2o4gpu.predict(X))
    print("h2o4gpu score()")
    print(enet_h2o4gpu.score(X,y))

    enet = Solver(dual=True, max_iter=100, tol=1E-4, intercept_scaling=0.99, random_state=1234)
    print("h2o4gpu scikit wrapper fit()")
    enet.fit(X, y)
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
    
    from h2o4gpu.linear_model.logistic import  LogisticRegressionSklearn
    enet_sk = LogisticRegressionSklearn(dual=True, max_iter=100, tol=1E-4, intercept_scaling=0.99,
                                        random_state=1234, solver='liblinear')
    print("Scikit fit()")
    enet_sk.fit(X, y)
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
    print(enet_sk.intercept_)
    print("Coeffs, intercept, and n_iters should match")
    assert np.allclose(enet.coef_, enet_sk_coef)
    assert np.allclose(enet.intercept_, enet_sk.intercept_)
    assert np.allclose(enet.n_iter_, enet_sk.n_iter_)
    print("Preds should match")
    assert np.allclose(enet.predict_proba(X), enet_sk.predict_proba(X))
    assert np.allclose(enet.predict(X), enet_sk.predict(X))
    assert np.allclose(enet.predict_log_proba(X), enet_sk.predict_log_proba(X))

#def test_sklearn_logit(): test_fit_credit_backupsklearn()


if __name__ == '__main__':
    test_fit_credit_backupsklearn()
