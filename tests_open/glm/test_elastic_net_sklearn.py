# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import h2o4gpu as h2o4gpu
import math
import numpy as np

'''
Elastic Net

   minimize (1/2) ||Ax - b||_2^2 + \alpha * \lambda ||x||_1 + 0.5 * (1-\alpha) * \lambda ||x||_2 for family == 'elasticnet'
   
   minimize \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1 for family == 'logistic'

   for 100 values of \lambda, and alpha in [0,1]

   See <h2o4gpu>/matlab/examples/lasso_path.m for detailed description.
'''


def elastic_net(X, y, nGPUs=0, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2, family="elasticnet", verbose=0):
    # choose solver
    Solver = h2o4gpu.ElasticNetH2O

    fit_intercept = True
    lambda_min_ratio = 1e-9

    # Setup Train/validation Set Split
    morig = X.shape[0]
    norig = X.shape[1]
    print("Original m=%d n=%d" % (morig, norig))
    fortran = X.flags.f_contiguous
    print("fortran=%d" % fortran)

    # Do train/valid split
    HO = int(validFraction * morig)
    H = morig - HO
    print("Size of Train rows=%d valid rows=%d" % (H, HO))
    trainX = np.copy(X[0:H, :])
    trainY = np.copy(y[0:H])
    validX = np.copy(X[H:morig, :])
    validY = np.copy(y[H:morig])
    mTrain = trainX.shape[0]
    mvalid = validX.shape[0]
    print("mTrain=%d mvalid=%d" % (mTrain, mvalid))

    ## Constructor
    print("Setting up solver")
    enet = Solver(fit_intercept=fit_intercept, lambda_min_ratio=lambda_min_ratio, n_gpus=nGPUs, n_lambdas=nlambda,
                  n_folds=nfolds, n_alphas=nalpha, verbose=verbose, family=family)

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    ## Solve
    print("Solving")
    # enet.fit(trainX, trainY)
    enet.fit(trainX, trainY, validX, validY)
    # enet.fit(trainX, trainY, validX, validY, trainW)
    # enet.fit(trainX, trainY, validX, validY, trainW, 0)
    #  enet.fit(trainX, trainY, validX, validY, trainW, givefullpath)
    print("Done Solving")

    # show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(enet.X)
    print("np.shape(Xvsalpha)")
    print(np.shape(enet.X))

    error = enet.error
    if family == 'logistic':
        print("logloss")
    else:
        print("rmse")
    print(error)

    print("lambdas")
    lambdas = enet.lambdas
    print(lambdas)

    print("alphas")
    alphas = enet.alphas
    print(alphas)

    print("tols")
    tols = enet.tols
    print(tols)

    print(enet.X)
    print(len(enet.X))

    print("intercept")
    print(enet.intercept_)

    ############## consistency check
    if fit_intercept:
        if trainX is not None:
            trainX_intercept = np.hstack([trainX, np.ones((trainX.shape[0], 1),
                                                          dtype=trainX.dtype)])
        if validX is not None:
            validX_intercept = np.hstack([validX, np.ones((validX.shape[0], 1),
                                                          dtype=validX.dtype)])
    else:
        trainX_intercept = trainX
        validX_intercept = validX
    if validX is not None:
        testvalidY = np.dot(validX_intercept, enet.X.T)

    print("testvalidY (newvalidY should be this)")
    inverse_logit = lambda t: 1 / (1 + math.exp(-t))
    func = np.vectorize(inverse_logit)
    print(func(testvalidY))

    print("Predicting, assuming unity weights")
    if validX is None or mvalid == 0:
        print("Using trainX for validX")
        newvalidY = enet.predict(trainX)  # for testing
    else:
        print("Using validX for validX")
        newvalidY = enet.predict(validX)
    print("newvalidY")
    print(newvalidY)
    print("newvalidY Predictions")
    print((newvalidY))
    print("Predictions max")
    print(newvalidY.max())
    print("Predictions min")
    print(newvalidY.min())
    print("Done Reporting")
    return enet


def test_elastic_net_sklearn():
    import numpy as np
    import pandas as pd


    df = pd.read_csv("./open_data/simple.txt", sep=" ", header=None)
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    elastic_net(X, y, nGPUs=1, nlambda=100, nfolds=1, nalpha=1, validFraction=0.2, family="elasticnet", verbose=0)

if __name__ == "__main__":
    test_elastic_net_sklearn()
