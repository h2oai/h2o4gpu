import h2ogpuml as h2ogpuml
from h2ogpuml.types import *

'''
Elastic Net

   minimize    (1/2) ||Ax - b||_2^2 + \alpha * \lambda ||x||_1 + 0.5 * (1-\alpha) * \lambda ||x||_2

   for 100 values of \lambda, and alpha in [0,1]
   See <h2ogpuml>/matlab/examples/lasso_path.m for detailed description.
'''


def elastic_net(X, y, nGPUs=0, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2):
    # choose solver
    Solver = h2ogpuml.GLM

    sharedA = 0
    nThreads = None  # let internal method figure this out
    intercept = 1
    standardize = 0
    lambda_min_ratio = 1e-9
    nFolds = nfolds
    nLambdas = nlambda
    nAlphas = nalpha

    if standardize:
        print("implement standardization transformer")
        exit()

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
    validX = np.copy(X[H:-1, :])

    mTrain = trainX.shape[0]
    mvalid = validX.shape[0]
    print("mTrain=%d mvalid=%d" % (mTrain, mvalid))

    if intercept == 1:
        trainX = np.hstack([trainX, np.ones((trainX.shape[0], 1), dtype=trainX.dtype)])
        validX = np.hstack([validX, np.ones((validX.shape[0], 1), dtype=validX.dtype)])
        n = trainX.shape[1]
        print("New n=%d" % n)

    ## Constructor
    print("Setting up solver")
    enet = Solver(sharedA, nThreads, nGPUs, 'c' if fortran else 'r', intercept, standardize, lambda_min_ratio, nLambdas,
                  nFolds, nAlphas)

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    ## Solve
    print("Solving")
    Xvsalpha = enet.fit(trainX, trainY)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW, 0)
    # givefullpath=1
    #  Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW, givefullpath)
    print("Done Solving")

    # show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(Xvsalpha)
    print("np.shape(Xvsalpha)")
    print(np.shape(Xvsalpha))

    rmse = enet.getrmse()
    print("rmse")
    print(rmse)

    print("lambdas")
    lambdas = enet.getlambdas()
    print(lambdas)

    print("alphas")
    alphas = enet.getalphas()
    print(alphas)

    print("tols")
    tols = enet.gettols()
    print(tols)

    testvalidY = np.dot(trainX, Xvsalpha.T)
    print("testvalidY (newvalidY should be this)")
    print(testvalidY)

    print("Predicting, assuming unity weights")
    if validX == None or mvalid == 0:
        print("Using trainX for validX")
        newvalidY = enet.predict(trainX)  # for testing
    else:
        print("Using validX for validX")
        newvalidY = enet.predict(validX)
    print("newvalidY")
    print(newvalidY)

    print("Done Reporting")
    return enet


if __name__ == "__main__":
    import numpy as np
    #from numpy.random import randn
    #  m=1000
    #  n=100
    #  A=randn(m,n)
    #  x_true=(randn(n)/n)*float64(randn(n)<0.8)
    #  b=A.dot(x_true)+0.5*randn(m)
    import pandas as pd
    #import feather

    # NOTE: cd ~/h2oai-prototypes/glm-bench/ ; gunzip ipums.csv.gz ; Rscript h2oai-prototypes/glm-bench/ipums.R to produce ipums.feather
    df = feather.read_dataframe("../../../h2oai-prototypes/glm-bench/ipums.feather")
    # df = pd.read_csv("../cpp/train.txt", sep=" ", header=None)
    #df = pd.read_csv("../cpp/simple.txt", sep=" ", header=None)
    #df = pd.read_csv("Hyatt_Subset.csv")
    #df = pd.read_csv("Hyatt_Subset.nohead.csv")
    print(df.shape)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    # elastic_net(X, y, nGPUs=2, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2)
    elastic_net(X, y, nGPUs=1, nlambda=100, nfolds=1, nalpha=1, validFraction=0)
    # elastic_net(X, y, nGPUs=0, nlambda=100, nfolds=1, nalpha=1, validFraction=0)
