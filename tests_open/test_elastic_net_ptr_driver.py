# -*- encoding: utf-8 -*-
"""
ElasticNet_h2o4gpu solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import h2o4gpu as h2o4gpu
import numpy as np

from ctypes import *
from h2o4gpu.types import *


'''
Elastic Net

   minimize    (1/2) ||Ax - b||_2^2 + \alpha * \lambda ||x||_1 + 0.5 * (1-\alpha) * \lambda ||x||_2

   for 100 values of \lambda, and alpha in [0,1]
   See <h2o4gpu>/matlab/examples/lasso_path.m for detailed description.
'''

def ElasticNet(X, y, nGPUs=0, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2):
  Solver = h2o4gpu.ElasticNet_h2o4gpu

  sourceDev = 0
  intercept = True
  lambda_min_ratio = 1e-9
  nFolds = nfolds
  nLambdas = nlambda
  nAlphas = nalpha

  # Setup Train/validation Set Split
  morig = X.shape[0]
  norig = X.shape[1]
  print("Original m=%d n=%d" % (morig,norig))
  fortran = X.flags.f_contiguous
  print("fortran=%d" % (fortran))

  
  # Do train/valid split
  HO=int(validFraction*morig)
  H=morig-HO
  print("Size of Train rows=%d valid rows=%d" % (H,HO))
  trainX = np.copy(X[0:H,:])
  trainY = np.copy(y[0:H])
  validX = np.copy(X[H:norig,:])
  validY = np.copy(y[H:norig])
  trainW = np.copy(trainY)*0.0 + 1.0 # constant unity weight

  mTrain = trainX.shape[0]
  mvalid = validX.shape[0]
  print("mTrain=%d mvalid=%d" % (mTrain,mvalid))
  
  if intercept==1:
    trainX = np.hstack([trainX, np.ones((trainX.shape[0],1),dtype=trainX.dtype)])
    validX = np.hstack([validX, np.ones((validX.shape[0],1),dtype=validX.dtype)])
  n = trainX.shape[1]
  print("New n=%d" % (n))

  ## Constructor
  print("Setting up solver")
  enet = Solver(n_gpus = nGPUs,  fit_intercept = intercept, lambda_min_ratio = lambda_min_ratio, n_lambdas = nLambdas, n_folds = nFolds, n_alphas = nAlphas)

  ## First, get backend pointers
  print("Uploading")
  print(trainX.dtype)
  print(trainY.dtype)
  print(validX.dtype)
  print(validY.dtype)
  print(trainW.dtype)
  a,b,c,d,e = enet.prepare_and_upload_data(trainX, trainY, validX, validY, trainW, source_dev = sourceDev)

  ## Solve
  print("Solving")
  double_precision=0 # float
  order = 'c' if fortran else 'r'
  enet.fit_ptr(mTrain, n, mvalid, double_precision, order, a, b, c, d, e, source_dev = sourceDev)
  print("Done Solving")

  # show something about Xvsalphalambda and Xvsalpha
  print("Xvsalpha")
  print(enet.X)

  rmse=enet.error
  print("rmse")
  print(rmse)

  print("lambdas")
  lambdas=enet.lambdas
  print(lambdas)

  print("alphas")
  alphas=enet.alphas
  print(alphas)

  print("tols")
  tols=enet.tols
  print(tols)
                    

  print("Predicting")
  if 1==1:
       validPredsvsalphapure = enet.predict_ptr(c, d)
  else:
       validPredsvsalphapure = enet.predict(validX, validY)

  print("validPredsvsalphapure")
  print(validPredsvsalphapure)
          
  print("Done Predicting")

  # show something about validPredsvsalphalambdapure, validPredsvsalphapure

  return enet

def test_elastic_net_ptr_driver():
    import numpy as np
    from numpy.random import randn
    #  m=1000
    #  n=100
    #  A=randn(m,n)
    #  x_true=(randn(n)/n)*float64(randn(n)<0.8)
    #  b=A.dot(x_true)+0.5*randn(m)
    import pandas as pd
    import feather
    df = pd.read_csv("./open_data/simple.txt", sep=" ", header=None)
    print(df.shape)
    X = np.array(df.iloc[:,:df.shape[1]-1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1]-1], dtype='float32', order='C')
    #ElasticNet(X, y, nGPUs=2, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2)
    ElasticNet(X, y, nGPUs=1, nlambda=100, nfolds=1, nalpha=1, validFraction=0.2)

if __name__ == "__main__":
    test_elastic_net_ptr_driver()