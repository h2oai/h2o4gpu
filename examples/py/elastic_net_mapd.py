import sys
#sys.path.insert(0, "/home/arno/pogs/src/interface_py/")
import pogs as pogs
import numpy as np
from numpy import abs, exp, float32, float64, log, max, zeros

'''
Elastic Net

   minimize    (1/2) ||Ax - b||_2^2 + \alpha * \lambda ||x||_1 + 0.5 * (1-\alpha) * \lambda ||x||_2

   for 100 values of \lambda, and alpha in [0,1]
   See <pogs>/matlab/examples/lasso_path.m for detailed description.
'''

def ElasticNet(X, y, gpu=True, double_precision=False, nlambda=100, nalpha=16):
  # set solver cpu/gpu according to input args
  if gpu and pogs.ElasticNetSolverGPU is None:
    print("\nGPU solver unavailable, using CPU solver\n")
    gpu=False

  Solver = pogs.ElasticNetSolverGPU if gpu else pogs.ElasticNetSolverCPU
  assert Solver != None, "Couldn't instantiate ElasticNetSolver"

  A=np.array(X, dtype='float64') if double_precision else np.array(X, dtype='float32')

  sourceDev = 0
  nGPUs = 2
  intercept = 1
  standardize = 0
  lambda_min_ratio = 1e-5
  nLambdas = nlambda
  nAlphas = nalpha

  if standardize:
    print ("implement standardization transformer")
    exit()

  trainX = A
  trainY = y
  validX = None
  validY = None


  ## TODO: compute these in C++ (CPU or GPU)
  sdTrainY = np.sqrt(np.var(y))
  meanTrainY = np.mean(y)
  mTrain = trainX.shape[0]
  mValid = validX.shape[0] if validX != None else 0
  fortran = trainX.flags.f_contiguous


  if intercept==1:
    weights = 1./mTrain
    lambda_max0 = weights * max(abs(A.T.dot(y-meanTrainY)))
  else:
    lambda_max0 = max(abs(A.T.dot(y)))

  if intercept==1:
    A = np.hstack([A, np.ones((A.shape[0],1))])
    trainX = A

  n = trainX.shape[1]
  print(mTrain)
  print(n)

  ## Constructor
  enet = Solver(nGPUs, 'c' if fortran else 'r', intercept, lambda_min_ratio, nLambdas, nAlphas)

  ## First, get backend pointers
  a,b,c,d = enet.upload_data(sourceDev, trainX, trainY, validX, validY)

  ## Solve
  enet.fit(sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, a, b, c, d)

  return enet

if __name__ == "__main__":
  from numpy.random import randn
#  m=1000
#  n=100
#  A=randn(m,n)
#  x_true=(randn(n)/n)*float64(randn(n)<0.8)
#  b=A.dot(x_true)+0.5*randn(m)
  import pandas as pd
  df = pd.read_csv("../R/data.csv")
  X = np.array(df.ix[:,:10], dtype='float64', order='C')
  y = np.array(df.ix[:,10], dtype='float64')
  print(X)
  print(y)
  ElasticNet(X, y, gpu=True, double_precision=True, nlambda=100, nalpha=1)
