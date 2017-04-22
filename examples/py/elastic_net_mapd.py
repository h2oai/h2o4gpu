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

def ElasticNet(trainX, trainY, gpu=True, double_precision=False, nlambda=100, nalpha=16):
  # set solver cpu/gpu according to input args
  if gpu and pogs.ElasticNetSolverGPU is None:
    print("\nGPU solver unavailable, using CPU solver\n")
    gpu=False

  Solver = pogs.ElasticNetSolverGPU if gpu else pogs.ElasticNetSolverCPU
  assert Solver != None, "Couldn't instantiate ElasticNetSolver"


  sourceDev = 0
  nThreads = 2
  nGPUs = 2
  intercept = 1
  standardize = 0
  lambda_min_ratio = 1e-5
  nLambdas = nlambda
  nAlphas = nalpha

  if standardize:
    print ("implement standardization transformer")
    exit()

  validX = trainX# TODO FIXME
  validY = trainY# TODO FIXME

  ## TODO: compute these in C++ (CPU or GPU)
  sdTrainY = np.sqrt(np.var(y))
  print("sdTrainY: " + str(sdTrainY))
  meanTrainY = np.mean(y)
  print("meanTrainY: " + str(meanTrainY))
  sdValidY = np.sqrt(np.var(y))
  print("sdValidY: " + str(sdValidY))
  meanValidY = np.mean(y)
  print("meanValidY: " + str(meanValidY))
  mTrain = trainX.shape[0]
  mValid = validX.shape[0] if validX is None else 0
  fortran = trainX.flags.f_contiguous


  weights = 1./mTrain
  if intercept==1:
    lambda_max0 = weights * max(abs(trainX.T.dot(trainY-meanTrainY)))
  else:
    lambda_max0 = weights * max(abs(trainX.T.dot(trainY)))

  print("lambda_max0: " + str(lambda_max0))

  if intercept==1:
    trainX = np.hstack([trainX, np.ones((trainX.shape[0],1))])
    validX = np.hstack([validX, np.ones((validX.shape[0],1))])

  n = trainX.shape[1]
  print(mTrain)
  print(n)

  ## Constructor
  enet = Solver(nThreads, nGPUs, 'c' if fortran else 'r', intercept, standardize, lambda_min_ratio, nLambdas, nAlphas, double_precision)

  ## First, get backend pointers
  a,b,c,d = enet.upload_data(sourceDev, trainX, trainY, validX, validY)

  ## Solve
  enet.fit(sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d)

  return enet

if __name__ == "__main__":
  import numpy as np
  from numpy.random import randn
#  m=1000
#  n=100
#  A=randn(m,n)
#  x_true=(randn(n)/n)*float64(randn(n)<0.8)
#  b=A.dot(x_true)+0.5*randn(m)
  import pandas as pd
  import feather
  #df = feather.read_dataframe("../../../h2oai-prototypes/glm-bench/ipums.feather")
  df = pd.read_csv("../cpp/ipums.1k.txt", sep=" ", header=None)
  print(df.shape)
  X = np.array(df.iloc[:,:df.shape[1]-1], dtype='float32', order='C')
  y = np.array(df.iloc[:, df.shape[1]-1], dtype='float32', order='C')
  ElasticNet(X, y, gpu=True, double_precision=False, nlambda=100, nalpha=16)
