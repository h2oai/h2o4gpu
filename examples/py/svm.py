import h2oaiglm as h2oaiglm
import numpy as np
from numpy import float32, floor, hstack, ones
from numpy.random import rand, randn

'''
Support vector machine.
   minimize    (1/2) ||w||_2^2 + \lambda \sum (a_i^T * [w; b] + 1)_+.

See <h2oaiglm>/matlab/examples/svm.m for detailed description.
'''

def Svm(m,n, gpu=False, double_precision=False):
  # set solver cpu/gpu according to input args
  if gpu and h2oaiglm.SolverGPU is None:
    print("\nGPU solver unavailable, using CPU solver\n")
    gpu=False

  Solver = h2oaiglm.SolverGPU if gpu else h2oaiglm.SolverCPU

  # Generate A according to:
  #   X = [randn(m/2, n) + ones(m/2, n); randn(m/2, n) - ones(m/2, n)]
  #   yhat = [ones(m/2, 1); -ones(m/2, 1)]
  #   A = [(-yhat * ones(1, n)) .* X, -yhat]
  ind_half = int(floor(m/2.))

  X=randn(m,n)
  X[:ind_half,:]+=1
  X[ind_half:,:]-=1

  yhat=ones((m,1))
  yhat[ind_half:]*=-1

  A = -yhat*hstack((X,ones((m,1))))

  # cast A as float/double according to input args
  A=A if double_precision else np.float32(A)

  _lambda = 1.

  # f(y)
  f = h2oaiglm.FunctionVector(m,double_precision=double_precision)
  f.b[:]=-1
  f.c[:]=_lambda
  f.h[:]=h2oaiglm.FUNCTION["MAXPOS0"]


  # g( [w; b] ) 
  g = h2oaiglm.FunctionVector(n+1,double_precision=double_precision)
  g.a[:] = 0.5
  g.h[:-1] = h2oaiglm.FUNCTION["SQUARE"]
  g.h[:-1] = h2oaiglm.FUNCTION["ZERO"]


  # intialize solver 
  s = Solver(A) 

  # solve
  s.solve(f, g)

  # get solve time
  t = s.info.solvetime

  # tear down solver in C++/CUDA
  s.finish()

  return t

if __name__ == "__main__":
   print("Solve time:\t{:.2e} seconds".format(Svm(1000,200)))

