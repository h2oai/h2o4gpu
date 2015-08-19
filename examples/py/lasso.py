import pogs as pogs
import numpy as np
from numpy.random import rand, randn


'''
Lasso
   
  minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1

See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
'''

def Lasso(m,n, gpu=False, double_precision=False):
  # set solver
  if gpu and pogs.SolverGPU is None:
    print "\nGPU solver unavailable, using CPU solver\n"
    gpu=False

  Solver = pogs.SolverGPU if gpu else pogs.SolverCPU

  # random matrix A
  A=randn(m,n)

  # true x vector, ~20% zeros
  x_true=(randn(n)/np.sqrt(n))*np.float64(randn(n)<0.8)

  # b= A*x_true + v (noise)
  b=A.dot(x_true)+0.5*randn(m)

  # lambda
  lambda_max = np.max(A.T.dot(b))

  # use problem data A to create solver (select GPU/CPU and float/double according to input args)
  sol = Solver(A) if double_precision else Solver(np.float32(A))

  # f(Ax) = ||Ax - b||_2^2
  f = pogs.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.h[:]=pogs.FUNCTION["SQUARE"]

  # g(x) = 0.2*lambda_max*||x||_1
  g = pogs.FunctionVector(n,double_precision=double_precision)
  g.a[:] = 0.2*lambda_max 
  g.h[:] = pogs.FUNCTION["ABS"]

  # solve
  sol.solve(f, g)

  # get solve time
  t = sol.info.solvetime

  # tear down solver in C++/CUDA
  sol.finish()

  return t


