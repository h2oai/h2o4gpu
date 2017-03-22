import pogs as pogs
from numpy import abs, exp, float32, float64, max
from numpy.random import rand, randn

'''
Logistic regression

  minimize    \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
  subject to  y = Ax

See <pogs>/matlab/examples/logistic_regression.m for detailed description.
'''

def Logistic(m,n, gpu=False, double_precision=False):
  # set solver cpu/gpu according to input args
  if gpu and pogs.SolverGPU is None:
    print "\nGPU solver unavailable, using CPU solver\n"
    gpu=False

  Solver = pogs.SolverGPU if gpu else pogs.SolverCPU

  # random matrix A
  A=randn(m,n)

  # cast A as float/double according to input args
  A=A if double_precision else float32(A)

  # true x vector, ~20% zeros
  x_true=(randn(n)/n)*float64(randn(n)<0.8)

  # generate labels  
  d = 1./(1+exp(-A.dot(x_true))) > rand(m)

  # lambda_max
  lambda_max = max(abs(A.T.dot(0.5-d)))

  # f(y) = \sum_i -d_i y_i + log(1 + e ^ y_i)
  f = pogs.FunctionVector(m,double_precision=double_precision)
  f.d[:]=-d[:]
  f.h[:]=pogs.FUNCTION["LOGISTIC"]


  # g(x) = \lambda ||x||_1
  g = pogs.FunctionVector(n,double_precision=double_precision)
  g.a[:] = 0.5*lambda_max 
  g.h[:] = pogs.FUNCTION["ABS"]


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
   print "Solve time:\t{:.2e} seconds".format(Logistic(1000,100))


