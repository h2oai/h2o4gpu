import pogs as pogs
import numpy as np
from numpy import float32, floor, ones
from numpy.random import rand, randn

'''
Non-negative least squares.
  minimize    (1/2) ||Ax - b||_2^2
  subject to  x >= 0.

See <pogs>/matlab/examples/nonneg_l2.m for detailed description.
'''

def NonNegL2(m,n, gpu=False, double_precision=False):
  # set solver cpu/gpu according to input args
  if gpu and pogs.SolverGPU is None:
    print "\nGPU solver unavailable, using CPU solver\n"
    gpu=False

  Solver = pogs.SolverGPU if gpu else pogs.SolverCPU

  # Generate A according to:
  #   A = 1 / n * rand(m, n)
  A = rand(m,n)/n

  # cast A as float/double according to input args
  A=A if double_precision else float32(A)

  # Generate b according to:
  #   n_half = floor(2 * n / 3);
  #   b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1)
  n_half = int(floor(2.*n/3.))  
  right_vec = ones(n)
  right_vec[n_half:]*=-1
  b=A.dot(right_vec)+0.1*randn(m)


  # f(Ax) = 1/2 || A*x - b ||_2^2 
  f = pogs.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.c[:]=0.5
  f.h[:]=pogs.FUNCTION["SQUARE"]

  # f(x) = Ind( x >= 0 )
  g = pogs.FunctionVector(n,double_precision=double_precision)
  g.h[:] = pogs.FUNCTION["INDGE0"]


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
   print "Solve time:\t{:.2e} seconds".format(NonNegL2(1000,200))




