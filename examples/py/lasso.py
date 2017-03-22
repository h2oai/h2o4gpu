import pogs as pogs
from numpy import abs, float32, float64, max, sqrt
from numpy.random import rand, randn


'''
Lasso

  minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1

See <pogs>/matlab/examples/lasso.m for detailed description.
template <typename T>
'''

def Lasso(m,n, gpu=False, double_precision=False):
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
  x_true=(randn(n)/sqrt(n))*float64(randn(n)<0.8)

  # b= A*x_true + v (noise)
  b=A.dot(x_true)+0.5*randn(m)

  # lambda
  lambda_max = max(abs(A.T.dot(b)))


  # f(Ax) = ||Ax - b||_2^2
  f = pogs.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.h[:]=pogs.FUNCTION["SQUARE"]

  # g(x) = 0.2*lambda_max*||x||_1
  g = pogs.FunctionVector(n,double_precision=double_precision)
  g.a[:] = 0.2*lambda_max 
  g.h[:] = pogs.FUNCTION["ABS"]

  # use problem data A to create solver 
  s = Solver(A) 

  # solve
  s.solve(f, g)

  # get solve time
  t = s.info.solvetime

  # tear down solver in C++/CUDA
  s.finish()

  return t

if __name__ == "__main__":
   print "Solve time:\t{:.2e} seconds".format(Lasso(200,2000))

