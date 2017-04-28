import h2oaiglm as h2oaiglm
import scipy as sp
from numpy import abs, exp, float32, float64, log, max, sum, zeros
from numpy.random import rand, randn


'''
LassoPath

   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1

   for 50 values of \lambda.
   See <h2oaiglm>/matlab/examples/lasso_path.m for detailed description.
'''

def LassoPath(m, n, gpu=True, double_precision=False, nlambda=50):
  # set solver cpu/gpu according to input args
  if gpu and h2oaiglm.SolverGPU is None:
    print("\nGPU solver unavailable, using CPU solver\n")
    gpu=False

  Solver = h2oaiglm.SolverGPU if gpu else h2oaiglm.SolverCPU

  # random matrix A
  A=randn(m,n)

  # cast A as float/double according to input args
  A=A if double_precision else float32(A)

  # true x vector, ~20% zeros
  x_true=(randn(n)/n)*float64(randn(n)<0.8)

  # b= A*x_true + v (noise)
  b=A.dot(x_true)+0.5*randn(m)

  # lambda_max
  lambda_max = max(abs(A.T.dot(b)))



  # f(Ax) = ||Ax - b||_2^2
  f = h2oaiglm.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.h[:]=h2oaiglm.FUNCTION["SQUARE"]

  # g(x) = 0.2*lambda_max*||x||_1
  g = h2oaiglm.FunctionVector(n,double_precision=double_precision)
  g.a[:] = 0.2*lambda_max 
  g.h[:] = h2oaiglm.FUNCTION["ABS"]


  # store results for comparison 
  x_prev = zeros(n)

  # timer
  runtime = 0.

  # use problem data A to create solver

  #s = Solver(sp.sparse.csr_matrix(A))
  #s = Solver(sp.sparse.csc_matrix(A))
  s = Solver(A)

  for i in range(nlambda):
    _lambda= exp( (log(lambda_max)*(nlambda-1-i)+1e-2*log(lambda_max)*i )/ (nlambda-1))

    g.c[:]=_lambda

    # solve
    s.solve(f, g)

    # add run time
    runtime += s.info.solvetime

    # copy
    x_curr=s.solution.x

    # check stopping condition
    if max(abs(x_prev-x_curr)) < 1e-3* sum(abs(x_curr)):
      break
    
    x_prev[:]=x_curr[:]


  # tear down solver in C++/CUDA
  s.finish()

  return runtime

if __name__ == "__main__":
   print("Solve time:\t{:.2e} seconds".format(LassoPath(200000,1000)))




