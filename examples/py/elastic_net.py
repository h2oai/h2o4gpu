import h2ogpuml as h2ogpuml
import numpy as np
from numpy import abs, exp, float32, float64, log, max, sum, zeros


'''
Elastic Net

   minimize    (1/2) ||Ax - b||_2^2 + \alpha * \lambda ||x||_1 + 0.5 * (1-\alpha) * \lambda ||x||_2

   for 100 values of \lambda, and alpha in [0,1]
   See <h2ogpuml>/matlab/examples/lasso_path.m for detailed description.
'''

def ElasticNet(X, y, gpu=True, double_precision=False, nlambda=100, alpha=0.5):
  # set solver cpu/gpu according to input args
  if gpu and h2ogpuml.SolverGPU is None:
    print("\nGPU solver unavailable, using CPU solver\n")
    gpu=False

  Solver = h2ogpuml.SolverGPU if gpu else h2ogpuml.SolverCPU

  # cast A as float/double according to input args
  A=np.array(X, dtype='float64') if double_precision else np.array(X, dtype='float32')
  m = A.shape[0]
  n = A.shape[1]

  b=y

  # lambda_max
  lambda_max = max(abs(A.T.dot(b)))


  # f(Ax) = ||Ax - b||_2^2
  f = h2ogpuml.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.h[:]=h2ogpuml.FUNCTION["SQUARE"]

  # g(x) = 0.2*lambda_max*||x||_1
  g = h2ogpuml.FunctionVector(n,double_precision=double_precision)
  g.a[:] = 1
  g.h[:] = h2ogpuml.FUNCTION["ABS"]


  # store results for comparison 
  x_prev = zeros(n)

  # timer
  runtime = 0.

  #s = Solver(sp.sparse.csr_matrix(A)) ##TODO: compare to sparse
  #s = Solver(sp.sparse.csc_matrix(A))
  s = Solver(A)

  for i in list(range(nlambda)):
    _lambda= exp( (log(lambda_max)*(nlambda-1-i)+1e-2*log(lambda_max)*i )/ (nlambda-1))

    g.c[:]=    alpha*_lambda
    g.e[:]=(1-alpha)*_lambda

    # solve ##TODO: let C++ do the lambda-path
    s.solve(f, g)

    # add run time
    runtime += s.info.solvetime

    # copy
    x_curr=s.solution.x

    # check stopping condition
#    if max(abs(x_prev-x_curr)) < 1e-3* sum(abs(x_curr)):
#      break
    
    x_prev[:]=x_curr[:]


  # tear down solver in C++/CUDA
  s.finish()

  return runtime

if __name__ == "__main__":
  from numpy.random import rand,randn
  m=100
  n=10
  A=randn(m,n)
  x_true=(randn(n)/n)*float64(randn(n)<0.8)
  b=A.dot(x_true)+0.5*randn(m)
  ElasticNet(A, b, gpu=True, double_precision=True, nlambda=100, alpha=0.5)
