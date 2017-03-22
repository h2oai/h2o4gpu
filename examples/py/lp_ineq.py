import pogs as pogs
from numpy import eye, float32, vstack
from numpy.random import rand, randn

'''
Linear program in equality form.
  minimize    c^T * x
  subject to  Ax <= b

See <pogs>/matlab/examples/lp_eq.m for detailed description.
'''

def LpIneq(m,n, gpu=False, double_precision=False):
  # set solver cpu/gpu according to input args
  if gpu and pogs.SolverGPU is None:
    print "\nGPU solver unavailable, using CPU solver\n"
    gpu=False

  Solver = pogs.SolverGPU if gpu else pogs.SolverCPU

  # Generate A according to:
  #   A = [-1 / n *rand(m-n, n); -eye(n)]  
  A=-vstack((rand(m-n,n)/n,eye(n)))

  # cast A as float/double according to input args
  A=A if double_precision else float32(A)

  # Generate b according to:
  #   b = A * rand(n, 1) + 0.2 * rand(m, 1)
  b=A.dot(rand(n))+0.2*rand(m)

  # Generate c according to:
  #   c = rand(n, 1)
  c= rand(n)



  # f(x) = Ind( (Ax-b) <= 0 ) 
  f = pogs.FunctionVector(m,double_precision=double_precision)
  f.b[:]=b[:]
  f.h[:]=pogs.FUNCTION["IDENTITY"]

  # g(x) = c^Tx
  g = pogs.FunctionVector(n,double_precision=double_precision)
  g.a[:] = c[:] 
  g.h[:] = pogs.FUNCTION["IDENTITY"]

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
   print "Solve time:\t{:.2e} seconds".format(LpIneq(1000,200))

 
 


