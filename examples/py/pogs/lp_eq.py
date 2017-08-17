import h2ogpuml as h2ogpuml
from h2ogpuml.types import H2OFunctions
from numpy import float32, vstack
from numpy.random import rand, randn

'''
Linear program in equality form.
  minimize    c^T * x
  subject to  Ax = b
               x >= 0.

See <h2ogpuml>/matlab/examples/lp_eq.m for detailed description.
'''


def lp_eq(m, n, gpu=False, double_precision=False):
    # Generate A and c according to:
    #   A = 1 / n * rand(m, n)
    #   c = 1 / n * rand(n, 1)

    A = rand(m, n) / n
    c = rand(n, 1) / n

    # Generate b according to:
    #   v = rand(n, 1)
    #   b = A * v
    b = A.dot(rand(n))

    # Gather A and c into one matrix
    A = vstack((A, c.T))

    # cast A as float/double according to input args
    A = A if double_precision else float32(A)

    # f(Ax) = Ind ( (Ax-b) == 0 ) + c^Tx
    f = h2ogpuml.FunctionVector(m + 1, double_precision=double_precision)
    f.b[:-1] = b[:]
    f.h[:-1] = H2OFunctions.INDEQ0
    f.h[-1] = H2OFunctions.IDENTITY.value

    # g(x) = Ind( x >= 0 )
    g = h2ogpuml.FunctionVector(n, double_precision=double_precision)
    g.h[:] = H2OFunctions.INDGE0

    # intialize solver
    s = h2ogpuml.Pogs(A) if gpu else h2ogpuml.Pogs(A, n_gpus=0)

    # solve
    s.fit(f, g)

    # get solve time
    t = s.info.solvetime

    # tear down solver in C++/CUDA
    s.finish()

    return t


if __name__ == "__main__":
    print("Solve time:\t{:.2e} seconds".format(lp_eq(1000, 200)))
