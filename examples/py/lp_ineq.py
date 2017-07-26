import h2ogpuml as h2ogpuml
from h2ogpuml.types import H2OFunctions
from numpy import eye, float32, vstack
from numpy.random import rand, randn

'''
Linear program in equality form.
  minimize    c^T * x
  subject to  Ax <= b

See <h2ogpuml>/matlab/examples/lp_eq.m for detailed description.
'''


def lp_ineq(m, n, gpu=False, double_precision=False):
    # Generate A according to:
    #   A = [-1 / n *rand(m-n, n); -eye(n)]
    A = -vstack((rand(m - n, n) / n, eye(n)))

    # cast A as float/double according to input args
    A = A if double_precision else float32(A)

    # Generate b according to:
    #   b = A * rand(n, 1) + 0.2 * rand(m, 1)
    b = A.dot(rand(n)) + 0.2 * rand(m)

    # Generate c according to:
    #   c = rand(n, 1)
    c = rand(n)

    # f(x) = Ind( (Ax-b) <= 0 )
    f = h2ogpuml.FunctionVector(m, double_precision=double_precision)
    f.b[:] = b[:]
    f.h[:] = H2OFunctions.IDENTITY

    # g(x) = c^Tx
    g = h2ogpuml.FunctionVector(n, double_precision=double_precision)
    g.a[:] = c[:]
    g.h[:] = H2OFunctions.IDENTITY

    # intialize solver
    s = h2ogpuml.Pogs(A) if gpu else h2ogpuml.Pogs(A, n_gpus=0)

    # solve
    s.solve(f, g)

    # get solve time
    t = s.info.solvetime

    # tear down solver in C++/CUDA
    s.finish()

    return t


if __name__ == "__main__":
    print("Solve time:\t{:.2e} seconds".format(lp_ineq(1000, 200)))
