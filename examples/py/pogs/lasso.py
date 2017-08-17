import h2ogpuml as h2ogpuml
from h2ogpuml.types import H2OFunctions
from numpy import abs, float32, float64, max, sqrt
from numpy.random import rand, randn

'''
Lasso

  minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1

See <h2ogpuml>/matlab/examples/lasso.m for detailed description.
template <typename T>
'''


def lasso(m, n, gpu=True, double_precision=False):
    # random matrix A
    A = randn(m, n)

    # cast A as float/double according to input args
    A = A if double_precision else float32(A)

    # true x vector, ~20% zeros
    x_true = (randn(n) / sqrt(n)) * float64(randn(n) < 0.8)

    # b= A*x_true + v (noise)
    b = A.dot(x_true) + 0.5 * randn(m)

    # lambda
    lambda_max = max(abs(A.T.dot(b)))

    # f(Ax) = ||Ax - b||_2^2
    f = h2ogpuml.FunctionVector(m, double_precision=double_precision)
    f.b[:] = b[:]
    f.h[:] = H2OFunctions.SQUARE

    # g(x) = 0.2*lambda_max*||x||_1
    g = h2ogpuml.FunctionVector(n, double_precision=double_precision)
    g.a[:] = 0.2 * lambda_max
    g.h[:] = H2OFunctions.ABS

    # use problem data A to create solver
    s = h2ogpuml.Pogs(A) if gpu else h2ogpuml.Pogs(A, n_gpus=0)

    # solve
    s.fit(f, g)

    # get solve time
    t = s.info.solvetime

    # tear down solver in C++/CUDA
    s.finish()

    return t


if __name__ == "__main__":
    print("Solve time:\t{:.2e} seconds".format(lasso(10000000, 100)))
