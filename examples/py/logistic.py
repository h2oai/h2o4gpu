import h2ogpuml as h2ogpuml
from h2ogpuml.types import H2OFunctions
from numpy import abs, exp, float32, float64, max
from numpy.random import rand, randn

'''
Logistic regression

  minimize    \sum_i -d_i y_i + log(1 + e ^ y_i) + \lambda ||x||_1
  subject to  y = Ax

See <h2ogpuml>/matlab/examples/logistic_regression.m for detailed description.
'''


def logistic(m, n, gpu=False, double_precision=False):
    # random matrix A
    A = randn(m, n)

    # cast A as float/double according to input args
    A = A if double_precision else float32(A)

    # true x vector, ~20% zeros
    x_true = (randn(n) / n) * float64(randn(n) < 0.8)

    # generate labels
    d = 1. / (1 + exp(-A.dot(x_true))) > rand(m)

    # lambda_max
    lambda_max = max(abs(A.T.dot(0.5 - d)))

    # f(y) = \sum_i -d_i y_i + log(1 + e ^ y_i)
    f = h2ogpuml.FunctionVector(m, double_precision=double_precision)
    f.d[:] = -d[:]
    f.h[:] = H2OFunctions.LOGISTIC

    # g(x) = \lambda ||x||_1
    g = h2ogpuml.FunctionVector(n, double_precision=double_precision)
    g.a[:] = 0.5 * lambda_max
    g.h[:] = H2OFunctions.ABS

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
    print("Solve time:\t{:.2e} seconds".format(logistic(1000, 100)))
