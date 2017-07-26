import h2ogpuml as h2ogpuml
from h2ogpuml.types import H2OFunctions
import sys
import numpy as np


def main(argv):
    gpu = len(argv) > 0 and argv[0] in ["-g", "-gpu", "-G", "-GPU"]

    m = 1000
    n = 100

    A = np.random.rand(m, n)
    f = h2ogpuml.FunctionVector(m, double_precision=True)
    g = h2ogpuml.FunctionVector(n, double_precision=True)

    wt_1_over = 25
    wt_1_under = 35
    wt_2 = 1

    for i in range(m):
        if np.random.rand() > 0.75:
            # f.a[i]=1 (from initialization)
            f.b[i] = 1
            f.c[i] = (wt_1_over + wt_1_under) / 2
            f.d[i] = (wt_1_over - wt_1_under) / 2
            # f.e[i]=0 (from initialization)
        else:
            # f.a[i]=1 (from initialization)
            # f.b[i]=0 (from initialization)
            f.c[i] = wt_2
            # f.d[i]=0 (from initialization)
            # f.e[i]=0 (from initialization)

    # f(y) =  c|y-b|+dy --- skewed 1-norm, or equivalently piecewise linear 
    f.h[:] = H2OFunctions.ABS

    # g(x) = I(x>= 0) 
    g.h[:] = H2OFunctions.INDGE0

    # initialize solver
    s = h2ogpuml.Pogs(A) if gpu else h2ogpuml.Pogs(A, n_gpus=0)

    # solve
    s.solve(f, g)
    print((h2ogpuml.STATUS[s.info.status]))

    # tear down solver
    s.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
