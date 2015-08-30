import sys
import numpy as np
import pogs as pogs


def main(argv):
    gpu = len(argv)>0 and argv[0] in ["-g","-gpu","-G","-GPU"]
    if gpu and not pogs.SolverGPU:
        print "\nPOGS GPU library not compiled, please call again without `-g/gpu/G/GPU` option\n"
        return
    Solver = pogs.SolverGPU if gpu else pogs.SolverCPU

    m=1000
    n=100

    A=np.random.rand(m,n)
    f=pogs.FunctionVector(m,double_precision=True)
    g=pogs.FunctionVector(n,double_precision=True)


    wt_1_over=25
    wt_1_under=35
    wt_2 =1

    for i in xrange(m):
        if np.random.rand()>0.75:
            # f.a[i]=1 (from initialization)
            f.b[i]=1
            f.c[i]=(wt_1_over+wt_1_under)/2
            f.d[i]=(wt_1_over-wt_1_under)/2
            # f.e[i]=0 (from initialization)
        else:
            # f.a[i]=1 (from initialization)
            # f.b[i]=0 (from initialization)
            f.c[i]=wt_2
            # f.d[i]=0 (from initialization)
            # f.e[i]=0 (from initialization)
         
    # f(y) =  c|y-b|+dy --- skewed 1-norm, or equivalently piecewise linear 
    f.h[:]=pogs.FUNCTION["ABS"]

    # g(x) = I(x>= 0) 
    g.h[:]=pogs.FUNCTION["INDGE0"]

    # initialize solver
    s=Solver(A)

    # solve
    s.solve(f,g)
    print pogs.STATUS[s.info.status]

    # tear down solver
    s.finish()


if __name__ == "__main__":
   main(sys.argv[1:])


