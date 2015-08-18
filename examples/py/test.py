import sys
import numpy as np

# 1. REPLACE WITH '/path/to/pogs/'
POGS_ROOT = "/Users/Baris/Documents/Thesis/modules/pogs/"
# POGS_ROOT = "/home/baris/pogs/"
sys.path.append(POGS_ROOT+"src/interface_py/")
import pogs as pogs


def main(argv):
    gpu = len(argv)>0 and argv[0] in ["-g","-gpu","-G","-GPU"]
    if gpu and not pogs.SolverGPU:
        print "\nPOGS GPU library not compiled, please call again without `-g/gpu/G/GPU` option\n"
        return

    m=1000
    n=100

    A=np.random.rand(m,n)
    f=pogs.FunctionVector(m,double_precision=True)
    g=pogs.FunctionVector(n,double_precision=True)

    n_targ = 0;
    n_oar = 0;
    for b in f.b:
        if np.random.rand()>0.75:
            b=1
            n_targ+=1
        else:
            n_oar +=1

    wt_targ = 1
    wt_oar = np.true_divide(1,20)*np.true_divide(n_targ,n_oar)

    alpha = 0.05
    c = (alpha+1)/2
    d = (alpha-1)/2

    for i in xrange(len(f.b)):
        if np.random.rand() > 0.75:
            f.b[i]=1.
     

    for i,bi in enumerate(f.b):
        if bi >0:
            f.c[i] = c
            f.d[i] = d
        else:
            f.c[i] = wt_oar

    f.h[:]=pogs.FUNCTION["ABS"] 
    g.h[:]=pogs.FUNCTION["INDGE0"]
    if gpu:
        p = pogs.SolverGPU(m,n,A)
    else:
        p = pogs.SolverCPU(m,n,A)
    p.solve(f,g)
    pogs.STATUS[p.info.status]
    p.finish()


if __name__ == "__main__":
   main(sys.argv[1:])


