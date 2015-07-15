import sys
import numpy as np
# 1. REPLACE WITH '/path/to/pogs'
POGS_ROOT = "/home/baris/pogs/"
sys.path.append(POGS_ROOT+"src/interface_c/")
import pogs as pogs


m=40
n=20


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
        
for i,b in enumerate(f.b):
    if b>0:
        f.c[i] = c
        f.d[i] = d
    else:
        f.c[i] = wt_oar

f.h[:]=pogs.FUNCTION["ABS"] 
g.h[:]=pogs.FUNCTION["INDGE0"]
pgs = pogs.Solver(m,n,A)
pgs.solve(f,g)
pogs.STATUS[pgs.info.status]
# pgs.finish()