from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU
from h2oaiglm.solvers.elastic_net_base import ElasticNetBaseSolver
from ctypes import *

if not h2oaiglmElasticNetGPU:
    print('\nWarning: Cannot create a H2OAIGLM Elastic Net GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
    print('> Setting h2oaiglm.ElasticNetSolverGPU=None')
    print('> Use h2oaiglm.ElasticNetSolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n')
    ElasticNetSolverGPU=None
else:
    class ElasticNetSolverGPU(object):
        def __init__(self, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas):
            self.solver = ElasticNetBaseSolver(h2oaiglmElasticNetGPU, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas)
            
        def upload_data(self, sourceDev, trainX, trainY, validX, validY, weight):
            return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY, weight)
        
        def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c=c_void_p(0), d=c_void_p(0), e=c_void_p(0), givefullpath=0,dopredict=0):
            return self.solver.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath,dopredict)
        
        def fit(self, trainX, trainY, validX=c_void_p(0), validY=c_void_p(0), weight=c_void_p(0), givefullpath=0, dopredict=0):
            return self.solver.fit(trainX, trainY, validX, validY, weight, givefullpath, dopredict)
        def getrmse(self):
            return self.solver.getrmse()
        def getlambdas(self):
            return self.solver.getlambdas()
        def getalphas(self):
            return self.solver.getalphas()
        def gettols(self):
            return self.solver.gettols()
        def predict(self, validX, testweight=None, givefullpath=0):
            return self.solver.predict(validX, testweight, givefullpath)
        def fit_predict(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0):
            return self.solver.fit_predict(trainX, trainY, validX, validY, weight, givefullpath)
        def finish1(self):
            return self.solver.finish1()
        def finish2(self):
            return self.solver.finish2()
        def finish3(self):
            return self.solver.finish3()
        def finish(self):
            return self.solver.finish()
        
