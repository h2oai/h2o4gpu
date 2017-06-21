from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.solvers.elastic_net_base import ElasticNetBaseSolver
from ctypes import *

if not h2oaiglmElasticNetCPU:
    print('\nWarning: Cannot create a H2OAIGLM Elastic Net CPU Solver instance without linking Python module to a compiled H2OAIGLM CPU library')
    print('> Setting h2oaiglm.ElasticNetSolverCPU=None')
    print('> Use h2oaiglm.ElasticNetSolverGPU(args...) and re-run setup.py\n\n')
    ElasticNetSolverCPU=None
else:
    class ElasticNetSolverCPU(object):
        def __init__(self, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas):
            self.solver = ElasticNetBaseSolver(h2oaiglmElasticNetCPU, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas)

        def upload_data(self, sourceDev, trainX, trainY, validX, validY, weight):
            return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY, weight)

        def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath):
            return self.solver.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath)

        def fit(self, trainX, trainY, validX=c_void_p(0), validY=c_void_p(0), weight=c_void_p(0), givefullpath=0):
            return self.solver.fit(trainX, trainY, validX, validY, weight, givefullpath)
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
