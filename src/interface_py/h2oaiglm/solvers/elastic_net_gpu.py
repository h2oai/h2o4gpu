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

                def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath):
                        return self.solver.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath)

                def fit(self, trainX, trainY, validX=c_void_p(0), validY=c_void_p(0), weight=c_void_p(0), givefullpath=0):
                        return self.solver.fit(trainX, trainY, validX, validY, weight, givefullpath)
