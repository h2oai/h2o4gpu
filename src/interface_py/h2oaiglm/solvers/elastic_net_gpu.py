from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU
from h2oaiglm.solvers.elastic_net_base import ElasticNetBaseSolver

if not h2oaiglmElasticNetGPU:
	print('\nWarning: Cannot create a H2OAIGLM Elastic Net GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
	print('> Setting h2oaiglm.ElasticNetSolverGPU=None')
	print('> Use h2oaiglm.ElasticNetSolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n')
	ElasticNetSolverGPU=None
else:
	class ElasticNetSolverGPU(object):
		def __init__(self, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas):
			self.solver = ElasticNetBaseSolver(h2oaiglmElasticNetGPU, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas)

		def upload_data(self, sourceDev, trainX, trainY, validX, validY):
			return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY)

		def fit(self, sourceDev, mTrain, n, mValid, intercept, standardize, a, b, c, d):
			return self.solver.fit(sourceDev, mTrain, n, mValid, intercept, standardize, a, b, c, d)
