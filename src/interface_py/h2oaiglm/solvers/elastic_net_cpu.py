from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.solvers.elastic_net_base import ElasticNetBaseSolver

if not h2oaiglmElasticNetCPU:
	print('\nWarning: Cannot create a H2OAIGLM Elastic Net CPU Solver instance without linking Python module to a compiled H2OAIGLM CPU library')
	print('> Setting h2oaiglm.ElasticNetSolverCPU=None')
	print('> Use h2oaiglm.ElasticNetSolverGPU(args...) and re-run setup.py\n\n')
	ElasticNetSolverCPU=None
else:
	class ElasticNetSolverCPU(object):
		def __init__(self, sharedA, nThreads, nCPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas):
			self.solver = ElasticNetBaseSolver(h2oaiglmElasticNetCPU, sharedA, nThreads, nCPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas)

		def upload_data(self, sourceDev, trainX, trainY, validX, validY, weight):
			return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY, weight)

		def fit(self, sourceDev, mTrain, n, mValid, intercept, standardize, precision, a, b, c, d, e):
			return self.solver.fit(sourceDev, mTrain, n, mValid, intercept, standardize, precision, a, b, c, d, e)
