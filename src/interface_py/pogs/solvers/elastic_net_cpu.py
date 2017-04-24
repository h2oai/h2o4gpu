from pogs.libs.elastic_net_cpu import pogsElasticNetCPU
from pogs.solvers.elastic_net_base import ElasticNetBaseSolver

if not pogsElasticNetCPU:
	print('\nWarning: Cannot create a POGS Elastic Net CPU Solver instance without linking Python module to a compiled POGS CPU library')
	print('> Setting pogs.ElasticNetSolverCPU=None')
	print('> Use pogs.ElasticNetSolverGPU(args...) and re-run setup.py\n\n')
	ElasticNetSolverCPU=None
else:
	class ElasticNetSolverCPU(object):
		def __init__(self, sharedA, nThreads, nCPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas, double_precision):
			self.solver = ElasticNetBaseSolver(pogsElasticNetCPU, sharedA, nThreads, nCPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas, double_precision)

		def upload_data(self, sourceDev, trainX, trainY, validX, validY):
			return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY)

		def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d):
			return self.solver.fit(sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d)
