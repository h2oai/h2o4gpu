from pogs.libs.elastic_net_gpu import pogsElasticNetGPU
from pogs.solvers.elastic_net_base import ElasticNetBaseSolver

if not pogsElasticNetGPU:
	print('\nWarning: Cannot create a POGS Elastic Net GPU Solver instance without linking Python module to a compiled POGS GPU library')
	print('> Setting pogs.ElasticNetSolverGPU=None')
	print('> Use pogs.ElasticNetSolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n')
	ElasticNetSolverGPU=None
else:
	class ElasticNetSolverGPU(object):
		def __init__(self, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas):
			self.solver = ElasticNetBaseSolver(pogsElasticNetGPU, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas)

		def upload_data(self, sourceDev, trainX, trainY, validX, validY):
			return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY)

		def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d):
			return self.solver.fit(sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d)
