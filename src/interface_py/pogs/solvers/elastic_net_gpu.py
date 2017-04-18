from pogs.libs.elastic_net_gpu import pogsElasticNetGPU
print("pogsElasticNetGPU!=None: " + str(pogsElasticNetGPU!=None))
from pogs.solvers.elastic_net_base import ElasticNetBaseSolver
print("ElasticNetBaseSolver!=None: " + str(ElasticNetBaseSolver!=None))

if not pogsElasticNetGPU:
	print('\nWarning: Cannot create a POGS Elastic Net GPU Solver instance without linking Python module to a compiled POGS GPU library')
	print('> Setting pogs.ElasticNetSolverGPU=None')
	print('> Use pogs.ElasticNetSolverCPU(args...) or add CUDA libraries to $PATH and re-run setup.py\n\n')
	ElasticNetSolverGPU=None
else:
	class ElasticNetSolverGPU(object):
		def __init__(self, nGPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas):
			self.solver = ElasticNetBaseSolver(pogsElasticNetGPU,
											   nGPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas)

		def upload_data(gpu, sourceDev, trainX, trainY, validX, validY):
			pogsElasticNetGPU.upload_data(gpu, sourceDev, trainX, trainY, validX, validY)

		def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, a, b, c, d):
			## C++ CALL
			self.solver.ElasticNetptr(sourceDev, 1, self.nGPUs, self.ord, mTrain, n, mValid,
						  self.intercept, lambda_max0, self.lambda_min_ratio,
						  self.n_lambdas, self.n_alphas,
						  sdTrainY, meanTrainY, a, b, c, d)
			return self
