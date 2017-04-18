from pogs.libs.elastic_net_cpu import pogsElasticNetCPU
print("pogsElasticNetCPU!=None: " + str(pogsElasticNetCPU!=None))
from pogs.solvers.elastic_net_base import ElasticNetBaseSolver
print("ElasticNetBaseSolver!=None: " + str(ElasticNetBaseSolver!=None))
from pogs.types import cptr, c_double

if not pogsElasticNetCPU:
	print('\nWarning: Cannot create a POGS Elastic Net CPU Solver instance without linking Python module to a compiled POGS CPU library')
	print('> Setting pogs.ElasticNetSolverCPU=None')
	print('> Use pogs.ElasticNetSolverGPU(args...) and re-run setup.py\n\n')
	ElasticNetSolverCPU=None
else:
	class ElasticNetSolverCPU(object):
		def __init__(self, nCPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas):
                        self.solver = ElasticNetBaseSolver(pogsElasticNetCPU,
                                                           nCPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas)

		def upload_data(self, sourceDev, trainX, trainY, validX, validY):
			#TODO: double/float
			return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY)

		def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, a, b, c, d):
			## C++ CALL
			self.solver.ElasticNetptr(sourceDev, 1, self.nCPUs, self.ord, mTrain, n, mValid,
						  self.intercept, lambda_max0, self.lambda_min_ratio,
						  self.n_lambdas, self.n_alphas,
						  sdTrainY, meanTrainY, a, b, c, d)
			return self
