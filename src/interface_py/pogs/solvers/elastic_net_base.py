from ctypes import *
from pogs.types import ORD, cptr, c_double_p, c_void_pp
from pogs.libs.elastic_net_cpu import pogsElasticNetCPU
from pogs.libs.elastic_net_gpu import pogsElasticNetGPU

class ElasticNetBaseSolver(object):
    def __init__(self, lib, nGPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas):
        assert lib and (lib==pogsElasticNetCPU or lib==pogsElasticNetGPU)
        self.lib=lib
        self.nGPUs=nGPUs
        self.ord=1 if ord=='r' else 0
        self.intercept=intercept
        self.lambda_min_ratio=lambda_min_ratio
        self.n_lambdas=n_lambdas
        self.n_alphas=n_alphas

    def upload_data(self, sourceDev, trainX, trainY, validX, validY):
        mTrain = trainX.shape[0]
        mValid = validX.shape[0]
        n = validX.shape[1]
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        A = cptr(trainX,c_double)
        B = cptr(trainY,c_double)
        C = cptr(validX,c_double)
        D = cptr(validY,c_double)
        ## C++ CALL
        # ##TODO: float
        status = self.lib.make_ptr_double(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                          A, B, C, D, pointer(a), pointer(b), pointer(c), pointer(d))
        assert status==0, "Failure uploading the data"
        print(a)
        print(b)
        print(c)
        print(d)
        return a, b, c, d

    def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, a, b, c, d):
        ## C++ CALL
        # ##TODO: float
        self.lib.elastic_net_ptr_double(
            c_int(sourceDev), c_int(1), c_int(self.nGPUs),
            c_int(self.ord), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
            c_int(self.intercept), c_double(lambda_max0),
            c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_alphas),
            c_double(sdTrainY), c_double(meanTrainY),
            a, b, c, d)
