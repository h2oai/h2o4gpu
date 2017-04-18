from ctypes import *
from pogs.types import ORD, cptr, c_double_p, c_void_pp
from pogs.libs.elastic_net_cpu import pogsElasticNetCPU
from pogs.libs.elastic_net_gpu import pogsElasticNetGPU

class ElasticNetBaseSolver(object):
    def __init__(self, lib, nGPUs, ord, intercept, lambda_min_ratio, n_lambdas, n_alphas):
        print("Trying to instantiate a elastic net base solver")
        assert lib and (lib==pogsElasticNetCPU or lib==pogsElasticNetGPU)
        self.lib=lib
        self.nGPUs=nGPUs
        self.ord=ord
        self.intercept=intercept
        self.lambda_min_ratio=lambda_min_ratio
        self.n_lambdas=n_lambdas
        self.n_alphas=n_alphas

    def upload_data(self, sourceDev, trainX, trainY, validX, validY):
        mTrain = trainX.shape[0]
        mValid = validX.shape[0]
        n = validX.shape[1]
        aa = c_void_p(0)
        bb = c_void_p(0)
        cc = c_void_p(0)
        dd = c_void_p(0)
        a=pointer(aa)
        b=pointer(bb)
        c=pointer(cc)
        d=pointer(dd)
        A = cptr(trainX,c_double)
        B = cptr(trainY,c_double)
        C = cptr(validX,c_double)
        D = cptr(validY,c_double)
        ## C++ CALL
        status = self.lib.make_ptr_double(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                          A, B, C, D, ## input
                                          a, b, c, d) ## output
        assert status==0, "Failure uploading the data"
        print(aa)
        print(bb)
        print(cc)
        print(dd)
        return aa, bb, cc, dd

    def fit(self, sourceDev, mTrain, n, mValid, lambda_max0, sdTrainY, meanTrainY, a, b, c, d):
        print("Implement!")
        # ## C++ CALL
        # self.solver.ElasticNetptr(sourceDev, 1, self.nGPUs, self.ord, mTrain, n, mValid,
        #                           self.intercept, lambda_max0, self.lambda_min_ratio,
        #                           self.n_lambdas, self.n_alphas,
        #                           sdTrainY, meanTrainY, a, b, c, d)
        # self.lib.ElasticNetptr(self.work, pointer(self.settings), pointer(self.solution), pointer(self.info),
        #                            cptr(f.a,c_double), cptr(f.b,c_double), cptr(f.c,c_double),
        #                            cptr(f.d,c_double), cptr(f.e,c_double), cptr(f.h,c_int),
        #                            cptr(g.a,c_double), cptr(g.b,c_double), cptr(g.c,c_double),
        #                            cptr(g.d,c_double), cptr(g.e,c_double), cptr(g.h,c_int))