import numpy as np
from ctypes import *
from pogs.types import ORD, cptr, c_double_p, c_void_pp
from pogs.libs.elastic_net_cpu import pogsElasticNetCPU
from pogs.libs.elastic_net_gpu import pogsElasticNetGPU

class ElasticNetBaseSolver(object):
    def __init__(self, lib, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas, double_precision=False):
        assert lib and (lib==pogsElasticNetCPU or lib==pogsElasticNetGPU)
        self.lib=lib
        self.nGPUs=nGPUs
        self.sourceDev=0 # assume Dev=0 is source of data for upload_data
        self.sourceme=0 # assume thread=0 is source of data for upload_data
        self.sharedA=sharedA
        self.nThreads=nThreads
        self.ord=1 if ord=='r' else 0
        self.intercept=intercept
        self.standardize=standardize
        self.lambda_min_ratio=lambda_min_ratio
        self.n_lambdas=n_lambdas
        self.n_alphas=n_alphas
        self.double_precision = double_precision

    def upload_data(self, sourceDev, trainX, trainY, validX, validY):
        mTrain = trainX.shape[0]
        mValid = validX.shape[0]
        n = trainX.shape[1]
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        if (self.double_precision):
            print("converting numpy array to double precision data structures")
            A = cptr(np.array(trainX, dtype='float64', order='C'),c_double)
            B = cptr(np.array(trainY, dtype='float64', order='C'),c_double)
            C = cptr(np.array(validX, dtype='float64', order='C'),c_double)
            D = cptr(np.array(validY, dtype='float64', order='C'),c_double)
            status = self.lib.make_ptr_double(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                              A, B, C, D, pointer(a), pointer(b), pointer(c), pointer(d))
        else:
            print("converting numpy array to single precision data structures")
            A = cptr(np.array(trainX, dtype='float32', order='C'),c_float)
            B = cptr(np.array(trainY, dtype='float32', order='C'),c_float)
            C = cptr(np.array(validX, dtype='float32', order='C'),c_float)
            D = cptr(np.array(validY, dtype='float32', order='C'),c_float)
            status = self.lib.make_ptr_float(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                              A, B, C, D, pointer(a), pointer(b), pointer(c), pointer(d))
        assert status==0, "Failure uploading the data"
        print(a)
        print(b)
        print(c)
        print(d)
        return a, b, c, d

    # sourceDev here because generally want to take in any pointer, not just from our test code
    def fit(self, sourceDev, mTrain, n, mValid, intercept, standardize, lambda_max0, sdTrainY, meanTrainY, sdValidY, meanValidY, a, b, c, d):
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if self.double_precision:
            print("double precision fit")
            self.lib.elastic_net_ptr_double(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize), c_double(lambda_max0),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_alphas),
                c_double(sdTrainY), c_double(meanTrainY),
                c_double(sdValidY), c_double(meanValidY),
                a, b, c, d)
        else:
            print("single precision fit")
            self.lib.elastic_net_ptr_float(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize), c_double(lambda_max0),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_alphas),
                c_float(sdTrainY), c_float(meanTrainY),
                c_float(sdValidY), c_float(meanValidY),
                a, b, c, d)
