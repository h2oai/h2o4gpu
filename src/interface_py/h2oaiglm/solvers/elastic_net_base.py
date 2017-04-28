import sys
import numpy as np
from ctypes import *
from h2oaiglm.types import ORD, cptr, c_double_p, c_void_pp
from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU

class ElasticNetBaseSolver(object):
    def __init__(self, lib, sharedA, nThreads, nGPUs, ord, intercept, standardize, lambda_min_ratio, n_lambdas, n_alphas):
        assert lib and (lib==h2oaiglmElasticNetCPU or lib==h2oaiglmElasticNetGPU)
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

    def upload_data(self, sourceDev, trainX, trainY, validX, validY):
        mTrain = trainX.shape[0]
        mValid = validX.shape[0]
        n = trainX.shape[1]
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        if (trainX.dtype==np.float64):
            print("Detected np.float64");sys.stdout.flush()
            self.double_precision=1
            A = cptr(trainX,dtype=c_double)
            B = cptr(trainY,dtype=c_double)
            C = cptr(validX,dtype=c_double)
            D = cptr(validY,dtype=c_double)
            status = self.lib.make_ptr_double(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                              A, B, C, D, pointer(a), pointer(b), pointer(c), pointer(d))
        elif (trainX.dtype==np.float32):
            print("Detected np.float32");sys.stdout.flush()
            self.double_precision=0
            A = cptr(trainX,dtype=c_float)
            B = cptr(trainY,dtype=c_float)
            C = cptr(validX,dtype=c_float)
            D = cptr(validY,dtype=c_float)
            status = self.lib.make_ptr_float(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid),
                                              A, B, C, D, pointer(a), pointer(b), pointer(c), pointer(d))
        else:
            print("Unknown numpy type detected")
            print(trainX.dtype)
            sys.stdout.flush()
            exit(1)
            
        assert status==0, "Failure uploading the data"
        print(a)
        print(b)
        print(c)
        print(d)
        return a, b, c, d

    # sourceDev here because generally want to take in any pointer, not just from our test code
    def fit(self, sourceDev, mTrain, n, mValid, intercept, standardize, a, b, c, d):
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if (self.double_precision==1):
            print("double precision fit")
            self.lib.elastic_net_ptr_double(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_alphas),
                a, b, c, d)
        else:
            print("single precision fit")
            self.lib.elastic_net_ptr_float(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_alphas),
                a, b, c, d)
