import sys
import numpy as np
from ctypes import *
from h2oaiglm.types import ORD, cptr, c_double_p, c_void_pp
from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU

class ElasticNetBaseSolver(object):
    def __init__(self, lib, sharedA, nThreads, nGPUs, ordin, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas):
        assert lib and (lib==h2oaiglmElasticNetCPU or lib==h2oaiglmElasticNetGPU)
        self.lib=lib
        self.nGPUs=nGPUs
        self.sourceDev=0 # assume Dev=0 is source of data for upload_data
        self.sourceme=0 # assume thread=0 is source of data for upload_data
        self.sharedA=sharedA
        self.nThreads=nThreads
        self.ord=ord(ordin)
        self.intercept=intercept
        self.standardize=standardize
        self.lambda_min_ratio=lambda_min_ratio
        self.n_lambdas=n_lambdas
        self.n_folds=n_folds
        self.n_alphas=n_alphas

    def upload_data(self, sourceDev, trainX, trainY, validX, validY, weight):
        mTrain = trainX.shape[0]
        mValid = validX.shape[0]
        n = trainX.shape[1]
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if (trainX.dtype==np.float64):
            print("Detected np.float64");sys.stdout.flush()
            self.double_precision=1
            A = cptr(trainX,dtype=c_double)
            B = cptr(trainY,dtype=c_double)
            C = cptr(validX,dtype=c_double)
            D = cptr(validY,dtype=c_double)
            E = cptr(weight,dtype=c_double)
            status = self.lib.make_ptr_double(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        elif (trainX.dtype==np.float32):
            print("Detected np.float32");sys.stdout.flush()
            self.double_precision=0
            A = cptr(trainX,dtype=c_float)
            B = cptr(trainY,dtype=c_float)
            C = cptr(validX,dtype=c_float)
            D = cptr(validY,dtype=c_float)
            E = cptr(weight,dtype=c_float)
            status = self.lib.make_ptr_float(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        else:
            print("Unknown numpy type detected")
            print(trainX.dtype)
            sys.stdout.flush()
            exit(1)
            
        assert status==0, "Failure uploading the data"
        #print(a)
        #print(b)
        #print(c)
        #print(d)
        #print(e)
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        return a, b, c, d, e

    # sourceDev here because generally want to take in any pointer, not just from our test code
    def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath):
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if hasattr(self, 'double_precision'):
            whichprecision=self.double_precision
        else:
            whichprecision=precision
        #
        if (whichprecision==1):
            print("double precision fit")
            self.lib.elastic_net_ptr_double(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,givefullpath
                ,Xvsalphalambda, Xvsalpha
            )
        else:
            print("single precision fit")
            self.lib.elastic_net_ptr_float(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,self.givefullpath
                ,Xvsalphalambda, Xvsalpha
            )
        self.Xvsalphalambda=Xvsalphalambda
        self.Xvsalpha=Xvsalpha
        return(Xvsalphalambda,Xvsalpha)
        print("Done with fit")

    def fit(self, trainX, trainY, validX=c_void_p(0), validY=c_void_p(0), weight=c_void_p(0), givefullpath=0):
        #
        # get shapes
        shapeX=np.shape(trainX)
        mTrain=shapeX[0]
        n=shapeX[1]
        #
        shapeY=np.shape(trainY)
        mY=shapeY[0]
        if(mTrain!=mY):
            print("training X and Y must have same number of rows, but mTrain=%d mY=%d\n" % (mTrain,mY))
        #
        if(validX):
            shapevalidX=np.shape(validX)
            mValid=shapevalidX[0]
            nvalidX=shapevalidX[1]
            if(n!=nvalidX):
                print("trainX and validX must have same number of columns, but n=%d nvalidX=%d\n" % (n,nvalidX))
        else:
            mValid=0
        #
        if(validY):
            shapevalidY=np.shape(validY)
            mvalidY=shapevalidY[0]
            if(mValid!=mvalidY):
                print("validX and validY must have same number of rows, but mValid=%d mvalidY=%d\n" % (mValid,mvalidY))

        docalc=1
        if( (validX and !validY) or  (!validX and validY) ):
                print("Must input both validX and validY or neither.")
                docalc=0
            #
        if(docalc):
            sourceDev=0 # assume GPU=0 is fine as source
            a,b,c,d = upload_data(sourceDev, trainX, trainY, validX, validY, weight)
            precision=0 # won't be used
            Xvsalphalambda,Xvsalpha = fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath)
            return(Xvsalphalambda,Xvsalpha)
        else:
            # return NULL pointers
            return(c_void_p(0),c_void_p(0))
    
