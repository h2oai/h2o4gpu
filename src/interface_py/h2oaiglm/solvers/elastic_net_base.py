import sys
import numpy as np
from ctypes import *
from h2oaiglm.types import ORD, cptr, c_double_p, c_void_pp
from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU

class info:
    pass

class solution:
    pass

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
        self.uploadeddata=0
        self.didfitptr=0

    def upload_data(self, sourceDev, trainX, trainY, validX=None, validY=None, weight=None):
        finish1()
        self.uploadeddata=1
        if trainX is not None:
            try: 
                if trainX.value is not None:
                    mTrain = trainX.shape[0]
                else:
                    mTrain=0
            except:
                mTrain = trainX.shape[0]
        #
        if validX is not None:
            try: 
                if validX.value is not None:
                    mValid = validX.shape[0]
                else:
                    mValid=0
            except:
                mValid = validX.shape[0]
        #
        n = trainX.shape[1]
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if (trainX.dtype==np.float64):
            print("Detected np.float64");sys.stdout.flush()
            self.double_precision=1
            #
            A = cptr(trainX,dtype=c_double)
            B = cptr(trainY,dtype=c_double)
            null_ptr = POINTER(c_double)()
            if validX is not None:
                try:
                    if validX.value is not None:
                        C = cptr(validX,dtype=c_double)
                    else:
                        C = null_ptr
                except:
                    C = cptr(validX,dtype=c_double)
            else:
                C = null_ptr
            if validY is not None:
                try:
                    if validY.value is not None:
                        D = cptr(validY,dtype=c_double)
                    else:
                        D = null_ptr
                except:
                    D = cptr(validY,dtype=c_double)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight,dtype=c_double)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight,dtype=c_double)
            else:
                E = null_ptr
            status = self.lib.make_ptr_double(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        elif (trainX.dtype==np.float32):
            print("Detected np.float32");sys.stdout.flush()
            self.double_precision=0
            A = cptr(trainX,dtype=c_float)
            B = cptr(trainY,dtype=c_float)
            null_ptr = POINTER(c_float)()
            if validX is not None:
                try:
                    if validX.value is not None:
                        C = cptr(validX,dtype=c_float)
                    else:
                        C = null_ptr
                except:
                    C = cptr(validX,dtype=c_float)
            else:
                C = null_ptr
            if validY is not None:
                try:
                    if validY.value is not None:
                        D = cptr(validY,dtype=c_float)
                    else:
                        D = null_ptr
                except:
                    D = cptr(validY,dtype=c_float)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight,dtype=c_float)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight,dtype=c_float)
            else:
                E = null_ptr
            status = self.lib.make_ptr_float(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        else:
            print("Unknown numpy type detected")
            print(trainX.dtype)
            sys.stdout.flush()
            return a, b, c, d, e
            exit(1)
            
        assert status==0, "Failure uploading the data"
        #print(a)
        #print(b)
        #print(c)
        #print(d)
        #print(e)
        self.solution.double_precision=self.double_precision
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.e=e
        return a, b, c, d, e

    # sourceDev here because generally want to take in any pointer, not just from our test code
    def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath):
        finish2()
        self.didfitptr=1
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if hasattr(self, 'double_precision'):
            whichprecision=self.double_precision
        else:
            whichprecision=precision
        #
        Xvsalphalambda = c_void_p(0)
        Xvsalpha = c_void_p(0)
        countfull=c_size_t(0)
        countshort=c_size_t(0)
        countmore=c_size_t(0)
        c_size_t_p = POINTER(c_size_t)
        if (whichprecision==1):
            self.mydtype=np.double
            self.myctype=c_double
            print("double precision fit")
            self.lib.elastic_net_ptr_double(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,givefullpath
                ,pointer(Xvsalphalambda), pointer(Xvsalpha)
                ,cast(addressof(countfull),c_size_t_p), cast(addressof(countshort),c_size_t_p), cast(addressof(countmore),c_size_t_p)
            )
        else:
            self.mydtype=np.float
            self.myctype=c_float
            print("single precision fit")
            self.lib.elastic_net_ptr_float(
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,givefullpath
                ,pointer(Xvsalphalambda), pointer(Xvsalpha)
                ,cast(addressof(countfull),c_size_t_p), cast(addressof(countshort),c_size_t_p), cast(addressof(countmore),c_size_t_p)
            )
        #
        # save pointer
        self.Xvsalphalambda=Xvsalphalambda
        self.Xvsalpha=Xvsalpha
        #
        countfull_value=countfull.value
        countshort_value=countshort.value
        countmore_value=countmore.value
        #print("counts=%d %d %d" % (countfull_value,countshort_value,countmore_value))
        if givefullpath==1:
            numall=int(countfull_value/(self.n_alphas*self.n_lambdas))
        else:
            numall=int(countshort_value/(self.n_alphas))
        #
        NUMALLOTHER=numall-n
        NUMRMSE=3 # should be consistent with src/common/elastic_net_ptr.cpp
        NUMOTHER=NUMALLOTHER-NUMRMSE
        if NUMOTHER!=3:
            print("NUMOTHER=%d but expected 3" % (NUMOTHER))
            print("countfull_value=%d countshort_value=%d countmore_value=%d numall=%d NUMALLOTHER=%d" % (int(countfull_value), int(countshort_value), int(countmore_value), int(numall), int(NUMALLOTHER)))
            exit(0)
        #
        if givefullpath==1:
            # Xvsalphalambda contains solution (and other data) for all lambda and alpha
            self.Xvsalphalambdanew=np.fromiter(cast(Xvsalphalambda,POINTER(self.myctype)),dtype=self.mydtype,count=countfull_value)
            self.Xvsalphalambdanew=np.reshape(self.Xvsalphalambdanew,(self.n_lambdas,self.n_alphas,numall))
            self.Xvsalphalambdapure = self.Xvsalphalambdanew[:,:,0:n]
            self.rmsevsalphalambda = self.Xvsalphalambdanew[:,:,n:n+NUMRMSE]
            self.lambdas = self.Xvsalphalambdanew[:,:,n+NUMRMSE:n+NUMRMSE+1]
            self.alphas = self.Xvsalphalambdanew[:,:,n+NUMRMSE+1:n+NUMRMSE+2]
            self.tols = self.Xvsalphalambdanew[:,:,n+NUMRMSE+2:n+NUMRMSE+3]
            #
            self.solution.Xvsalphalambdapure  = self.Xvsalphalambdapure 
            self.info.rmsevsalphalambda  = self.rmsevsalphalambda 
            self.info.lambdas  = self.lambdas 
            self.info.alphas  = self.alphas 
            self.info.tols  = self.tols 
            #
        # Xvsalpha contains only best of all lambda for each alpha
        self.Xvsalpha=np.fromiter(cast(Xvsalpha,POINTER(self.myctype)),dtype=self.mydtype,count=countshort_value)
        self.Xvsalpha=np.reshape(self.Xvsalpha,(self.n_alphas,numall))
        self.Xvsalphapure = self.Xvsalpha[:,0:n]
        self.rmsevsalpha = self.Xvsalpha[:,n:n+NUMRMSE]
        self.lambdas2 = self.Xvsalpha[:,n+NUMRMSE:n+NUMRMSE+1]
        self.alphas2 = self.Xvsalpha[:,n+NUMRMSE+1:n+NUMRMSE+2]
        self.tols2 = self.Xvsalpha[:,n+NUMRMSE+2:n+NUMRMSE+3]
        #
        self.solution.Xvsalphapure  = self.Xvsalphapure 
        self.info.rmsevsalpha  = self.rmsevsalpha 
        self.info.lambdas2  = self.lambdas2 
        self.info.alphas2  = self.alphas2 
        self.info.tols2  = self.tols2 
        #
        # return numpy objects
        if givefullpath==1:
            return(self.Xvsalphalambdapure,self.Xvsalphapure)
        else:
            return(self.Xvsalphapure)
        #return(self.Xvsalphalambdanew,self.Xvsalpha)
        print("Done with fit")

    def fit(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0):
        #
        self.givefullpath=givefullpath
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
        if( (validX and validY==None) or  (validX==None and validY) ):
                print("Must input both validX and validY or neither.")
                docalc=0
            #
        if(docalc):
            sourceDev=0 # assume GPU=0 is fine as source
            a,b,c,d,e = self.upload_data(sourceDev, trainX, trainY, validX, validY, weight)
            precision=0 # won't be used
            self.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath)
            if givefullpath==1:
                return(self.Xvsalphalambdapure,self.Xvsalphapure)
            else:
                return(self.Xvsalphapure)
        else:
            # return NULL pointers
            if givefullpath==1:
                return(c_void_p(0),c_void_p(0))
            else:
                return(c_void_p(0))
        self.trainX=trainX
        self.trainY=trainY
        self.validX=validX
        self.validY=validY
        self.weight=weight
    def getrmse(self):
        if self.givefullpath:
            return(self.rmsevsalphalambda)
        else:
            return(self.rmsevsalpha)
    def getlambdas(self):
        if self.givefullpath:
            return(self.lambdas)
        else:
            return(self.lambdas2)
    def getalphas(self):
        if self.givefullpath:
            return(self.alphas)
        else:
            return(self.alphas2)
    def gettols(self):
        if self.givefullpath:
            return(self.tols)
        else:
            return(self.tols2)
    def predict(self,testX, testweight=None, givefullpath=0):
        # if pass None trainY, then do predict using testX and weight (if given)
        self.prediction=fit(testX, None, None, None, testweight, givefullpath)
        return(self.prediction) # something like testY
    def fit_predict(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0):
        fit(trainX, trainY, validX, validY, weight, givefullpath)
        if validX==None:
            self.prediction=predict(trainX)
        else:
            self.prediction=predict(validX)
        return(self.predictions)
    def finish1(self):
        if self.uploadeddata==1:
            modelfree(self.a)
            modelfree(self.b)
            modelfree(self.c)
            modelfree(self.d)
            modelfree(self.e)
    def finish2(self):
        if self.didfitptr==1:
            modelfree(self.Xvsalphalambda)
            modelfree(self.Xvsalpha)
    def finish(self):
        finish1()
        finish2()
        

