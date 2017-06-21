import sys
import numpy as np
from ctypes import *
from h2oaiglm.types import ORD, cptr, c_double_p, c_void_pp
from h2oaiglm.libs.elastic_net_cpu import h2oaiglmElasticNetCPU
from h2oaiglm.libs.elastic_net_gpu import h2oaiglmElasticNetGPU


class ElasticNetBaseSolver(object):
    class info:
        pass

    class solution:
        pass
    
    def __init__(self, lib, sharedA, nThreads, nGPUs, ordin, intercept, standardize, lambda_min_ratio, n_lambdas, n_folds, n_alphas):
        assert lib and (lib==h2oaiglmElasticNetCPU or lib==h2oaiglmElasticNetGPU)
        self.lib=lib

        self.n=0
        self.mTrain=0
        self.mValid=0
        
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
        self.didpredict=0

    def upload_data(self, sourceDev, trainX, trainY, validX=None, validY=None, weight=None):
        self.finish1()
        self.uploadeddata=1
        if trainX is not None:
            try: 
                if trainX.value is not None:
                    mTrain = trainX.shape[0]
                    n = trainX.shape[1]
                else:
                    mTrain=0
            except:
                mTrain = trainX.shape[0]
                n = trainX.shape[1]
            self.mTrain=mTrain
            self.n=n
        #
        if validX is not None:
            try: 
                if validX.value is not None:
                    mValid = validX.shape[0]
                    n = validX.shape[1]
                else:
                    mValid=0
            except:
                mValid = validX.shape[0]
                n = validX.shape[1]
            self.mValid=mValid
            self.n=n # should be same as trainX when wasn't doing prediction
        #
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if (trainX.dtype==np.float64):
            print("Detected np.float64");sys.stdout.flush()
            self.double_precision=1
            null_ptr = POINTER(c_double)()
            #
            if trainX is not None:
                try:
                    if trainX.value is not None:
                        A = cptr(trainX,dtype=c_double)
                    else:
                        A = null_ptr
                except:
                    A = cptr(trainX,dtype=c_double)
            else:
                A = null_ptr
            if trainY is not None:
                try:
                    if trainY.value is not None:
                        B = cptr(trainY,dtype=c_double)
                    else:
                        B = null_ptr
                except:
                    B = cptr(trainY,dtype=c_double)
            else:
                B = null_ptr
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
            null_ptr = POINTER(c_float)()
            #
            if trainX is not None:
                try:
                    if trainX.value is not None:
                        A = cptr(trainX,dtype=c_float)
                    else:
                        A = null_ptr
                except:
                    A = cptr(trainX,dtype=c_float)
            else:
                A = null_ptr
            if trainY is not None:
                try:
                    if trainY.value is not None:
                        B = cptr(trainY,dtype=c_float)
                    else:
                        B = null_ptr
                except:
                    B = cptr(trainY,dtype=c_float)
            else:
                B = null_ptr
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
    def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict):
        if dopredict==0:
            self.finish2()
            # otherwise don't clear solution, just use it
        #
        self.didfitptr=1
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if hasattr(self, 'double_precision'):
            whichprecision=self.double_precision
        else:
            whichprecision=precision
            self.double_precision=precision
        #
        Xvsalphalambda = c_void_p(0)
        Xvsalpha = c_void_p(0)
        validPredsvsalphalambda = c_void_p(0)
        validPredsvsalpha = c_void_p(0)
        countfull=c_size_t(0)
        countshort=c_size_t(0)
        countmore=c_size_t(0)
        c_size_t_p = POINTER(c_size_t)
        if (whichprecision==1):
            self.mydtype=np.double
            self.myctype=c_double
            print("double precision fit")
            self.lib.elastic_net_ptr_double(
                c_int(dopredict),
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,givefullpath
                ,pointer(Xvsalphalambda), pointer(Xvsalpha)
                ,pointer(validPredsvsalphalambda), pointer(validPredsvsalpha)
                ,cast(addressof(countfull),c_size_t_p), cast(addressof(countshort),c_size_t_p), cast(addressof(countmore),c_size_t_p)
            )
        else:
            self.mydtype=np.float
            self.myctype=c_float
            print("single precision fit")
            self.lib.elastic_net_ptr_float(
                c_int(dopredict),
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                ,givefullpath
                ,pointer(Xvsalphalambda), pointer(Xvsalpha)
                ,pointer(validPredsvsalphalambda), pointer(validPredsvsalpha)
                ,cast(addressof(countfull),c_size_t_p), cast(addressof(countshort),c_size_t_p), cast(addressof(countmore),c_size_t_p)
            )
        #
        # save pointer
        self.Xvsalphalambda=Xvsalphalambda
        self.Xvsalpha=Xvsalpha
        self.validPredsvsalphalambda=validPredsvsalphalambda
        self.validPredsvsalpha=validPredsvsalpha
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
            if dopredict==1:
                self.validPredsvsalphalambdanew=np.fromiter(cast(validPredsvsalphalambda,POINTER(self.myctype)),dtype=self.mydtype,count=countfull_value/(n+NUMALLOTHER)*mValid)
                self.validPredsvsalphalambdanew=np.reshape(self.validPredsvsalphalambdanew,(self.n_lambdas,self.n_alphas,mValid))
                self.validPredsvsalphalambdapure = self.validPredsvsalphalambda[:,:,0:mValid]
            #
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
        if givefullpath==0 and dopredict==1: # exclusive set of validPreds unlike X
            self.validPredsvsalphanew=np.fromiter(cast(validPredsvsalpha,POINTER(self.myctype)),dtype=self.mydtype,count=countfull_value/(n+NUMALLOTHER)*mValid)
            self.validPredsvsalphanew=np.reshape(self.validPredsvsalphanew,(self.n_s,self.n_alphas,mValid))
            self.validPredsvsalphapure = self.validPredsvsalpha[:,:,0:mValid]
        #
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
        if dopredict==0:
            self.didpredict=0
            if givefullpath==1:
                return(self.Xvsalphalambdapure,self.Xvsalphapure)
            else:
                return(self.Xvsalphapure)
            print("Done with fit")
        else:
            self.didpredict=1
            if givefullpath==1:
                return(self.validPredsvsalphalambdapure,self.validPredsvsalphapure)
            else:
                return(self.validPredsvsalphapure)
            print("Done with predict")


        
    def fit(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0, dopredict=0):
        #
        self.givefullpath=givefullpath
        #
        if trainX is not None:
            try:
                if trainX.value is not None:
                    # get shapes
                    shapeX=np.shape(trainX)
                    mTrain=shapeX[0]
                    n=shapeX[1]
                else:
                    print("no trainX")
            except:
                # get shapes
                shapeX=np.shape(trainX)
                mTrain=shapeX[0]
                n=shapeX[1]
        else:
            print("no trainX")
        #
        dopredict=0
        if trainY is not None:
            try:
                if trainY.value is not None:
                    # get shapes
                    print("Doing fit")
                    shapeY=np.shape(trainY)
                    mY=shapeY[0]
                    if(mTrain!=mY):
                        print("training X and Y must have same number of rows, but mTrain=%d mY=%d\n" % (mTrain,mY))
                else:
                    print("Doing predict")
                    dopredict=1
            except:
                # get shapes
                print("Doing fit")
                shapeY=np.shape(trainY)
                mY=shapeY[0]
                if(mTrain!=mY):
                    print("training X and Y must have same number of rows, but mTrain=%d mY=%d\n" % (mTrain,mY))
        else:
            print("Doing predict")
            dopredict=1

        #
        if validX is not None:
            try:
                if validX.value is not None:
                    shapevalidX=np.shape(validX)
                    mValid=shapevalidX[0]
                    nvalidX=shapevalidX[1]
                    if dopredict==0:
                        if(n!=nvalidX):
                            print("trainX and validX must have same number of columns, but n=%d nvalidX=%d\n" % (n,nvalidX))
                else:
                    print("no validX")
                    mValid=0
            except:
                shapevalidX=np.shape(validX)
                mValid=shapevalidX[0]
                nvalidX=shapevalidX[1]
                if dopredict==0:
                    if(n!=nvalidX):
                        print("trainX and validX must have same number of columns, but n=%d nvalidX=%d\n" % (n,nvalidX))
        else:
            print("no validX")
            mValid=0
        #
        #
        if validY is not None:
            try:
                if validY.value is not None:
                    shapevalidY=np.shape(validY)
                    mvalidY=shapevalidY[0]
                    if dopredict==0:
                        if(mValid!=mvalidY):
                            print("validX and validY must have same number of rows, but mValid=%d mvalidY=%d\n" % (mValid,mvalidY))
                else:
                    print("no validY")
            except:
                shapevalidY=np.shape(validY)
                mvalidY=shapevalidY[0]
                if dopredict==0:
                    if(mValid!=mvalidY):
                        print("validX and validY must have same number of rows, but mValid=%d mvalidY=%d\n" % (mValid,mvalidY))
        else:
            print("no validY")
        #
        #
        docalc=1
        if( (validX and validY==None) or  (validX==None and validY) ):
                print("Must input both validX and validY or neither.")
                docalc=0
            #
        if(docalc):
            sourceDev=0 # assume GPU=0 is fine as source
            a,b,c,d,e = self.upload_data(sourceDev, trainX, trainY, validX, validY, weight)
            precision=0 # won't be used
            self.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict=dopredict)
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
    def predict(self, validX, testweight=None, givefullpath=0):
        # if pass None trainx and trainY, then do predict using validX and weight (if given)
        # unlike upload_data and fitptr (and so fit) don't free-up predictions since for single model might request multiple predictions.  User has to call finish themselves to cleanup.
        dopredict=1
        self.prediction=self.fit(None, None, validX, None, testweight, givefullpath,dopredict)
        return(self.prediction) # something like validY
    def fit_predict(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0):
        self.fit(trainX, trainY, validX, validY, weight, givefullpath)
        if validX==None:
            self.prediction=self.predict(trainX, testweight=weight, givefullpath=givefullpath)
        else:
            self.prediction=self.predict(validX, testweight=weight, givefullpath=givefullpath)
        return(self.predictions)
    def finish1(self):
        if self.uploadeddata==1:
            self.uploadeddata=0
            if self.double_precision==1:
                self.lib.modelfree1_double(self.a)
                self.lib.modelfree1_double(self.b)
                self.lib.modelfree1_double(self.c)
                self.lib.modelfree1_double(self.d)
                self.lib.modelfree1_double(self.e)
            else:
                self.lib.modelfree1_float(self.a)
                self.lib.modelfree1_float(self.b)
                self.lib.modelfree1_float(self.c)
                self.lib.modelfree1_float(self.d)
                self.lib.modelfree1_float(self.e)
    def finish2(self):
        if self.didfitptr==1:
            self.didfitptr=0
            if self.double_precision==1:
                self.lib.modelfree2_double(self.Xvsalphalambda)
                self.lib.modelfree2_double(self.Xvsalpha)
            else:
                self.lib.modelfree2_float(self.Xvsalphalambda)
                self.lib.modelfree2_float(self.Xvsalpha)                
    def finish3(self):
        if self.didpredict==1:
            self.didpredict=0
            if self.double_precision==1:
                self.lib.modelfree2_double(self.validPredsvsalphalambda)
                self.lib.modelfree2_double(self.validPredsvsalpha)
            else:
                self.lib.modelfree2_float(self.validPredsvsalphalambda)
                self.lib.modelfree2_float(self.validPredsvsalpha)                
    def finish(self):
        self.finish1()
        self.finish2()
        self.finish3()
        

