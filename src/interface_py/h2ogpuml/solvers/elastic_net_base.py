from ctypes import *
from h2ogpuml.types import ORD, cptr
import numpy as np
import sys
from h2ogpuml.types import ORD, cptr, c_double_p, c_void_pp
from h2ogpuml.libs.elastic_net_cpu import h2ogpumlGLMCPU
from h2ogpuml.libs.elastic_net_gpu import h2ogpumlGLMGPU
from h2ogpuml.solvers.utils import devicecount


class GLM(object):
    def __init__(self, sharedA=0, nThreads=None, n_gpus=-1, ord='r', intercept=1, standardize=0, lambda_min_ratio=1E-7,
                 n_lambdas=100, n_folds=1,
                 n_alphas=1):

        n_gpus, deviceCount = devicecount(n_gpus)

        if nThreads == None:
            # not required number of threads, but normal.  Bit more optimal to use 2 threads for CPU, but 1 thread per GPU is optimal.
            nThreads = 1 if (n_gpus == 0) else n_gpus

        if not h2ogpumlGLMGPU:
            print(
                '\nWarning: Cannot create a H2OGPUML Elastic Net GPU Solver instance without linking Python module to a compiled H2OGPUML GPU library')
            print('> Use CPU or add CUDA libraries to $PATH and re-run setup.py\n\n')

        if not h2ogpumlGLMCPU:
            print(
                '\nWarning: Cannot create a H2OGPUML Elastic Net CPU Solver instance without linking Python module to a compiled H2OGPUML CPU library')
            print('> Use GPU or re-run setup.py\n\n')

        if ((n_gpus == 0) or (h2ogpumlGLMGPU is None) or (deviceCount == 0)):
            print("\nUsing CPU GLM solver %d %d\n" % (n_gpus, deviceCount))
            self.solver = GLMBaseSolver(h2ogpumlGLMCPU, sharedA, nThreads, n_gpus, ord, intercept, standardize,
                                        lambda_min_ratio, n_lambdas, n_folds, n_alphas)
        else:
            if ((n_gpus > 0) or (h2ogpumlGLMGPU is None) or (deviceCount == 0)):
                print("\nUsing GPU GLM solver with %d GPUs\n" % n_gpus)
                self.solver = GLMBaseSolver(h2ogpumlGLMGPU, sharedA, nThreads, n_gpus, ord, intercept, standardize,
                                            lambda_min_ratio, n_lambdas, n_folds, n_alphas)

        assert self.solver != None, "Couldn't instantiate GLM Solver"

    def upload_data(self, sourceDev, trainX, trainY, validX, validY, weight):
        return self.solver.upload_data(sourceDev, trainX, trainY, validX, validY, weight)

    def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c=c_void_p(0), d=c_void_p(0), e=c_void_p(0),
               givefullpath=0, dopredict=0, freeinputdata=0):
        return self.solver.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict, freeinputdata)

    def fit(self, trainX, trainY, validX=c_void_p(0), validY=c_void_p(0), weight=c_void_p(0), givefullpath=0,
            dopredict=0, freeinputdata=1):
        return self.solver.fit(trainX, trainY, validX, validY, weight, givefullpath, dopredict, freeinputdata)

    def getrmse(self):
        return self.solver.getrmse()

    def getlambdas(self):
        return self.solver.getlambdas()

    def getalphas(self):
        return self.solver.getalphas()

    def gettols(self):
        return self.solver.gettols()

    def predict(self, validX, validY=None, testweight=None, givefullpath=0):
        return self.solver.predict(validX, validY, testweight, givefullpath)

    def predictptr(self, validXptr, validYptr=None, givefullpath=0):
        return self.solver.predictptr(validXptr, validYptr, givefullpath)

    def fit_predict(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0, freeinputdata=0):
        return self.solver.fit_predict(trainX, trainY, validX, validY, weight, givefullpath, freeinputdata)

    def fit_predictptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath=0, freeinputdata=0):
        return self.solver.fit_predictptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, freeinputdata)

    def freedata(self):
        return self.solver.freedata()

    def freesols(self):
        return self.solver.freesols()

    def freepreds(self):
        return self.solver.freepreds()

    def finish(self):
        return self.solver.finish()


class GLMBaseSolver(object):
    class info:
        pass

    class solution:
        pass

    def __init__(self, lib, sharedA, nThreads, nGPUs, ordin, intercept, standardize, lambda_min_ratio, n_lambdas,
                 n_folds, n_alphas):
        assert lib and (lib == h2ogpumlGLMCPU or lib == h2ogpumlGLMGPU)
        self.lib = lib

        self.n = 0
        self.mTrain = 0
        self.mValid = 0

        self.nGPUs = nGPUs
        self.sourceDev = 0  # assume Dev=0 is source of data for upload_data
        self.sourceme = 0  # assume thread=0 is source of data for upload_data
        self.sharedA = sharedA
        self.nThreads = nThreads
        self.ord = ord(ordin)
        self.intercept = intercept
        self.standardize = standardize
        self.lambda_min_ratio = lambda_min_ratio
        self.n_lambdas = n_lambdas
        self.n_folds = n_folds
        self.n_alphas = n_alphas
        self.uploadeddata = 0
        self.didfitptr = 0
        self.didpredict = 0

    def upload_data(self, sourceDev, trainX, trainY, validX=None, validY=None, weight=None):
        if self.uploadeddata == 1:
            self.freedata()
        self.uploadeddata = 1
        #
        #################
        if trainX is not None:
            try:
                if (trainX.dtype == np.float64):
                    print("Detected np.float64 trainX")
                    sys.stdout.flush()
                    self.double_precision1 = 1
                if (trainX.dtype == np.float32):
                    print("Detected np.float32 trainX")
                    sys.stdout.flush()
                    self.double_precision1 = 0
            except:
                self.double_precision1 = -1
            try:
                if trainX.value is not None:
                    mTrain = trainX.shape[0]
                    n1 = trainX.shape[1]
                else:
                    mTrain = 0
                    n1 = -1
            except:
                mTrain = trainX.shape[0]
                n1 = trainX.shape[1]
        else:
            mTrain = 0
            n1 = -1
        self.mTrain = mTrain
        ################
        if validX is not None:
            try:
                if (validX.dtype == np.float64):
                    self.double_precision2 = 1
                if (validX.dtype == np.float32):
                    self.double_precision2 = 0
            except:
                self.double_precision2 = -1
            #
            try:
                if validX.value is not None:
                    mValid = validX.shape[0]
                    n2 = validX.shape[1]
                else:
                    mValid = 0
                    n2 = -1
            except:
                mValid = validX.shape[0]
                n2 = validX.shape[1]
        else:
            mValid = 0
            n2 = -1
        self.mValid = mValid
        ###############
        if self.double_precision1 >= 0 and self.double_precision2 >= 0:
            if (self.double_precision1 != self.double_precision2):
                print("trainX and validX must be same precision")
                exit(0)
            else:
                self.double_precision = self.double_precision1  # either one
        elif self.double_precision1 >= 0:
            self.double_precision = self.double_precision1
        elif self.double_precision2 >= 0:
            self.double_precision = self.double_precision2
        ###############
        if n1 >= 0 and n2 >= 0:
            if (n1 != n2):
                print("trainX and validX must have same number of columns")
                exit(0)
            else:
                n = n1  # either one
        elif n1 >= 0:
            n = n1
        elif n2 >= 0:
            n = n2
        self.n = n
        ################
        a = c_void_p(0)
        b = c_void_p(0)
        c = c_void_p(0)
        d = c_void_p(0)
        e = c_void_p(0)
        if (self.double_precision == 1):
            null_ptr = POINTER(c_double)()
            #
            if trainX is not None:
                try:
                    if trainX.value is not None:
                        A = cptr(trainX, dtype=c_double)
                    else:
                        A = null_ptr
                except:
                    A = cptr(trainX, dtype=c_double)
            else:
                A = null_ptr
            if trainY is not None:
                try:
                    if trainY.value is not None:
                        B = cptr(trainY, dtype=c_double)
                    else:
                        B = null_ptr
                except:
                    B = cptr(trainY, dtype=c_double)
            else:
                B = null_ptr
            if validX is not None:
                try:
                    if validX.value is not None:
                        C = cptr(validX, dtype=c_double)
                    else:
                        C = null_ptr
                except:
                    C = cptr(validX, dtype=c_double)
            else:
                C = null_ptr
            if validY is not None:
                try:
                    if validY.value is not None:
                        D = cptr(validY, dtype=c_double)
                    else:
                        D = null_ptr
                except:
                    D = cptr(validY, dtype=c_double)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight, dtype=c_double)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight, dtype=c_double)
            else:
                E = null_ptr
            status = self.lib.make_ptr_double(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev),
                                              c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                              A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        elif (self.double_precision == 0):
            print("Detected np.float32")
            sys.stdout.flush()
            self.double_precision = 0
            null_ptr = POINTER(c_float)()
            #
            if trainX is not None:
                try:
                    if trainX.value is not None:
                        A = cptr(trainX, dtype=c_float)
                    else:
                        A = null_ptr
                except:
                    A = cptr(trainX, dtype=c_float)
            else:
                A = null_ptr
            if trainY is not None:
                try:
                    if trainY.value is not None:
                        B = cptr(trainY, dtype=c_float)
                    else:
                        B = null_ptr
                except:
                    B = cptr(trainY, dtype=c_float)
            else:
                B = null_ptr
            if validX is not None:
                try:
                    if validX.value is not None:
                        C = cptr(validX, dtype=c_float)
                    else:
                        C = null_ptr
                except:
                    C = cptr(validX, dtype=c_float)
            else:
                C = null_ptr
            if validY is not None:
                try:
                    if validY.value is not None:
                        D = cptr(validY, dtype=c_float)
                    else:
                        D = null_ptr
                except:
                    D = cptr(validY, dtype=c_float)
            else:
                D = null_ptr
            if weight is not None:
                try:
                    if weight.value is not None:
                        E = cptr(weight, dtype=c_float)
                    else:
                        E = null_ptr
                except:
                    E = cptr(weight, dtype=c_float)
            else:
                E = null_ptr
            status = self.lib.make_ptr_float(c_int(self.sharedA), c_int(self.sourceme), c_int(sourceDev),
                                             c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.ord),
                                             A, B, C, D, E, pointer(a), pointer(b), pointer(c), pointer(d), pointer(e))
        else:
            print("Unknown numpy type detected")
            print(trainX.dtype)
            sys.stdout.flush()
            return a, b, c, d, e

        assert status == 0, "Failure uploading the data"
        # print("a=",hex(a.value))
        # print("b=",hex(b.value))
        # print("c=",hex(c.value))
        # print("d=",hex(d.value))
        # print("e=",hex(e.value))
        self.solution.double_precision = self.double_precision
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        return a, b, c, d, e

    # sourceDev here because generally want to take in any pointer, not just from our test code
    def fitptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict, freeinputdata):
        # store some things for later call to predictptr()
        self.sourceDev = sourceDev
        self.mTrain = mTrain
        self.n = n
        self.mValid = mValid
        self.precision = precision
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.givefullpath = givefullpath

        #print("a"); print(a)
        #print("b"); print(b)
        #print("c"); print(c)
        #print("d"); print(d)
        #print("e"); print(e)
        #sys.stdout.flush()


        ############
        if dopredict == 0 and self.didfitptr == 1:
            self.freesols()
        else:
            # otherwise don't clear solution, just use it
            pass
        ################
        self.didfitptr = 1
        ###############
        # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
        if hasattr(self, 'double_precision'):
            whichprecision = self.double_precision
        else:
            whichprecision = precision
            self.double_precision = precision
        ##############
        if dopredict == 0:
            # initialize if doing fit
            Xvsalphalambda = c_void_p(0)
            Xvsalpha = c_void_p(0)
            validPredsvsalphalambda = c_void_p(0)
            validPredsvsalpha = c_void_p(0)
            countfull = c_size_t(0)
            countshort = c_size_t(0)
            countmore = c_size_t(0)
        else:
            # restore if predict
            Xvsalphalambda = self.Xvsalphalambda
            Xvsalpha = self.Xvsalpha
            validPredsvsalphalambda = self.validPredsvsalphalambda
            validPredsvsalpha = self.validPredsvsalpha
            countfull = self.countfull
            countshort = self.countshort
            countmore = self.countmore
        ################
        #
        c_size_t_p = POINTER(c_size_t)
        if (whichprecision == 1):
            self.mydtype = np.double
            self.myctype = c_double
            print("double precision fit")
            sys.stdout.flush()
            self.lib.elastic_net_ptr_double(
                c_int(dopredict),
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),
                c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                , givefullpath
                , pointer(Xvsalphalambda), pointer(Xvsalpha)
                , pointer(validPredsvsalphalambda), pointer(validPredsvsalpha)
                , cast(addressof(countfull), c_size_t_p), cast(addressof(countshort), c_size_t_p),
                cast(addressof(countmore), c_size_t_p)
            )
        else:
            self.mydtype = np.float
            self.myctype = c_float
            print("single precision fit")
            sys.stdout.flush()
            self.lib.elastic_net_ptr_float(
                c_int(dopredict),
                c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),
                c_int(self.ord),
                c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
                c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
                a, b, c, d, e
                , givefullpath
                , pointer(Xvsalphalambda), pointer(Xvsalpha)
                , pointer(validPredsvsalphalambda), pointer(validPredsvsalpha)
                , cast(addressof(countfull), c_size_t_p), cast(addressof(countshort), c_size_t_p),
                cast(addressof(countmore), c_size_t_p)
            )
        #
        # if should or user wanted to save or free data, do that now that we are done using a,b,c,d,e
        # This means have to upload_data() again before fitptr or predictptr or only call fit and predict
        if freeinputdata==1:
            freedata()
        #####################################
        # PROCESS OUTPUT
        # save pointers
        self.Xvsalphalambda = Xvsalphalambda
        self.Xvsalpha = Xvsalpha
        self.validPredsvsalphalambda = validPredsvsalphalambda
        self.validPredsvsalpha = validPredsvsalpha
        self.countfull = countfull
        self.countshort = countshort
        self.countmore = countmore
        #
        countfull_value = countfull.value
        countshort_value = countshort.value
        countmore_value = countmore.value
        # print("counts=%d %d %d" % (countfull_value,countshort_value,countmore_value))
        ######
        if givefullpath == 1:
            numall = int(countfull_value / (self.n_alphas * self.n_lambdas))
        else:
            numall = int(countshort_value / (self.n_alphas))
        #
        NUMALLOTHER = numall - n
        NUMRMSE = 3  # should be consistent with src/common/elastic_net_ptr.cpp
        NUMOTHER = NUMALLOTHER - NUMRMSE
        if NUMOTHER != 3:
            print("NUMOTHER=%d but expected 3" % (NUMOTHER))
            print("countfull_value=%d countshort_value=%d countmore_value=%d numall=%d NUMALLOTHER=%d" % (
            int(countfull_value), int(countshort_value), int(countmore_value), int(numall), int(NUMALLOTHER)))
            sys.stdout.flush()
            exit(0)
        #
        if givefullpath == 1 and dopredict == 0:
            # Xvsalphalambda contains solution (and other data) for all lambda and alpha
            self.Xvsalphalambdanew = np.fromiter(cast(Xvsalphalambda, POINTER(self.myctype)), dtype=self.mydtype,
                                                 count=countfull_value)
            self.Xvsalphalambdanew = np.reshape(self.Xvsalphalambdanew, (self.n_lambdas, self.n_alphas, numall))
            self.Xvsalphalambdapure = self.Xvsalphalambdanew[:, :, 0:n]
            self.rmsevsalphalambda = self.Xvsalphalambdanew[:, :, n:n + NUMRMSE]
            self.lambdas = self.Xvsalphalambdanew[:, :, n + NUMRMSE:n + NUMRMSE + 1]
            self.alphas = self.Xvsalphalambdanew[:, :, n + NUMRMSE + 1:n + NUMRMSE + 2]
            self.tols = self.Xvsalphalambdanew[:, :, n + NUMRMSE + 2:n + NUMRMSE + 3]
            #
            self.solution.Xvsalphalambdapure = self.Xvsalphalambdapure
            self.info.rmsevsalphalambda = self.rmsevsalphalambda
            self.info.lambdas = self.lambdas
            self.info.alphas = self.alphas
            self.info.tols = self.tols
        #
        if givefullpath == 1 and dopredict == 1:
            thecount = int(countfull_value / (n + NUMALLOTHER) * mValid)
            self.validPredsvsalphalambdanew = np.fromiter(cast(validPredsvsalphalambda, POINTER(self.myctype)),
                                                          dtype=self.mydtype, count=thecount)
            self.validPredsvsalphalambdanew = np.reshape(self.validPredsvsalphalambdanew,
                                                         (self.n_lambdas, self.n_alphas, mValid))
            self.validPredsvsalphalambdapure = self.validPredsvsalphalambdanew[:, :, 0:mValid]
            #
        if givefullpath == 0 and dopredict == 0:
            # Xvsalpha contains only best of all lambda for each alpha
            self.Xvsalphanew = np.fromiter(cast(Xvsalpha, POINTER(self.myctype)), dtype=self.mydtype,
                                           count=countshort_value)
            self.Xvsalphanew = np.reshape(self.Xvsalphanew, (self.n_alphas, numall))
            self.Xvsalphapure = self.Xvsalphanew[:, 0:n]
            self.rmsevsalpha = self.Xvsalphanew[:, n:n + NUMRMSE]
            self.lambdas2 = self.Xvsalphanew[:, n + NUMRMSE:n + NUMRMSE + 1]
            self.alphas2 = self.Xvsalphanew[:, n + NUMRMSE + 1:n + NUMRMSE + 2]
            self.tols2 = self.Xvsalphanew[:, n + NUMRMSE + 2:n + NUMRMSE + 3]
            #
            self.solution.Xvsalphapure = self.Xvsalphapure
            self.info.rmsevsalpha = self.rmsevsalpha
            self.info.lambdas2 = self.lambdas2
            self.info.alphas2 = self.alphas2
            self.info.tols2 = self.tols2
        #
        if givefullpath == 0 and dopredict == 1:  # exclusive set of validPreds unlike X
            thecount = int(countshort_value / (n + NUMALLOTHER) * mValid)
            print("thecount=%d countfull_value=%d countshort_value=%d n=%d NUMALLOTHER=%d mValid=%d" % (
            thecount, countfull_value, countshort_value, n, NUMALLOTHER, mValid))
            sys.stdout.flush()
            self.validPredsvsalphanew = np.fromiter(cast(validPredsvsalpha, POINTER(self.myctype)), dtype=self.mydtype,
                                                    count=thecount)
            self.validPredsvsalphanew = np.reshape(self.validPredsvsalphanew, (self.n_alphas, mValid))
            self.validPredsvsalphapure = self.validPredsvsalphanew[:, 0:mValid]
        #
        #######################
        # return numpy objects
        if dopredict == 0:
            self.didpredict = 0
            if givefullpath == 1:
                return (self.Xvsalphalambdapure, self.Xvsalphapure)
            else:
                return (self.Xvsalphapure)
        else:
            self.didpredict = 1
            if givefullpath == 1:
                return (self.validPredsvsalphalambdapure, self.validPredsvsalphapure)
            else:
                return (self.validPredsvsalphapure)

    def fit(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0, dopredict=0, freeinputdata=1):
        #
        self.givefullpath = givefullpath
        ################
        self.trainX = trainX
        self.trainY = trainY
        self.validX = validX
        self.validY = validY
        self.weight = weight
        ##############
        if trainX is not None:
            try:
                if trainX.value is not None:
                    # get shapes
                    shapeX = np.shape(trainX)
                    mTrain = shapeX[0]
                    n1 = shapeX[1]
                else:
                    print("no trainX")
                    n1 = -1
            except:
                # get shapes
                shapeX = np.shape(trainX)
                mTrain = shapeX[0]
                n1 = shapeX[1]
        else:
            print("no trainX")
            mTrain = 0
            n1 = -1
        #############
        if trainY is not None:
            try:
                if trainY.value is not None:
                    # get shapes
                    print("Doing fit")
                    shapeY = np.shape(trainY)
                    mY = shapeY[0]
                    if (mTrain != mY):
                        print("training X and Y must have same number of rows, but mTrain=%d mY=%d\n" % (mTrain, mY))
                else:
                    mY = -1
            except:
                # get shapes
                print("Doing fit")
                shapeY = np.shape(trainY)
                mY = shapeY[0]
                if (mTrain != mY):
                    print("training X and Y must have same number of rows, but mTrain=%d mY=%d\n" % (mTrain, mY))
        else:
            print("Doing predict")
            mY = -1
        ###############
        if validX is not None:
            try:
                if validX.value is not None:
                    shapevalidX = np.shape(validX)
                    mValid = shapevalidX[0]
                    n2 = shapevalidX[1]
                else:
                    print("no validX")
                    mValid = 0
                    n2 = -1
            except:
                shapevalidX = np.shape(validX)
                mValid = shapevalidX[0]
                n2 = shapevalidX[1]
        else:
            print("no validX")
            mValid = 0
            n2 = -1
        print("mValid=%d" % (mValid))
        sys.stdout.flush()
        ###############
        if validY is not None:
            try:
                if validY.value is not None:
                    shapevalidY = np.shape(validY)
                    mvalidY = shapevalidY[0]
                else:
                    print("no validY")
                    mvalidY = -1
            except:
                shapevalidY = np.shape(validY)
                mvalidY = shapevalidY[0]
        else:
            print("no validY")
            mvalidY = -1
        ################
        # check dopredict input
        if dopredict == 0:
            if n1 >= 0 and mY >= 0:
                print("Correct train inputs")
            else:
                print("Incorrect train inputs")
        if dopredict == 1:
            if (n1 == -1 and n2 >= 0 and mvalidY == -1 and mY == -1) or (n1 == -1 and n2 >= 0 and mY == -1):
                print("Correct prediction inputs")
            else:
                print("Incorrect prediction inputs")
        #################
        if dopredict == 0:
            if (n1 >= 0 and n2 >= 0 and n1 != n2):
                print("trainX and validX must have same number of columns, but n=%d n2=%d\n" % (n1, n2))
            else:
                n = n1  # either
        else:
            n = n2  # pick validX
        ##################
        if dopredict == 0:
            if (mValid >= 0 and mvalidY >= 0 and mValid != mvalidY):
                print("validX and validY must have same number of rows, but mValid=%d mvalidY=%d\n" % (mValid, mvalidY))
        else:
            # otherwise mValid is used, and mvalidY can be there or not (sets whether do RMSE or not)
            pass
        #################
        if dopredict == 0:
            if ((validX and validY == None) or (validX == None and validY)):
                print(
                    "Must input both validX and validY or neither.")  # TODO FIXME: Don't need validY if just want preds and no error, but don't return error in fit, so leave for now
                exit(0)
                #
        ##############
        sourceDev = 0  # assume GPU=0 is fine as source
        a, b, c, d, e = self.upload_data(sourceDev, trainX, trainY, validX, validY, weight)
        precision = 0  # won't be used
        self.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict=dopredict, freeinputdata=freeinputdata)
        if dopredict == 0:
            if givefullpath == 1:
                return (self.Xvsalphalambdapure, self.Xvsalphapure)
            else:
                return (self.Xvsalphapure)
        else:
            if givefullpath == 1:
                return (self.validPredsvsalphalambdapure, self.validPredsvsalphapure)
            else:
                return (self.validPredsvsalphapure)

    def getrmse(self):
        if self.givefullpath:
            return (self.rmsevsalphalambda)
        else:
            return (self.rmsevsalpha)

    def getlambdas(self):
        if self.givefullpath:
            return (self.lambdas)
        else:
            return (self.lambdas2)

    def getalphas(self):
        if self.givefullpath:
            return (self.alphas)
        else:
            return (self.alphas2)

    def gettols(self):
        if self.givefullpath:
            return (self.tols)
        else:
            return (self.tols2)

    def predict(self, validX, validY=None, testweight=None, givefullpath=0, freeinputdata=1):
        # if pass None trainx and trainY, then do predict using validX and weight (if given)
        # unlike upload_data and fitptr (and so fit) don't free-up predictions since for single model might request multiple predictions.  User has to call finish themselves to cleanup.
        dopredict = 1
        self.prediction = self.fit(None, None, validX, validY, testweight, givefullpath, dopredict, freeinputdata)
        return (self.prediction)  # something like validY

    def predictptr(self, validXptr, validYptr=None, givefullpath=0, freeinputdata=0):
        dopredict = 1
        #print("%d %d %d %d %d" % (self.sourceDev, self.mTrain, self.n, self.mValid, self.precision)) ; sys.stdout.flush()
        self.prediction = self.fitptr(self.sourceDev, self.mTrain, self.n, self.mValid, self.precision, self.a, self.b, validXptr, validYptr, self.e, givefullpath, dopredict, freeinputdata)
        return (self.prediction)  # something like validY

    def fit_predict(self, trainX, trainY, validX=None, validY=None, weight=None, givefullpath=0, freeinputdata=1):
        dopredict = 0  # only fit at first
        self.fit(trainX, trainY, validX, validY, weight, givefullpath, dopredict, freeinputdata=0)
        if validX == None:
            self.prediction = self.predict(trainX, trainY, testweight=weight, givefullpath=givefullpath, freeinputdata=freeinputdata)
        else:
            self.prediction = self.predict(validX, validY, testweight=weight, givefullpath=givefullpath, freeinputdata=freeinputdata)
        return (self.prediction)

    def fit_predictptr(self, sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath=0, freeinputdata=0):
        dopredict = 0  # only fit at first
        self.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath, dopredict, freeinputdata=0)
        if c is None or c is c_void_p(0):
            self.prediction = self.predictptr(a, b, givefullpath, freeinputdata=freeinputdata)
        else:
            self.prediction = self.predictptr(c, d, givefullpath, freeinputdata=freeinputdata)
        return (self.prediction)

    def freedata(self):
        # NOTE: For now, these are automatically freed when done with fit -- ok, since not used again
        if self.uploadeddata == 1:
            self.uploadeddata = 0
            if self.double_precision == 1:
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

    def freesols(self):
        if self.didfitptr == 1:
            self.didfitptr = 0
            if self.double_precision == 1:
                self.lib.modelfree2_double(self.Xvsalphalambda)
                self.lib.modelfree2_double(self.Xvsalpha)
            else:
                self.lib.modelfree2_float(self.Xvsalphalambda)
                self.lib.modelfree2_float(self.Xvsalpha)

    def freepreds(self):
        if self.didpredict == 1:
            self.didpredict = 0
            if self.double_precision == 1:
                self.lib.modelfree2_double(self.validPredsvsalphalambda)
                self.lib.modelfree2_double(self.validPredsvsalpha)
            else:
                self.lib.modelfree2_float(self.validPredsvsalphalambda)
                self.lib.modelfree2_float(self.validPredsvsalpha)

    def finish(self):
        self.freedata()
        self.freesols()
        self.freepreds()
