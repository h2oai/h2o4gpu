import cProfile
import pstats
import os

import pytest
import h2ogpuml
import numpy as np
import os, sys
from numba import cuda
from numba.cuda.cudadrv import driver
from time import time
from ctypes import *


def cprofile(file):
    def inner_cprofile(func):
        def cprofiled_func(*args, **kwargs):
            if True or os.environ.get('CPROFILE_TEST') is not None:
                profile = cProfile.Profile()
                try:
                    profile.enable()
                    result = func(*args, **kwargs)
                    profile.disable()
                    return result
                finally:
                    # if not os.path.exists("cprof"):
                    #     os.makedirs("cprof")
                    #
                    # basename = os.path.basename(file)
                    # profile_dump = "cprof/{}_{}.prof".format(os.path.splitext(basename)[0],
                    #                                          func.__name__)
                    profile.create_stats()
                    # profile.dump_stats(profile_dump)
                    print("Profile:")
                    s = pstats.Stats(profile)
                    s.sort_stats('cumulative').print_stats(20)
                    # os.remove("cprof")
            else:
                return func(*args, **kwargs)

        return cprofiled_func

    return inner_cprofile


def find_file(file):
    prefs = ["../../tests/data", "../tests/data", "tests/data", "tests", "data", "..", "."]
    for pre in prefs:
        file2 = os.path.abspath(os.path.join(pre, file))
        if os.path.isfile(file2):
            return file2
    raise FileNotFoundError("Couldn't find file %s in the predefined locations." % file)


def find_dir(dir):
    prefs = ["../../tests/data", "../tests/data", "tests/data", "tests", "data", "..", "."]
    for pre in prefs:
        dir2 = os.path.abspath(os.path.join(pre, dir))
        if os.path.isdir(dir2):
            return dir2
    raise FileNotFoundError("Couldn't find directory %s in the predefined locations." % dir)


def skip_if_no_smalldata():
    try:
        find_dir("smalldata")
    except FileNotFoundError:
        return True
    return False


need_small_data = pytest.mark.skipif(skip_if_no_smalldata(), reason="smalldata folder not found")




def runglm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, use_gpu):

    if use_gpu == 1:

        # nFolds, nAlphas, nLambdas = arg
        train_data_mat = cuda.to_device(xtrain)
        train_result_mat = cuda.to_device(ytrain)
        test_data_mat = cuda.to_device(xtest)
        test_result_mat = cuda.to_device(ytest)
        train_w_mat = cuda.to_device(wtrain)

        train_data_mat_ptr = train_data_mat.device_ctypes_pointer
        train_result_mat_ptr = train_result_mat.device_ctypes_pointer
        test_data_mat_ptr = test_data_mat.device_ctypes_pointer
        test_result_mat_ptr = test_result_mat.device_ctypes_pointer
        train_w = train_w_mat.device_ctypes_pointer

        print(train_data_mat_ptr)
        print(train_result_mat_ptr)
        print(test_data_mat_ptr)
        print(test_result_mat_ptr)

        import subprocess
        maxNGPUS = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
        print("Maximum Number of GPUS:", maxNGPUS)
        nGPUs = maxNGPUS  # choose all GPUs
        #nGPUs = 1

        n = train_data_mat.shape[1]
        mTrain = train_data_mat.shape[0]
        mValid = test_data_mat.shape[0]

    else:
        nGPUs = 0
        n = xtrain.shape[1]
        mTrain = xtrain.shape[0]
        mValid = xtest.shape[0]

    print("No. of Features=%d mTrain=%d mValid=%d" % (n, mTrain, mValid))

    # Order of data
    fortran = 1
    print("fortran=%d" % (fortran))

    sharedA = 0
    sourceme = 0
    sourceDev = 0
    intercept = 1
    nThreads = None
    standardize = 0
    lambda_min_ratio = 1e-9
    givefullpath = 1
    precision = 0
    # variables
    if use_gpu == 1:
        a, b = c_void_p(train_data_mat_ptr.value), c_void_p(train_result_mat_ptr.value)
        c, d = c_void_p(test_data_mat_ptr.value), c_void_p(test_result_mat_ptr.value)
        e = c_void_p(train_w.value)
        print(a, b, c, d, e)

    else:
        a, b = xtrain, ytrain
        c, d = xtest, ytest
        e = wtrain

    print("Setting up Solver") ; sys.stdout.flush()

    Solver = h2ogpuml.GLM
    enet = Solver(sharedA, nThreads, nGPUs, 'c' if fortran else 'r', intercept, standardize, lambda_min_ratio, nLambdas, nFolds, nAlphas, verbose=5)

    print("Solving") ; sys.stdout.flush()
    if use_gpu == 1:
        enet.fitptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, givefullpath)
    else:
        enet.fit(a, b, c, d, e, givefullpath)
    #t1 = time.time()
    print("Done Solving\n") ; sys.stdout.flush()

    if display == 1:
        # Display most important metrics
        rmse_full, rmse = enet.get_error
        alphas_full, alphas = enet.get_alphas
        lambdas_full, lambdas = enet.get_lambdas
        tols_full, tols = enet.get_tols
        if givefullpath==1:
            print('Train RMSE full path: ', (rmse_full))
            print('Train ALPHAS full path: ', (alphas_full))
            print('Train LAMBDAS full path: ', (lambdas_full))
            print('Train TOLS full path: ', (tols_full))
        print('Train RMSE best: ', (rmse))
        print('Train ALPHAS best: ', (alphas))
        print('Train LAMBDAS best: ', (lambdas))
        print('Train TOLS best: ', (tols))

    # trmse = enet.get_error
    # print(trmse)

    print('Predicting') ; sys.stdout.flush()
    if use_gpu == 1:
        pred_valfull, pred_val = enet.predictptr(c, d, givefullpath)
    else:
        pred_valfull, pred_val = enet.predict(c, givefullpath)
    print('Done Predicting') ; sys.stdout.flush()
    print('predicted values:\n', pred_val)
    if givefullpath==1:
        print('full predicted values:\n', pred_valfull)

    if display == 1:
        # Display most important metrics
        rmse_full, rmse = enet.get_error
        alphas_full, alphas = enet.get_alphas
        lambdas_full, lambdas = enet.get_lambdas
        tols_full, tols = enet.get_tols
        if givefullpath==1:
            print('Test RMSE full path: ', (rmse_full))
            print('Test ALPHAS full path: ', (alphas_full))
            print('Test LAMBDAS full path: ', (lambdas_full))
            print('Test TOLS full path: ', (tols_full))
        print('Test RMSE best: ', (rmse))
        print('Test ALPHAS best: ', (alphas))
        print('Test LAMBDAS best: ', (lambdas))
        print('Test TOLS best: ', (tols))


    if write == 0:
        os.system('rm -f rmse.txt; rm -f pred*.txt; rm -f varimp.txt; rm -f me*.txt; rm -f stats.txt')

    return pred_val, rmse



def elastic_net(X, y, nGPUs=0, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2, family="elasticnet", verbose=0):
    # choose solver
    Solver = h2ogpuml.GLM

    sharedA = 0
    nThreads = None  # let internal method figure this out
    intercept = 1
    standardize = 0
    lambda_min_ratio = 1e-9
    nFolds = nfolds
    nLambdas = nlambda
    nAlphas = nalpha

    if standardize:
        print("implement standardization transformer")
        exit()

    # Setup Train/validation Set Split
    morig = X.shape[0]
    norig = X.shape[1]
    print("Original m=%d n=%d" % (morig, norig))
    fortran = X.flags.f_contiguous
    print("fortran=%d" % fortran)

    # Do train/valid split
    HO = int(validFraction * morig)
    H = morig - HO
    print("Size of Train rows=%d valid rows=%d" % (H, HO))
    trainX = np.copy(X[0:H, :])
    trainY = np.copy(y[0:H])
    validX = np.copy(X[H:-1, :])

    mTrain = trainX.shape[0]
    mvalid = validX.shape[0]
    print("mTrain=%d mvalid=%d" % (mTrain, mvalid))

    if intercept == 1:
        trainX = np.hstack([trainX, np.ones((trainX.shape[0], 1), dtype=trainX.dtype)])
        validX = np.hstack([validX, np.ones((validX.shape[0], 1), dtype=validX.dtype)])
        n = trainX.shape[1]
        print("New n=%d" % n)

    ## Constructor
    print("Setting up solver")
    enet = Solver(sharedA, nThreads, nGPUs, 'c' if fortran else 'r', intercept, standardize, lambda_min_ratio, nLambdas, nFolds, nAlphas, verbose=verbose,family=family)

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    ## Solve
    print("Solving")
    Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW)
    # Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW, 0)
    # givefullpath=1
    #  Xvsalphalambda, Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW, givefullpath)
    print("Done Solving")

    # show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(Xvsalpha)
    print("np.shape(Xvsalpha)")
    print(np.shape(Xvsalpha))

    rmsefull, rmse = enet.get_error
    print("rmse")
    print(rmse)

    print("lambdas")
    lambdasfull, lambdas = enet.get_lambdas
    print(lambdas)

    print("alphas")
    alphasfull, alphas = enet.get_alphas
    print(alphas)

    print("tols")
    tolsfull, tols = enet.get_tols
    print(tols)

    testvalidY = np.dot(trainX, Xvsalpha.T)
    print("testvalidY (newvalidY should be this)")
    print(testvalidY)

    print("Predicting, assuming unity weights")
    if validX == None or mvalid == 0:
        print("Using trainX for validX")
        newvalidYfull, newvalidY = enet.predict(trainX)  # for testing
    else:
        print("Using validX for validX")
        newvalidYfull, newvalidY = enet.predict(validX)
    print("newvalidY")
    print(newvalidY)

    print("Done Reporting")
    return enet, rmse
