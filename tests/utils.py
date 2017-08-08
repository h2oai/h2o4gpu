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
    give_full_path = 1
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
        enet.fit_ptr(sourceDev, mTrain, n, mValid, precision, a, b, c, d, e, give_full_path=give_full_path)
    else:
        enet.fit(a, b, c, d, e, give_full_path=give_full_path)
    #t1 = time.time()
    print("Done Solving\n") ; sys.stdout.flush()

    rmse_train = printallerrors(display, enet, "Train", give_full_path)

    print('Predicting') ; sys.stdout.flush()
    if use_gpu == 1:
        pred_val = enet.predict_ptr(c, d, give_full_path=give_full_path)
    else:
        pred_val = enet.predict(c, give_full_path=give_full_path)
    print('Done Predicting') ; sys.stdout.flush()
    print('predicted values:\n', pred_val)

    rmse_test = printallerrors(display, enet, "Test", give_full_path)

    if write == 0:
        os.system('rm -f rmse.txt; rm -f pred*.txt; rm -f varimp.txt; rm -f me*.txt; rm -f stats.txt')

    return pred_val, rmse_train, rmse_test

def printallerrors(display, enet, str, give_full_path):
    rmse = enet.get_error
    alphas = enet.get_alphas
    lambdas = enet.get_lambdas
    tols = enet.get_tols
    if give_full_path == 1:
        rmse_full = enet.get_error_full
        alphas_full = enet.get_alphas_full
        lambdas_full = enet.get_lambdas_full
        tols_full = enet.get_tols_full
    rmse_best = enet.get_error_best
    alphas_best = enet.get_alphas_best
    lambdas_best = enet.get_lambdas_best
    tols_best = enet.get_tols_best

    if display == 1:
        # Display most important metrics
        print('Default %s RMSE best: ', (str,rmse))
        print('Default %s ALPHAS best: ', (str, alphas))
        print('Default %s LAMBDAS best: ', (str, lambdas))
        print('Default %s TOLS best: ', (str, tols))
        if give_full_path == 1:
            print('%s RMSE full path: ', (str, rmse_full))
            print('%s ALPHAS full path: ', (str, alphas_full))
            print('%s LAMBDAS full path: ', (str, lambdas_full))
            print('%s TOLS full path: ', (str, tols_full))
        print('%s RMSE best: ', (str, rmse_best))
        print('%s ALPHAS best: ', (str, alphas_best))
        print('%s LAMBDAS best: ', (str, lambdas_best))
        print('%s TOLS best: ', (str, tols_best))
    return rmse_best


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
    enet = Solver(sharedA, nThreads, nGPUs, 'c' if fortran else 'r', intercept, standardize, lambda_min_ratio, nLambdas, nFolds, nAlphas, verbose=verbose, family=family)

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    ## Solve
    print("Solving")
    Xvsalpha = enet.fit(trainX, trainY)
    # Xvsalpha = enet.fit(trainX, trainY, validX, validY)
    # Xvsalpha = enet.fit(trainX, trainY, validX, validY, trainW)
    # Xvsalphalambda = enet.fit(trainX, trainY, validX, validY, trainW, 0)
    # give_full_path=1 ; Xvsalphalambda = enet.fit(trainX, trainY, validX, validY, trainW, give_full_path)
    print("Done Solving")

    X=enet.get_X
    print("X")
    print(X)

    # show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(Xvsalpha)
    print("np.shape(Xvsalpha)")
    print(np.shape(Xvsalpha))

    rmse = enet.get_error
    print("rmse")
    print(rmse)

    print("lambdas")
    lambdas = enet.get_lambdas
    print(lambdas)

    print("alphas")
    alphas = enet.get_alphas
    print(alphas)

    print("tols")
    tols = enet.get_tols
    print(tols)

    testvalidY = np.dot(trainX, Xvsalpha.T)
    print("testvalidY (newvalidY should be this)")
    print(testvalidY)

    print("Predicting, assuming unity weights")
    if validX == None or mvalid == 0:
        print("Using trainX for validX")
        newvalidY = enet.predict(trainX)  # for testing
    else:
        print("Using validX for validX")
        newvalidY = enet.predict(validX)
    print("newvalidY")
    print(newvalidY)

    print("Done Reporting")
    return enet, rmse
