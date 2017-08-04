import h2ogpuml
import numpy as np
import os, sys
from numba import cuda
from numba.cuda.cudadrv import driver
from time import time
from ctypes import *


def runH2oAiGlm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, use_gpu):

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
        rmse_full, rmse = enet.getrmse()
        alphas_full, alphas = enet.getalphas()
        lambdas_full, lambdas = enet.getlambdas()
        tols_full, tols = enet.gettols()
        if givefullpath==1:
            print('Train RMSE full path: ', (rmse_full))
            print('Train ALPHAS full path: ', (alphas_full))
            print('Train LAMBDAS full path: ', (lambdas_full))
            print('Train TOLS full path: ', (tols_full))
        print('Train RMSE best: ', (rmse))
        print('Train ALPHAS best: ', (alphas))
        print('Train LAMBDAS best: ', (lambdas))
        print('Train TOLS best: ', (tols))

    # trmse = enet.getrmse()
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
        rmse_full, rmse = enet.getrmse()
        alphas_full, alphas = enet.getalphas()
        lambdas_full, lambdas = enet.getlambdas()
        tols_full, tols = enet.gettols()
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

    return pred_val

if __name__ == "__main__":

    xtrain = np.loadtxt("/home/jon/h2ogpuml-data/wamsi1/xtrain.txt", delimiter=',', dtype=np.float32)
    ytrain = np.loadtxt("/home/jon/h2ogpuml-data/wamsi1/ytrain.txt", delimiter=',', dtype=np.float32)
    xtest = np.loadtxt("/home/jon/h2ogpuml-data/wamsi1/xtest.txt", delimiter=',', dtype=np.float32)
    ytest = np.loadtxt("/home/jon/h2ogpuml-data/wamsi1/ytest.txt", delimiter=',', dtype=np.float32)
    wtrain = np.ones((xtrain.shape[0], 1), dtype=np.float32)

    use_gpu = 1  # set it to 1 for using GPUS, 0 for CPU
    display = 1
    if 1==1:
        write = 0
        nFolds=5
        nLambdas=100
        nAlphas=8
    else:
        write = 1
        nFolds=1
        nLambdas=1
        nAlphas=1
    t1 = time()
    runH2oAiGlm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, use_gpu)
    print('/n Total execution time:%d' % (time() - t1))
