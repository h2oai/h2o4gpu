import cProfile
import pstats
import math
import pytest
import h2o4gpu
import numpy as np
import os, sys
from numba import cuda
from numba.cuda.cudadrv import driver
from time import time
from ctypes import *
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


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


def runglm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, nGPUs=1, name=None):

    if nGPUs > 0:
        use_gpu = True
    else:
        use_gpu = False
    

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
        #nGPUs = maxNGPUS  # choose all GPUs
        # nGPUs = 1

        n = train_data_mat.shape[1]
        mTrain = train_data_mat.shape[0]
        mValid = test_data_mat.shape[0]

    else:
        #nGPUs = 0
        n = xtrain.shape[1]
        mTrain = xtrain.shape[0]
        mValid = xtest.shape[0]

    print("No. of Features=%d mTrain=%d mValid=%d" % (n, mTrain, mValid))

    # Order of data
    fortran = 1
    print("fortran=%d" % (fortran))

    sourceme = 0
    sourceDev = 0
    intercept = True # should be passed in from above if user added intercept
    nThreads = None
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

    print("Setting up Solver")
    sys.stdout.flush()

    Solver = h2o4gpu.GLM
    enet = Solver(n_threads=nThreads, n_gpus=nGPUs, order='c' if fortran else 'r', intercept=intercept,
                  lambda_min_ratio=lambda_min_ratio, n_lambdas=nLambdas,
                  n_folds=nFolds, n_alphas=nAlphas, verbose=5, give_full_path=give_full_path)

    print("Solving")
    sys.stdout.flush()
    if use_gpu == 1:
        enet.fit_ptr(sourceDev, mTrain, n, mValid, precision, None, a, b, c, d, e)
    else:
        enet.fit(a, b, c, d, e)
    # t1 = time.time()
    print("Done Solving\n")
    sys.stdout.flush()

    error_train = printallerrors(display, enet, "Train", give_full_path)

    print('Predicting')
    sys.stdout.flush()
    if use_gpu == 1:
        pred_val = enet.predict_ptr(c, d)
    else:
        pred_val = enet.predict(c)
    print('Done Predicting')
    sys.stdout.flush()
    print('predicted values:\n', pred_val)

    error_test = printallerrors(display, enet, "Test", give_full_path)

    if write == 0:
        os.system('rm -f error.txt; rm -f pred*.txt; rm -f varimp.txt; rm -f me*.txt; rm -f stats.txt')
    enet.finish()

    return pred_val, error_train, error_test


def printallerrors(display, enet, str, give_full_path):
    error = enet.error
    alphas = enet.alphas
    lambdas = enet.lambdas
    tols = enet.tols
    if give_full_path == 1:
        error_full = enet.error_full
        alphas_full = enet.alphas_full
        lambdas_full = enet.lambdas_full
        tols_full = enet.tols_full
    error_best = enet.error_best
    alphas_best = enet.alphas_best
    lambdas_best = enet.lambdas_best
    tols_best = enet.tols_best

    loss = "RMSE"

    if enet.family == "logistic":
        loss = "LOGLOSS"
    if display == 1:
        # Display most important metrics
        print('%s for %s  ' % (loss, str), error)
        print('ALPHAS for %s  ' % str, alphas)
        print('LAMBDAS for %s  ' % str, lambdas)
        print('TOLS for %s  ' % str, tols)
        if give_full_path == 1:
            print('full path : ', (str, loss, error_full))
            print('ALPHAS full path : ', (str, alphas_full))
            print('LAMBDAS full path : ', (str, lambdas_full))
            print('TOLS full path : ', (str, tols_full))
        print('Best %s for %s  ' % (loss, str), error_best)
        print('Best ALPHAS for %s  ' % str, alphas_best)
        print('Best LAMBDAS for %s  ' % str, lambdas_best)
        print('Best TOls for %s  ' % str, tols_best)
    return error_best


def elastic_net(X, y, nGPUs=0, nlambda=100, nfolds=5, nalpha=5, validFraction=0.2, family="elasticnet", verbose=0,
                print_all_errors=False, get_preds=False, run_h2o=False, tolerance=.01, name=None):
    print("Doing %s" % (name))
    sys.stdout.flush()
    doassert=0

    # override run_h2o False default if environ exists
    if os.getenv("H2OGLM_PERFORMANCE") is not None:
        run_h2o = True

    # Start up h2o
    if run_h2o:
        h2o.init(strict_version_check=False)

    # Choose solver
    Solver = h2o4gpu.GLM

    nThreads = None  # let internal method figure this out
    intercept = True
    lambda_min_ratio = 1e-9 # Causes issue for h2o-3 when using 1k ipums dataset
    nFolds = nfolds
    nLambdas = nlambda
    nAlphas = nalpha
    give_full_path = 1

    # Setup Train/validation Set Split
    morig = X.shape[0]
    norig = X.shape[1]
    print("Original m=%d n=%d" % (morig, norig))
    sys.stdout.flush()
    fortran = X.flags.f_contiguous
    print("fortran=%d" % fortran)
    sys.stdout.flush()

    # Do train/valid split
    HO = int(validFraction * morig)
    H = morig - HO
    print("Size of Train rows=%d valid rows=%d" % (H, HO))
    sys.stdout.flush()
    trainX = np.copy(X[0:H, :])
    trainY = np.copy(y[0:H])

    if run_h2o:
        print("Build Training H2OFrames")
        trainX_h2o = h2o.H2OFrame(trainX)
        trainY_h2o = h2o.H2OFrame(trainY)
        train_h2o = trainX_h2o.cbind(trainY_h2o)

    validX = None
    validY = None
    if validFraction != 0.0:
        validX = np.copy(X[H:morig, :])
        validY = np.copy(y[H:morig])
        mvalid = validX.shape[0]
        validX = np.hstack([validX, np.ones((validX.shape[0], 1), dtype=validX.dtype)])

        if run_h2o:
            print("Build Validation H2OFrames")
            validX_h2o = h2o.H2OFrame(validX)
            validY_h2o = h2o.H2OFrame(validY)
            valid_h2o = validX_h2o.cbind(validY_h2o)

    mTrain = trainX.shape[0]
    if validFraction != 0.0:
        print("mTrain=%d mvalid=%d" % (mTrain, mvalid))
    else:
        print("mTrain=%d" % mTrain)

    if intercept is True:
        trainX = np.hstack([trainX, np.ones((trainX.shape[0], 1), dtype=trainX.dtype)])
        n = trainX.shape[1]
        print("New n=%d" % n)

    # Constructor
    print("Setting up solver")
    sys.stdout.flush()
    enet = Solver(n_threads=nThreads, n_gpus=nGPUs, intercept=intercept,
                  lambda_min_ratio=lambda_min_ratio,
                  n_lambdas=nLambdas, n_folds=nFolds, n_alphas=nAlphas, verbose=verbose, family=family, give_full_path=give_full_path)

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    # Solve
    if validFraction == 0.0:
        print("Solving")
        Xvsalpha = enet.fit(trainX, trainY)
    else:
        Xvsalpha = enet.fit(trainX, trainY, validX, validY)

    print("Done Solving")

    X = enet.X
    print("X")
    print(X)

    # Show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(Xvsalpha.x_vs_alphapure)
    print("np.shape(Xvsalpha)")
    print(np.shape(Xvsalpha.x_vs_alphapure))

    error_train = enet.error_vs_alpha
    if family != "logistic":
        print("error_train")
    else:
        print("logloss_train")
    print(error_train)

    print("Best lambdas")
    lambdas = enet.lambdas_best
    print(lambdas)

    print("Best alphas")
    alphas = enet.alphas_best
    print(alphas)

    print("Best tols")
    tols = enet.tols_best
    print(tols)

    Xvsalphabest=enet.X_best

    testvalidY = np.dot(trainX, Xvsalphabest.T)
    print("testvalidY (newvalidY should be this)")
    if family != "logistic":
        print(testvalidY)
    else:
        try:
            inverse_logit = lambda t: 1 / (1 + math.exp(-t))
            testvalidY = np.round(testvalidY, 1)  # Round to avoid math OverFlow error
            func = np.vectorize(inverse_logit)
            print(func(testvalidY))
        except OverflowError:
            print(testvalidY)

    print(testvalidY)

    print("Predicting, assuming unity weights")
    if validFraction == 0.0:
        print("Using trainX for validX")
        if trainY is not None:
            newvalidY = enet.predict(trainX, trainY)  # for testing
        else:
            newvalidY = enet.predict(trainX)  # for testing
        if get_preds:
            print("Saving train preds (Need to tranpose output)")
            np.savetxt("preds_train.csv", newvalidY, delimiter=",")
    else:
        print("Using validX for validX")
        if validY is not None:
            newvalidY = enet.predict(validX, validY)
        else:
            newvalidY = enet.predict(validX)
        if get_preds:
            print("Saving valid preds (Need to tranpose output)")
            np.savetxt("preds_valid.csv", newvalidY, delimiter=",")
    print("newvalidY")
    print(newvalidY)

    error_test = enet.error_vs_alpha
    if family != "logistic":
        print("rmse_test")
    else:
        print("logloss_test")
    print(error_test)

    if print_all_errors:
        print("PRINT ALL ERRORS")
        print(printallerrors(display=1, enet=enet, str="Train", give_full_path=give_full_path))

    enet.finish()
    print("Done Reporting")

    if run_h2o:
        alphas_h2o = [item for alphas[0] in alphas for item in alphas[0]]
        for alpha in alphas_h2o:
            print("Setting up H2O Solver with alpha = %s" % alpha)
            nfoldsh2o = nfolds
            if nfoldsh2o == 1:
                nfoldsh2o = 0
            if family == "logistic":
                #TODO need to figure out a proper lamba min for h2o-3
                h2o_glm = H2OGeneralizedLinearEstimator(intercept=intercept,
                                                        lambda_search=True, nlambdas=nLambdas, nfolds=nfoldsh2o,
                                                        family="binomial", alpha=alpha)
            else:
                h2o_glm = H2OGeneralizedLinearEstimator(intercept=intercept,
                                                        lambda_search=True, nlambdas=nLambdas, nfolds=nfoldsh2o,
                                                        family="gaussian", alpha=alpha)
            # Solve
            if validFraction == 0.0:
                print("Solving using H2O")
                h2o_glm.train(x=train_h2o.columns[:-1], y=train_h2o.columns[-1], training_frame=train_h2o)
            else:
                print("Solving using H2O")
                h2o_glm.train(x=train_h2o.columns[:-1], y=train_h2o.columns[-1], training_frame=train_h2o,
                              validation_frame=valid_h2o)
            print("\nComparing results to H2O")
            print("\nH2O GLM Summary")
            print(h2o_glm)

            if family == "logistic":
                print("\nTraining Logloss")
                print(h2o_glm.logloss())
                h2o_train_error = h2o_glm.logloss()
            else:
                print("\nTraining RMSE")
                print(h2o_glm.rmse())
                h2o_train_error = h2o_glm.rmse()

            if validFraction > 0.0:
                if family == "logistic":
                    print("\nValidation Logloss")
                    print(h2o_glm.logloss(valid=True))
                    h2o_valid_error = h2o_glm.logloss(valid=True)
                else:
                    print("\nValidation RMSE")
                    print(h2o_glm.rmse(valid=True))
                    h2o_valid_error = h2o_glm.rmse(valid=True)

            if nFolds > 1:
                if family == "logistic":
                    print("\nCross Validation Logloss")
                    print( h2o_glm.model_performance(xval=True).logloss())
                    print("\n")
                    h2o_cv_error = h2o_glm.model_performance(xval=True).logloss()

                else:
                    print("\nCross Validation RMSE")
                    print(h2o_glm.model_performance(xval=True).rmse())
                    print("\n")
                    h2o_cv_error = h2o_glm.model_performance(xval=True).rmse()

            # Tolerance for h2o glm - gpu glm logloss
            tolerance = tolerance

            # Train and nfolds
            if validFraction == 0.0 and nfolds > 1:
                error_range = 2
            # Train, valid, and nfolds
            elif validFraction > 0.0 and nfolds > 1:
                error_range = 3
            # Train set only
            else:
                error_range = 1

            path="./results"
            os.makedirs(path, exist_ok=True)
            f1=open(os.path.join(path, name + ".dat"), 'wt+')
            # TODO(navdeep): output error and performance metrics summary
            print('%s' % (name), file=f1, end="")

            # Compare to H2O
            index = alphas_h2o.index(alpha)
            for j in range(error_range):
                if j == 0:  # Compare to train error
                    thisrelerror = -(error_train[index, j] - h2o_train_error)/(abs(error_train[index, j]) + abs(h2o_train_error))
                    if error_train[index, j] > h2o_train_error:
                        if abs(error_train[index, j] - h2o_train_error) > tolerance:
                            print("Train error failure: %g %g" % (error_train[index, j],h2o_train_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                        else:
                            print(' -', file=f1, end="")
                    else:
                        print("H2O Train Error is larger than GPU GLM with alpha = %s" % alpha)
                        print("H2O Train Error is %s" % h2o_train_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' -', file=f1, end="")
                elif j == 1:  # Compare to average cv error
                    thisrelerror = -(error_train[index, j] - h2o_train_error)/(abs(error_train[index, j]) + abs(h2o_cv_error))
                    if error_train[index, j] > h2o_cv_error:
                        if abs(error_train[index, j] - h2o_cv_error) > tolerance:
                            print("CV error failure: %g %g" % (error_train[index, j],h2o_cv_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                        else:
                            print(' -', file=f1, end="")
                    else:
                        print("H2O CV Error is larger than GPU GLM with alpha = %s" % alpha)
                        print("H2O CV Error is %s" % h2o_cv_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' -', file=f1, end="")
                elif j == 2:  # Compare to validation error
                    thisrelerror = -(error_train[index, j] - h2o_train_error)/(abs(error_train[index, j]) + abs(h2o_valid_error))
                    if error_train[index, j] > h2o_valid_error:
                        if abs(error_train[index, j] - h2o_valid_error) > tolerance:
                            print("Valid error failure: %g %g" % (error_train[index, j],h2o_valid_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                        else:
                            print(' -', file=f1, end="")
                    else:
                        print("H2O Valid Error is larger than GPU GLM with alpha = %s" % alpha)
                        print("H2O Valid Error is %s" % h2o_valid_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' -', file=f1, end="")


                # TODO(navdeep): output error and performance metrics summary
                #print(' %g' % thisrelerror, file=f1, end="")

            print('',file=f1)

        # for pytest only:
        if os.getenv("H2OGLM_DISABLEPYTEST") is None:
            assert doassert==0

    if len(np.shape(error_train)) == 2:
       myerror_train = error_train
       myerror_test = error_test
    if len(np.shape(error_train)) == 3:
        myerror_train = error_train[-1]
        myerror_test = error_test[-1]

    return myerror_train, myerror_test

# TODO(navdeep): Does h2o-3 use validation frame to choose best fit or stop early, when nfolds>1?
# TODO(navdeep): You can mimic my "showresults.sh" process for make testperf and have timers that measure fit and predict performance for h2o4gpu and h2o-3.  I think just showing the ratio of the two is good, showing time for h2o4gpu over h2o-3 (so smaller is better).
# TODO(navdeep): Can use timers like:
#import time
#start = time.time()
#print("hello")
#end = time.time()
#print(end - start)
# TODO(navdeep): So every time we do make testperf, we'll get error and performance info.  For error info, would be cool if markdown or something (instead of text) and bad numbers were highlighted.  Also need h2o-3 results as separate file that's also printed so we can compare and see what went wrong.
