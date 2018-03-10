# pylint: skip-file
#- * - encoding : utf - 8 - * -
"""
Test utils

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import math
import os
import sys
from time import time
import h2o4gpu
import numpy as np


def find_file(file):
    prefs = [
        "../../tests/data", "../tests/data", "tests/data", "tests", "data",
        "..", "."
    ]
    for pre in prefs:
        file2 = os.path.abspath(os.path.join(pre, file))
        if os.path.isfile(file2):
            return file2
    raise FileNotFoundError(
        "Couldn't find file %s in the predefined locations." % file)


def find_dir(directory):
    prefs = [
        "../../tests/data", "../tests/data", "tests/data", "tests", "data",
        "..", "."
    ]
    for pre in prefs:
        dir2 = os.path.abspath(os.path.join(pre, directory))
        if os.path.isdir(dir2):
            return dir2
    raise FileNotFoundError(
        "Couldn't find directory %s in the predefined locations." % directory)


def skip_if_no_smalldata():
    try:
        find_dir("smalldata")
    except FileNotFoundError:
        return True
    return False


#need_small_data = pytest.mark.skipif(
#    skip_if_no_smalldata(), reason="smalldata folder not found")


#assumes has intercept at last column already in xtrain and xtest
def run_glm_ptr(nFolds,
                nAlphas,
                nLambdas,
                xtrain,
                ytrain,
                xtest,
                ytest,
                wtrain,
                write,
                display,
                nGPUs=1):
    """Runs ElasticNetH2O test"""
    use_gpu = nGPUs > 0

    if use_gpu == 1:

        from numba import cuda
        #nFolds, nAlphas, nLambdas = arg
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
        maxNGPUS = int(
            subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
        print("Maximum Number of GPUS:", maxNGPUS)
        #nGPUs = maxNGPUS #choose all GPUs
        #nGPUs = 1

        n = train_data_mat.shape[1]
        mTrain = train_data_mat.shape[0]
        mValid = test_data_mat.shape[0]

    else:
        #nGPUs = 0
        n = xtrain.shape[1]
        mTrain = xtrain.shape[0]
        mValid = xtest.shape[0]

    print("No. of Features=%d mTrain=%d mValid=%d" % (n, mTrain, mValid))

    #Order of data
    fortran = 1
    print("fortran=%d" % (fortran))

    sourceDev = 0
    # should be passed from above if user set fit_intercept
    fit_intercept = True
    lambda_min_ratio = 1e-9
    store_full_path = 1
    double_precision = 0
    #variables
    if use_gpu == 1:
        from ctypes import c_void_p
        a, b = c_void_p(train_data_mat_ptr.value), c_void_p(
            train_result_mat_ptr.value)
        c, d = c_void_p(test_data_mat_ptr.value), c_void_p(
            test_result_mat_ptr.value)
        e = c_void_p(train_w.value)
        print(a, b, c, d, e)

    else:
        a, b = xtrain, ytrain
        c, d = xtest, ytest
        e = wtrain

    print("Setting up Solver")
    sys.stdout.flush()

    Solver = h2o4gpu.ElasticNetH2O
    enet = Solver(
        n_gpus=nGPUs,
        order='c' if fortran else 'r',
        fit_intercept=fit_intercept,
        lambda_min_ratio=lambda_min_ratio,
        n_lambdas=nLambdas,
        n_folds=nFolds,
        n_alphas=nAlphas,
        verbose=5,
        store_full_path=store_full_path)

    print("Solving")
    sys.stdout.flush()
    if use_gpu == 1:
        enet.fit_ptr(
            mTrain,
            n,
            mValid,
            double_precision,
            None,
            a,
            b,
            c,
            d,
            e,
            source_dev=sourceDev)
    else:
        enet.fit(a, b, c, d, e)


#t1 = time()
    print("Done Solving\n")
    sys.stdout.flush()

    error_train = printallerrors(display, enet, "Train", store_full_path)

    print('Predicting')
    sys.stdout.flush()
    if use_gpu == 1:
        pred_val = enet.predict_ptr(c, d)
    else:
        pred_val = enet.predict(c)
    print('Done Predicting')
    sys.stdout.flush()
    print('predicted values:\n', pred_val)

    error_test = printallerrors(display, enet, "Test", store_full_path)

    if write == 0:
        os.system('rm -f error.txt; '
                  'rm -f pred*.txt; '
                  'rm -f varimp.txt; '
                  'rm -f me*.txt; '
                  'rm -f stats.txt')
    from ..solvers.utils import finish
    finish(enet)

    return pred_val, error_train, error_test


def printallerrors(display, enet, string, store_full_path):
    """Pretty print all the errors"""
    error = enet.error
    alphas = enet.alphas
    lambdas = enet.lambdas
    tols = enet.tols
    if store_full_path == 1:
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
        #Display most important metrics
        print('%s for %s  ' % (loss, string), error)
        print('ALPHAS for %s  ' % string, alphas)
        print('LAMBDAS for %s  ' % string, lambdas)
        print('TOLS for %s  ' % string, tols)
        if store_full_path == 1:
            print('full path : ', (string, loss, error_full))
            print('ALPHAS full path : ', (string, alphas_full))
            print('LAMBDAS full path : ', (string, lambdas_full))
            print('TOLS full path : ', (string, tols_full))
        print('Best %s for %s  ' % (loss, string), error_best)
        print('Best ALPHAS for %s  ' % string, alphas_best)
        print('Best LAMBDAS for %s  ' % string, lambdas_best)
        print('Best TOls for %s  ' % string, tols_best)
    return error_best


def run_glm(X,
            y,
            Xtest=None,
            ytest=None,
            nGPUs=0,
            nlambda=100,
            nfolds=5,
            nalpha=5,
            validFraction=0.2,
            family="elasticnet",
            verbose=0,
            print_all_errors=False,
            get_preds=False,
            run_h2o=False,
            tolerance=.01,
            name=None,
            solver="glm",
            lambda_min_ratio=1e-9,
            alphas=None,
            lambdas=None,
            tol=1E-2,
            tol_seek_factor=1E-1):
    """Runs ElasticNetH2O test"""
    #Other default parameters for solving glm
    fit_intercept = True
    lambda_min_ratio = lambda_min_ratio  # Issues for h2o3 with 1k ipums dataset
    nFolds = nfolds
    nLambdas = nlambda
    nAlphas = nalpha
    store_full_path = 1

    print("Doing %s" % (name))
    sys.stdout.flush()
    doassert = 0  # default is not assert

    print("tol=%g tol_seek_factor=%g" % (tol, tol_seek_factor))
    sys.stdout.flush()

    #Override run_h2o False default if environ exists
    if os.getenv("CHECKPERFORMANCE") is not None:
        print("Doing performance testing")
        run_h2o = True
    else:
        print("Not Doing performance testing")

#Setup Train / validation Set Split
    morig = X.shape[0]
    norig = X.shape[1]

    mvalid = 0
    validX = None
    validY = None
    if Xtest is None and ytest is None:
        print("Original m=%d n=%d" % (morig, norig))
        sys.stdout.flush()
        #Do train / valid split
        HO = int(validFraction * morig)
        H = morig - HO
        print("Size of Train rows=%d valid rows=%d" % (H, HO))
        sys.stdout.flush()
        trainX = np.copy(X[0:H, :])
        trainY = np.copy(y[0:H])

        if validFraction != 0.0:
            validX = np.copy(X[H:morig, :])
            validY = np.copy(y[H:morig])
            mvalid = validX.shape[0]
        if mvalid == 0:
            validX = None
            validY = None
            validFraction = 0.0

        mTrain = trainX.shape[0]
        if validFraction != 0.0:
            print("mTrain=%d mvalid=%d validFraction=%g" % (mTrain, mvalid,
                                                            validFraction))
        else:
            print("mTrain=%d" % mTrain)
    else:
        trainX = X
        trainY = y
        validX = Xtest
        validY = ytest
        mvalid = validX.shape[0]
        mTrain = trainX.shape[0]
        if mvalid == 0:
            validX = None
            validY = None
            validFraction = 0.0
        else:
            validFraction = (1.0 * mvalid) / (1.0 * mTrain)
        print("Original m=%d n=%d" % (morig + mvalid, norig))
        sys.stdout.flush()
    print("mTrain=%d mvalid=%d validFraction=%g" % (mTrain, mvalid,
                                                    validFraction))

    #####################
    #
    #Start h2o4gpu
    #
    #####################
    start_h2o4gpu = time()
    print("Setting up solver")
    sys.stdout.flush()

    #######################
    #Choose solver
    if solver == "glm":
        Solver = h2o4gpu.ElasticNetH2O
        enet = Solver(
            n_gpus=nGPUs,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=nLambdas,
            n_folds=nFolds,
            n_alphas=nAlphas,
            verbose=verbose,
            family=family,
            store_full_path=store_full_path,
            alphas=alphas,
            lambdas=lambdas,
            tol=tol,
            tol_seek_factor=tol_seek_factor)
    elif solver == "lasso":
        Solver = h2o4gpu.ElasticNetH2O
        enet = Solver(
            n_gpus=nGPUs,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=nLambdas,
            n_folds=nFolds,
            verbose=verbose,
            family=family,
            store_full_path=store_full_path,
            lambdas=lambdas,
            tol=tol,
            tol_seek_factor=tol_seek_factor,
            alpha_max=1.0,
            alpha_min=1.0,
            n_alphas=1)
    elif solver == "ridge":
        Solver = h2o4gpu.ElasticNetH2O
        enet = Solver(
            n_gpus=nGPUs,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=nLambdas,
            n_folds=nFolds,
            verbose=verbose,
            family=family,
            store_full_path=store_full_path,
            lambdas=lambdas,
            tol=tol,
            tol_seek_factor=tol_seek_factor,
            alpha_max=0.0,
            alpha_min=0.0,
            n_alphas=1)
    elif solver == "linear_regression":
        Solver = h2o4gpu.LinearRegression
        enet = Solver()
    elif solver == "logistic":
        Solver = h2o4gpu.LogisticRegression
        enet = Solver()

    print("trainX")
    print(trainX)
    print("trainY")
    print(trainY)

    #######################
    #Fit
    if validFraction == 0.0:
        print("Solving")
        enet.fit(trainX, trainY)
    else:
        enet.fit(trainX, trainY, validX, validY)

    print("Done Solving")

    X = enet.X
    print("X")
    print(X)

    #Show something about Xvsalphalambda or Xvsalpha
    print("Xvsalpha")
    print(enet.x_vs_alphapure)
    print("np.shape(Xvsalpha)")
    print(np.shape(enet.x_vs_alphapure))

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

    print("All lambdas")
    if enet.store_full_path != 0:
        lambdas = enet.lambdas_full
        print(lambdas)

    print("Time Prepare")
    print(enet.time_prepare)
    print("Time Upload")
    print(enet.time_upload_data)
    print("Time fit only")
    print(enet.time_fitonly)

    assert np.isfinite(enet.X).all()
    if enet.store_full_path != 0:
        assert np.isfinite(enet.X_full).all()

    Xvsalphabest = enet.X_best

    ############## consistency check
    if fit_intercept:
        if validX is not None:
            validX_intercept = np.hstack(
                [validX,
                 np.ones((validX.shape[0], 1), dtype=validX.dtype)])
    else:
        validX_intercept = validX

    if validX is not None:
        testvalidY = np.dot(validX_intercept, Xvsalphabest.T)
        print("testvalidY (newvalidY should be this)")
        if family != "logistic":
            print(testvalidY)
        else:
            try:
                inverse_logit = lambda t: 1 / (1 + math.exp(-t))
                testvalidY = np.round(testvalidY,
                                      1)  # Round to avoid math OverFlow error
                func = np.vectorize(inverse_logit)
                print(func(testvalidY))
            except OverflowError:
                print(testvalidY)

        print(testvalidY)

    #######################
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
        print(
            printallerrors(
                display=1,
                enet=enet,
                string="Train",
                store_full_path=enet.store_full_path))

    from ..solvers.utils import finish
    finish(enet)
    print("Done Reporting")

    duration_h2o4gpu = time() - start_h2o4gpu

    if run_h2o:
        #####################
        #
        #Start h2o
        #
        #####################
        import h2o
        h2o.init(strict_version_check=False)

        print("Build Training H2OFrames")
        trainX_h2o = h2o.H2OFrame(trainX)
        trainY_h2o = h2o.H2OFrame(trainY)
        train_h2o = trainX_h2o.cbind(trainY_h2o)

        if validFraction != 0.0:
            print("Build Validation H2OFrames")
            validX_h2o = h2o.H2OFrame(validX)
            validY_h2o = h2o.H2OFrame(validY)
            valid_h2o = validX_h2o.cbind(validY_h2o)

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f1 = open(os.path.join(path, name + ".error.dat"), 'wt+')
        print('%s' % (name), file=f1, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f1q = open(os.path.join(path, name + ".error.quick.dat"), 'wt+')
        print('%s' % (name), file=f1q, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f1a = open(os.path.join(path, name + ".error.h2o.dat"), 'wt+')
        print('%s' % (name), file=f1a, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f1b = open(os.path.join(path, name + ".error.h2o4gpu.dat"), 'wt+')
        print('%s' % (name), file=f1b, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f2 = open(os.path.join(path, name + ".time.dat"), 'wt+')
        print('%s' % (name), file=f2, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f2a = open(os.path.join(path, name + ".time.h2o.dat"), 'wt+')
        print('%s' % (name), file=f2a, end="")

        path = "./results"
        os.makedirs(path, exist_ok=True)
        f2b = open(os.path.join(path, name + ".time.h2o4gpu.dat"), 'wt+')
        print('%s' % (name), file=f2b, end="")

        start_h2o = time()

        alphas_h2o = [item for alphas[0] in alphas for item in alphas[0]]
        for alpha in alphas_h2o:
            alpha_h2o = alpha.item()  # H2O only takes in python native numeric
            print("Setting up H2O Solver with alpha = %s" % alpha)
            nfoldsh2o = nfolds
            if nfoldsh2o == 1:
                nfoldsh2o = 0
            if family == "logistic":
                from h2o.estimators.glm import H2OGeneralizedLinearEstimator
                h2o_glm = H2OGeneralizedLinearEstimator(
                    intercept=fit_intercept,
                    lambda_search=True,
                    nlambdas=nLambdas,
                    nfolds=nfoldsh2o,
                    family="binomial",
                    alpha=alpha_h2o)
            else:
                from h2o.estimators.glm import H2OGeneralizedLinearEstimator
                h2o_glm = H2OGeneralizedLinearEstimator(
                    intercept=fit_intercept,
                    lambda_search=True,
                    nlambdas=nLambdas,
                    nfolds=nfoldsh2o,
                    family="gaussian",
                    alpha=alpha_h2o)
#Solve
            if validFraction == 0.0:
                print("Solving using H2O")
                h2o_glm.train(
                    x=train_h2o.columns[:-1],
                    y=train_h2o.columns[-1],
                    training_frame=train_h2o)
            else:
                print("Solving using H2O")
                h2o_glm.train(
                    x=train_h2o.columns[:-1],
                    y=train_h2o.columns[-1],
                    training_frame=train_h2o,
                    validation_frame=valid_h2o)
            print("\nComparing results to H2O")
            print("\nH2O ElasticNetH2O Summary")
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
                    print(h2o_glm.model_performance(xval=True).logloss())
                    print("\n")
                    h2o_cv_error = h2o_glm.model_performance(
                        xval=True).logloss()

                else:
                    print("\nCross Validation RMSE")
                    print(h2o_glm.model_performance(xval=True).rmse())
                    print("\n")
                    h2o_cv_error = h2o_glm.model_performance(xval=True).rmse()

            NUM_ERRORS = 3
            which_errors = [False] * NUM_ERRORS
            #Train and nfolds
            if validFraction == 0.0 and nfolds > 1:
                which_errors[0] = True
                which_errors[1] = True
#Train, valid, and nfolds
            elif validFraction > 0.0 and nfolds > 1:
                which_errors[0] = True
                which_errors[1] = True
                which_errors[2] = True
#Train set only
            else:
                which_errors[0] = True

#Compare to H2O
            index = alphas_h2o.index(alpha)
            for j in range(0, NUM_ERRORS):
                if j == 0 and which_errors[j]:  # Compare to train error
                    thisrelerror = -(
                        error_train[index, j] - h2o_train_error) / (
                            abs(error_train[index, j]) + abs(h2o_train_error))
                    if error_train[index, j] > h2o_train_error:
                        if abs(thisrelerror) > tolerance:
                            print("Train error failure: %g %g" %
                                  (error_train[index, j], h2o_train_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                        else:
                            print(' OK', file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                    else:
                        print(
                            "H2O Train Error is larger than GPU ElasticNetH2O "
                            "with alpha = %s" % alpha)
                        print("H2O Train Error is %s" % h2o_train_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' GOOD', file=f1, end="")
                        print(' %g' % thisrelerror, file=f1q, end="")
                        print(' %g' % h2o_train_error, file=f1a, end="")
                        print(' %g' % error_train[index, j], file=f1b, end="")
                elif j == 1 and which_errors[j]:  # Compare to average cv error
                    thisrelerror = -(
                        error_train[index, j] - h2o_train_error) / (
                            abs(error_train[index, j]) + abs(h2o_cv_error))
                    if error_train[index, j] > h2o_cv_error:
                        if abs(thisrelerror) > tolerance:
                            print("CV error failure: %g %g" %
                                  (error_train[index, j], h2o_cv_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                        else:
                            print(' OK', file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                    else:
                        print("H2O CV Error is larger than GPU ElasticNetH2O "
                              "with alpha = %s" % alpha)
                        print("H2O CV Error is %s" % h2o_cv_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' GOOD', file=f1, end="")
                        print(' %g' % thisrelerror, file=f1q, end="")
                        print(' %g' % h2o_train_error, file=f1a, end="")
                        print(' %g' % error_train[index, j], file=f1b, end="")
                elif j == 2 and which_errors[j]:  # Compare to validation error
                    thisrelerror = -(
                        error_train[index, j] - h2o_train_error) / (
                            abs(error_train[index, j]) + abs(h2o_valid_error))
                    if error_train[index, j] > h2o_valid_error:
                        if abs(thisrelerror) > tolerance:
                            print("Valid error failure: %g %g" %
                                  (error_train[index, j], h2o_valid_error))
                            doassert = 1
                            print(' %g' % thisrelerror, file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                        else:
                            print(' OK', file=f1, end="")
                            print(' %g' % thisrelerror, file=f1q, end="")
                            print(' %g' % h2o_train_error, file=f1a, end="")
                            print(
                                ' %g' % error_train[index, j], file=f1b, end="")
                    else:
                        print(
                            "H2O Valid Error is larger than GPU ElasticNetH2O "
                            "with alpha = %s" % alpha)
                        print("H2O Valid Error is %s" % h2o_valid_error)
                        print("H2O GPU ML Error is %s" % error_train[index, j])
                        print(' GOOD', file=f1, end="")
                        print(' %g' % thisrelerror, file=f1q, end="")
                        print(' %g' % h2o_train_error, file=f1a, end="")
                        print(' %g' % error_train[index, j], file=f1b, end="")
                else:
                    print(' NA', file=f1, end="")
                    print(' %g' % thisrelerror, file=f1q, end="")
                    print(' %g' % h2o_train_error, file=f1a, end="")
                    print(' %g' % error_train[index, j], file=f1b, end="")

        print('', file=f1)
        f1.flush()
        print('', file=f1q)
        f1q.flush()
        print('', file=f1a)
        f1a.flush()
        print('', file=f1b)
        f1b.flush()

        #time entire alpha - lambda path
        duration_h2o = time() - start_h2o

        ratio_time = duration_h2o4gpu / duration_h2o
        print(' %g' % ratio_time, file=f2, end="")
        print('', file=f2)
        f2.flush()

        print(' %g' % duration_h2o, file=f2a, end="")
        print('', file=f2a)
        f2a.flush()

        print(' %g' % duration_h2o4gpu, file=f2b, end="")
        print('', file=f2b)
        f2b.flush()

        #include asserts for timing

        #for pytest only:
        if os.getenv("DISABLEPYTEST") is None:
            assert doassert == 0

    if len(np.shape(error_train)) == 2:
        myerror_train = error_train
        myerror_test = error_test
    if len(np.shape(error_train)) == 3:
        myerror_train = error_train[-1]
        myerror_test = error_test[-1]

    return myerror_train, myerror_test


# Animation stuff


def new_alpha(row_fold):
    if row_fold == 0:
        return -0.025
    elif row_fold == 1:
        return -0.05
    elif row_fold == 3:
        return 0.025
    elif row_fold == 4:
        return 0.05
    else:
        return 0


def plot_cpu_perf(axis, cpu_labels, cpu_snapshot):
    axis.cla()
    axis.grid(False)
    axis.set_ylim([0, 100])
    axis.set_ylabel('Percent', labelpad=2, fontsize=14)
    axis.bar(cpu_labels, cpu_snapshot, color='dodgerblue', edgecolor='none')
    axis.set_title('CPU Utilization', fontsize=16)


def plot_gpu_perf(axis, gpu_labels, gpu_snapshot):
    axis.cla()
    axis.grid(False)
    axis.set_ylim([0, 100])
    axis.set_xticks(gpu_labels)
    axis.set_ylabel('Percent', labelpad=2, fontsize=14)
    axis.bar(
        gpu_labels,
        gpu_snapshot,
        width=0.5,
        color='limegreen',
        align='center',
        edgecolor='none')
    axis.set_title('GPU Utilization', fontsize=16)


def plot_glm_results(axis, results, best_rmse, cb):
    axis.cla()
    axis.set_xscale('log')
    axis.set_xlim([1e2, 1e9])
    axis.set_ylim([-0.12, 1.12])
    axis.set_yticks([x / 7. for x in range(0, 8)])
    axis.set_ylabel('Parameter 1:  ' + r'$\alpha$', fontsize=16)
    axis.set_xlabel('Parameter 2:  ' + r'$\lambda$', fontsize=16)
    num_models = min(4000, int(4000 * results.shape[0] / 2570))
    axis.set_title(
        'Elastic Net Models Trained and Evaluated: ' + str(num_models),
        fontsize=16)

    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        import pylab as pl
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(sns.color_palette("RdYlGn", 10).as_hex())
        cf = axis.scatter(
            results['lambda'],
            results['alpha_prime'],
            c=results['rel_acc'],
            cmap=cm,
            vmin=0,
            vmax=1,
            s=60,
            lw=0)
        axis.plot(
            best_rmse['lambda'],
            best_rmse['alpha_prime'],
            'o',
            ms=15,
            mec='k',
            mfc='none',
            mew=2)

        if not cb:
            cb = pl.colorbar(cf, ax=axis)
            cb.set_label(
                'Relative  Validation  Accuracy',
                rotation=270,
                labelpad=18,
                fontsize=16)
        cb.update_normal(cf)
    except:
        #print("plot_glm_results exception -- no frame")
        pass


def RunAnimation(arg):
    import os, sys, time
    import subprocess
    import psutil
    import pylab as pl
    from IPython import display
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import pandas as pd
    import numpy as np

    print("RunAnimation")
    sys.stdout.flush()

    deviceCount = arg
    # Need this only for animation of GPU usage to be consistent with
    #from py3nvml.py3nvml import *
    import py3nvml
    maxNGPUS = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
    print("\nNumber of GPUS:", maxNGPUS)

    py3nvml.py3nvml.nvmlInit()
    total_deviceCount = py3nvml.py3nvml.nvmlDeviceGetCount()
    if deviceCount == -1:
        deviceCount = total_deviceCount
    #for i in range(deviceCount):
    #    handle = nvmlDeviceGetHandleByIndex(i)
    #    print("Device {}: {}".format(i, nvmlDeviceGetName(handle)))
    #print ("Driver Version:", nvmlSystemGetDriverVersion())
    print("Animation deviceCount=%d" % (deviceCount))

    file = os.getcwd() + "/error.txt"
    print("opening %s" % (file))
    fig = pl.figure(figsize=(9, 9))
    pl.rcParams['xtick.labelsize'] = 14
    pl.rcParams['ytick.labelsize'] = 14
    gs = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.4)
    ax1 = pl.subplot(gs[0, -2])
    ax2 = pl.subplot(gs[0, 1])
    ax3 = pl.subplot(gs[1:, :])
    fig.suptitle(
        'H2O.ai Machine Learning $-$ Generalized Linear Modeling', size=18)

    pl.gcf().subplots_adjust(bottom=0.2)

    #cb = False
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(sns.color_palette("RdYlGn", 10).as_hex())
    cc = ax3.scatter([0.001, 0.001], [0, 0], c=[0, 1], cmap=cm)
    cb = pl.colorbar(cc, ax=ax3)
    os.system("mkdir -p images")
    i = 0
    while (True):
        #try:
        #print("In try i=%d" % i)
        #sys.stdout.flush()

        #cpu
        snapshot = psutil.cpu_percent(percpu=True)
        cpu_labels = range(1, len(snapshot) + 1)
        plot_cpu_perf(ax1, cpu_labels, snapshot)

        #gpu
        gpu_snapshot = []
        gpu_labels = list(range(1, deviceCount + 1))
        import py3nvml
        for j in range(deviceCount):
            handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(j)
            util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_snapshot.append(util.gpu)
        gpu_snapshot = gpu_snapshot
        plot_gpu_perf(ax2, gpu_labels, gpu_snapshot)

        res = pd.read_csv(
            file,
            sep="\s+",
            header=None,
            names=[
                'time', 'pass', 'fold', 'a', 'i', 'alpha', 'lambda',
                'trainrmse', 'ivalidrmse', 'validrmse'
            ])

        res['rel_acc'] = ((42665 - res['validrmse']) / (42665 - 31000))
        res['alpha_prime'] = res['alpha'] + res['fold'].apply(
            lambda x: new_alpha(x))

        best = res.loc[res['rel_acc'] == np.max(res['rel_acc']), :]
        plot_glm_results(ax3, res, best.tail(1), cb)
        # flag for colorbar to avoid redrawing
        #cb = True

        # Add footnotes
        footnote_text = "*U.S. Census dataset (predict Income): 45k rows, 10k cols\nParameters: 5-fold cross-validation, " + r'$\alpha = \{\frac{i}{7},i=0\ldots7\}$' + ", "\
'full $\lambda$-' + "search"
        #pl.figtext(.05, -.04, footnote_text, fontsize = 14,)
        pl.annotate(
            footnote_text, (0, 0), (-30, -50),
            fontsize=12,
            xycoords='axes fraction',
            textcoords='offset points',
            va='top')

        #update the graphics
        display.display(pl.gcf())
        display.clear_output(wait=True)
        time.sleep(0.01)

        #save the images
        saveimage = 0
        if saveimage:
            file_name = './images/glm_run_%04d.png' % (i,)
            pl.savefig(file_name, dpi=200)
        i = i + 1

    #except KeyboardInterrupt:
    #    break
    #except:
    #    #print("Could not Create Frame")
    #    pass


def RunH2Oaiglm(arg):
    import h2o4gpu as h2o4gpu
    import time

    trainX, trainY, validX, validY, family, intercept, lambda_min_ratio, n_folds, n_alphas, n_lambdas, n_gpus = arg

    # assume ok with 32-bit float for speed on GPU if using this wrapper
    if trainX is not None:
        trainX.astype(np.float32)
    if trainY is not None:
        trainY.astype(np.float32)
    if validX is not None:
        validX.astype(np.float32)
    if validY is not None:
        validY.astype(np.float32)

    print("Begin Setting up Solver")
    os.system(
        "rm -f error.txt ; touch error.txt ; rm -f varimp.txt ; touch varimp.txt"
    )  ## for visualization
    enet = h2o4gpu.ElasticNetH2O(
        n_gpus=n_gpus,
        fit_intercept=intercept,
        lambda_min_ratio=lambda_min_ratio,
        n_lambdas=n_lambdas,
        n_folds=n_folds,
        n_alphas=n_alphas,
        family=family)
    print("End Setting up Solver")

    # Solve
    print("Begin Solving")
    t0 = time.time()
    enet.fit(trainX, trainY, validX, validY)
    t1 = time.time()
    print("End Solving")

    print("Time to train H2O AI ElasticNetH2O: %r" % (t1 - t0))


def RunH2Oaiglm_ptr(arg):
    import h2o4gpu as h2o4gpu
    import time

    trainX, trainY, validX, validY, trainW, fortran, mTrain, n, mvalid, intercept, lambda_min_ratio, n_folds, n_alphas, n_lambdas, n_gpus = arg

    # assume ok with 32-bit float for speed on GPU if using this wrapper
    if trainX is not None:
        trainX.astype(np.float32)
    if trainY is not None:
        trainY.astype(np.float32)
    if validX is not None:
        validX.astype(np.float32)
    if validY is not None:
        validY.astype(np.float32)
    if trainW is not None:
        trainW.astype(np.float32)

    print("Begin Setting up Solver")
    os.system(
        "rm -f error.txt ; touch error.txt ; rm -f varimp.txt ; touch varimp.txt"
    )  ## for visualization
    enet = h2o4gpu.ElasticNetH2O(
        n_gpus=n_gpus,
        fit_intercept=intercept,
        lambda_min_ratio=lambda_min_ratio,
        n_lambdas=n_lambdas,
        n_folds=n_folds,
        n_alphas=n_alphas)
    print("End Setting up Solver")

    ## First, get backend pointers
    sourceDev = 0
    t0 = time.time()
    a, b, c, d, e = enet.prepare_and_upload_data(
        trainX, trainY, validX, validY, trainW, source_dev=sourceDev)
    t1 = time.time()
    print("Time to ingest data: %r" % (t1 - t0))

    ## Solve
    if 1 == 1:
        print("Solving")
        t0 = time.time()
        order = 'c' if fortran else 'r'
        double_precision = 0  # Not used
        store_full_path = 0
        enet.fit_ptr(
            mTrain,
            n,
            mvalid,
            double_precision,
            order,
            a,
            b,
            c,
            d,
            e,
            source_dev=sourceDev)
        t1 = time.time()
        print("Done Solving")
        print("Time to train H2O AI ElasticNetH2O: %r" % (t1 - t0))
