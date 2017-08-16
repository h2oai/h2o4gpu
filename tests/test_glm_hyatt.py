import time
import copy
import sys
import os
import numpy as np
import pandas as pd
import logging


print(sys.path)

try:
    from utils import find_file, runglm
except:
    from tests.utils import find_file, runglm

logging.basicConfig(level=logging.DEBUG)


def fun(use_gpu=False, nFolds=1, nLambdas=100, nAlphas=8, classification=False, use_seed=True):
    name = str(sys._getframe().f_code.co_name)
    name = str(sys._getframe(1).f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()
    
    print("Reading Data")
    if 1==0: # not yet
        target=None
        import datatable as dt # omp problem in pycharm
        train = find_file("xtrain_1k.txt")
        test = find_file("xtest_1k.txt")

        train = os.path.normpath(os.path.join(os.getcwd(), train))
        train_df = dt.fread(train).topandas()
        train_df = train_df[pd.notnull(train_df[target])].reset_index(drop=True)  # drop rows with NA response
        
        test = os.path.normpath(os.path.join(os.getcwd(), test))
        test_df = dt.fread(test).topandas()
        test_df = test_df[pd.notnull(test_df[target])].reset_index(drop=True)  # drop rows with NA response
        
        y = train_df[target]
        
        df_before = copy.deepcopy(train_df)
    
        classes = 1 if not classification else len(y.unique())
        print("Testing GLM for " + ((str(classes) + "-class classification") if classes >= 2 else "regression"))
    else:
        # should all be explicitly np.float32 or all np.float64
        xtrain = np.loadtxt("./data/xtrainhyatt_1k.csv", delimiter=',', dtype=np.float32)
        ytrain = np.loadtxt("./data/ytrainhyatt_1k.csv", delimiter=',', dtype=np.float32)
        xtest = np.loadtxt("./data/xtesthyatt_1k.csv", delimiter=',', dtype=np.float32)
        ytest = np.loadtxt("./data/ytesthyatt_1k.csv", delimiter=',', dtype=np.float32)
        wtrain = np.ones((xtrain.shape[0], 1), dtype=np.float32)
        print("Testing GLM")

    # seed = np.random.randint(0, 2 ** 31 - 1)
    seed = 1034753 if use_seed else None


    display = 1
    t1 = time.time()
    write=1
    pred_val, rmse_train, rmse_test = runglm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, use_gpu, name=name)

    # check rmse
    print(rmse_train[0,0])
    print(rmse_train[0,1])
    print(rmse_train[0,2])
    print(rmse_test[0,2])
    sys.stdout.flush()

    # FIXME: But these below should really be order 1 to 1.5 according to Wamsi!
    assert rmse_train[0,0]<14
    assert rmse_train[0,1]<14
    assert rmse_train[0,2]<21
    assert rmse_test[0,2]<21

    print('/n Total execution time:%d' % (time.time() - t1))


    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))
#    endfunnel(pipes)
    print("DONE.")
    sys.stdout.flush()

def test_glm_hyatt_gpu_fold1_quick(): fun(use_gpu=True, nFolds=1, nLambdas=5, nAlphas=3, classification=False)

def test_glm_hyatt_gpu_fold1(): fun(use_gpu=True, nFolds=1, nLambdas=100, nAlphas=8, classification=False)

def test_glm_hyatt_gpu_fold5(): fun(use_gpu=True, nFolds=5, nLambdas=100, nAlphas=3, classification=False)


# def test_glm_hyatt_cpu_fold1_quick(): fun(use_gpu=False, nFolds=1, nLambdas=5, nAlphas=3, classification=False)
#
# def test_glm_hyatt_cpu_fold1(): fun(use_gpu=False, nFolds=1, nLambdas=10, nAlphas=8, classification=False)
#
# def test_glm_hyatt_cpu_fold5(): fun(use_gpu=False, nFolds=5, nLambdas=10, nAlphas=3, classification=False)


if __name__ == '__main__':
	test_glm_hyatt_gpu_fold1_quick()
	test_glm_hyatt_gpu_fold1()
	test_glm_hyatt_gpu_fold5()

	# test_glm_hyatt_cpu_fold1_quick()
	# test_glm_hyatt_cpu_fold1()
	# test_glm_hyatt_cpu_fold5()
