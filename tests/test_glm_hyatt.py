import datatable as dt
import time
import copy
import sys
import os
import numpy
import pandas as pd
import logging

from tests.utils import find_file, runglm

logging.basicConfig(level=logging.DEBUG)


def fun(whichtest=None, classification=False):
    name = sys._getframe(1).f_code.co_name
    #    pipes = startfunnel(os.path.join(os.getcwd(), "tmp/"), name)

    if 1==0: # not yet
        train = find_file("xtrain.txt")
        test = find_file("xtest.txt")

        train = os.path.normpath(os.path.join(os.getcwd(), train))
        train_df = dt.fread(train).topandas()
        train_df = train_df[pd.notnull(train_df[target])].reset_index(drop=True)  # drop rows with NA response
        
        test = os.path.normpath(os.path.join(os.getcwd(), test))
        test_df = dt.fread(test).topandas()
        test_df = test_df[pd.notnull(test_df[target])].reset_index(drop=True)  # drop rows with NA response
        
        t = time.time()
        y = train_df[target]
        
    classes = 1 if not classification else len(y.unique())

    # seed = np.random.randint(0, 2 ** 31 - 1)
    seed = 1034753 if use_seed else None

    df_before = copy.deepcopy(train_df)
    print("Testing GLM for " + ((str(classes) + "-class classification") if classes >= 2 else "regression"))

    xtrain = np.loadtxt("../data/xtrain.csv", delimiter=',', dtype=np.float32)
    ytrain = np.loadtxt("../data/ytrain.csv", delimiter=',', dtype=np.float32)
    xtest = np.loadtxt("../data/xtest.csv", delimiter=',', dtype=np.float32)
    ytest = np.loadtxt("../data/ytest.csv", delimiter=',', dtype=np.float32)
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
    runglm(nFolds, nAlphas, nLambdas, xtrain, ytrain, xtest, ytest, wtrain, write, display, use_gpu)
    print('/n Total execution time:%d' % (time() - t1))

    print("TEST PASSED")

    print("Time taken: {}".format(time.time() - t))
#    endfunnel(pipes)
    print("DONE.")


def test_glm_hyatt1(): fun(1, classification=False)

def test_glm_hyatt2(): fun(2, classification=False)

if __name__ == '__main__':
    test_glm_hyatt1()
    test_glm_hyatt2()
    
