import time
import sys
import os
import logging

print(sys.path)

try:
    from utils import find_file, run_glm
except:
    from tests.utils import find_file, run_glm

logging.basicConfig(level=logging.DEBUG)


def fun(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, whichdata=0):
    name = str(sys._getframe().f_code.co_name)
    t = time.time()

    print("cwd: %s" % (os.getcwd()))
    sys.stdout.flush()

    name = sys._getframe(1).f_code.co_name
    #    pipes = startfunnel(os.path.join(os.getcwd(), "tmp/"), name)

    print("Getting Data")
    from h2o4gpu.datasets import fetch_20newsgroups,    fetch_20newsgroups_vectorized,    fetch_california_housing,    fetch_covtype,    fetch_kddcup99,    fetch_lfw_pairs,    fetch_lfw_people,    fetch_mldata,    fetch_olivetti_faces,    fetch_rcv1,    fetch_species_distributions
    from h2o4gpu.model_selection import train_test_split

    # Fetch dataset
    if whichdata == 0:
        data = fetch_20newsgroups()
    elif whichdata == 1:
        data = fetch_20newsgroups_vectorized()
    elif whichdata == 2:
        data = fetch_california_housing()
    elif whichdata == 3:
        data = fetch_covtype()
    elif whichdata == 4:
        data = fetch_kddcup99()
    elif whichdata == 5:
        data = fetch_lfw_pairs()
    elif whichdata == 6:
        data = fetch_lfw_people()
    elif whichdata == 7:
        data = fetch_mldata('iris')
    elif whichdata == 8:
        data = fetch_mldata('leukemia')
    elif whichdata == 9:
        data = fetch_mldata('Whistler Daily Snowfall')
    elif whichdata == 10:
        data = fetch_olivetti_faces()
    elif whichdata == 11:
        data = fetch_rcv1()
    elif whichdata == 12:
        data = fetch_species_distributions()
    else:
        ValueError("No such whichdata")

    X = data.data
    y = data.target
    print("Got Data")

    # Create 0.8/0.2 train/test split
    print("Split Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8,
                                        random_state=42)

    print("Encode Data")
    #from h2o4gpu.preprocessing import Imputer
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #imp.fit(X, y)
    #Xencoded = imp.transform(X)
    #yencoded = imp.transform(y)

    import pandas as pd
    X_test_pd = pd.DataFrame(X_test)
    X_train_pd = pd.DataFrame(X_train)

    # Importing LabelEncoder and initializing it
    from h2o4gpu.preprocessing import LabelEncoder
    le=LabelEncoder()
    # Iterating over all the common columns in train and test
    for col in X_test_pd.columns.values:
        # Encoding only categorical variables
        if X_test_pd[col].dtypes=='object' or X_test_pd[col].dtypes=='bool':
            # Using whole data to form an exhaustive list of levels
            data=X_train_pd[col].append(X_test_pd[col])
            le.fit(data.values)
            X_train_pd[col]=le.transform(X_train_pd[col])
            X_test_pd[col]=le.transform(X_test_pd[col])

    # get back numpy
    X_test = X_test_pd.values
    X_train = X_train_pd.values

    # TODO: Should write this to file and avoid doing encoding if already exists

    t1 = time.time()
    print("Start GLM")
    rmse_train, rmse_test = run_glm(X_train, y_train, X_test, y_test, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds, nalpha=nAlphas,
                                        validFraction=validFraction, verbose=0, name=name)
    print("End GLM")

    # check rmse
    print(rmse_train[0, 0])
    print(rmse_train[0, 1])
    print(rmse_train[0, 2])
    print(rmse_test[0, 2])
    sys.stdout.flush()

    print('/n Total execution time:%d' % (time.time() - t1))

    print("TEST PASSED")
    sys.stdout.flush()

    print("Time taken: {}".format(time.time() - t))
    #    endfunnel(pipes)
    print("DONE.")
    sys.stdout.flush()

def test_glm_sklearn_gpu_data0(): fun(whichdata=0)
def test_glm_sklearn_gpu_data1(): fun(whichdata=1)
def test_glm_sklearn_gpu_data2(): fun(whichdata=2)
def test_glm_sklearn_gpu_data3(): fun(whichdata=3)
def test_glm_sklearn_gpu_data4(): fun(whichdata=4)
def test_glm_sklearn_gpu_data5(): fun(whichdata=5)
def test_glm_sklearn_gpu_data6(): fun(whichdata=6)
def test_glm_sklearn_gpu_data7(): fun(whichdata=7)
def test_glm_sklearn_gpu_data8(): fun(whichdata=8)
def test_glm_sklearn_gpu_data9(): fun(whichdata=9)
def test_glm_sklearn_gpu_data10(): fun(whichdata=10)
def test_glm_sklearn_gpu_data11(): fun(whichdata=11)
def test_glm_sklearn_gpu_data12(): fun(whichdata=12)



if __name__ == '__main__':
    test_glm_sklearn_gpu_data0()
    test_glm_sklearn_gpu_data1()
    test_glm_sklearn_gpu_data2()
    test_glm_sklearn_gpu_data3()
    test_glm_sklearn_gpu_data4()
    test_glm_sklearn_gpu_data5()
    test_glm_sklearn_gpu_data6()
    test_glm_sklearn_gpu_data7()
    test_glm_sklearn_gpu_data8()
    test_glm_sklearn_gpu_data9()
    test_glm_sklearn_gpu_data10()
    test_glm_sklearn_gpu_data11()
    test_glm_sklearn_gpu_data12()