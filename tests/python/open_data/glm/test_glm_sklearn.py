# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using SKLearn datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import logging
import pytest

from h2o4gpu.util.testing_utils import find_file, run_glm
logging.basicConfig(level=logging.DEBUG)


class TestGlmSklearn(object):
    @classmethod
    def setup_class(cls):
        os.environ['SCIKIT_LEARN_DATA'] = "open_data"

    @staticmethod
    def fun(nGPUs=1, nFolds=1, nLambdas=100, nAlphas=8, validFraction=0.2, whichdata=0, double_precision=False):
        name = str(sys._getframe().f_code.co_name)
        t = time.time()

        print("cwd: %s" % (os.getcwd()))
        sys.stdout.flush()

        name = sys._getframe(1).f_code.co_name
        #    pipes = startfunnel(os.path.join(os.getcwd(), "tmp/"), name)

        print("Getting Data")
        from h2o4gpu.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_california_housing, \
            fetch_covtype, fetch_kddcup99, fetch_lfw_pairs, fetch_lfw_people, fetch_openml, fetch_olivetti_faces, \
            fetch_rcv1, fetch_species_distributions
        from h2o4gpu.model_selection import train_test_split

        # Fetch dataset
        if whichdata == 0:
            data = fetch_20newsgroups()  # runs
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 1:
            data = fetch_20newsgroups_vectorized()  # sparse
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 2:
            data = fetch_california_housing()  # runs
            sizetokeep = 1000
        elif whichdata == 3:
            data = fetch_covtype()
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 4:
            data = fetch_kddcup99()  # strings -> numeric
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 5:
            data = fetch_lfw_pairs()
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 6:
            data = fetch_lfw_people()
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 7:
            data = fetch_openml('iris')
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 8:
            data = fetch_openml('leukemia')  # runs
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 9:
            data = fetch_openml('Whistler Daily Snowfall')
            sizetokeep = 1000  # 1k rows for now
        elif whichdata == 10:
            data = fetch_olivetti_faces()  # runs
            sizetokeep = 100
        elif whichdata == 11:
            data = fetch_rcv1()
            sizetokeep = 1000  # 1k rows for now
            # data = data.todense() # FIXME: glm and kmeans h2o4gpu currently only supports dense matrices
        elif whichdata == 12:
            data = fetch_species_distributions()
            sizetokeep = 1000  # 1k rows for now
        else:
            ValueError("No such whichdata")

        try:
            sizetokeep = min(sizetokeep, len(data.data[:, 0]))
            X = data.data[0:sizetokeep, :]
        except:
            sizetokeep = min(sizetokeep, len(data.data[:]))
            X = data.data[0:sizetokeep]
        y = data.target[0:sizetokeep]
        print("Got Data")

        import numpy as np

        # Create 0.8/0.2 train/test split
        print("Split Data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8,
                                                            random_state=42)

        print("Encode Data")
        # from h2o4gpu.preprocessing import Imputer
        # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        # imp.fit(X, y)
        # Xencoded = imp.transform(X)
        # yencoded = imp.transform(y)

        import pandas as pd
        X_test_pd = pd.DataFrame(X_test)
        X_train_pd = pd.DataFrame(X_train)

        # Importing LabelEncoder and initializing it
        from h2o4gpu.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Iterating over all the common columns in train and test
        for col in X_test_pd.columns.values:
            # Encoding only categorical variables
            if X_test_pd[col].dtypes == 'object' or X_test_pd[col].dtypes == 'bool':
                # Using whole data to form an exhaustive list of levels
                data = X_train_pd[col].append(X_test_pd[col])
                le.fit(data.values)
                X_train_pd[col] = le.transform(X_train_pd[col])
                X_test_pd[col] = le.transform(X_test_pd[col])

        X_train_pd = pd.get_dummies(X_train_pd).fillna(0.0)
        X_test_pd = pd.get_dummies(X_test_pd).fillna(0.0)
        y_train_pd = pd.Series(y_train).fillna(0.0)
        y_test_pd = pd.Series(y_test).fillna(0.0)

        # get back numpy
        X_test = X_test_pd.values
        X_train = X_train_pd.values
        y_test = y_test_pd.values
        y_train = y_train_pd.values

        if double_precision:
            mynptype = np.float64
        else:
            mynptype = np.float32
        X_test = X_test.astype(mynptype)
        X_train = X_train.astype(np.float64)
        y_test = y_test.astype(np.float64)
        y_train = y_train.astype(np.float64)

        # TODO: Should write this to file and avoid doing encoding if already exists

        t1 = time.time()
        print("Start ElasticNetH2O")
        rmse_train, rmse_test = run_glm(X_train, y_train, X_test, y_test, nGPUs=nGPUs, nlambda=nLambdas, nfolds=nFolds,
                                        nalpha=nAlphas,
                                        validFraction=validFraction, verbose=0, name=name, tolerance=0.2, tol=1E-2, tol_seek_factor=1.0)
        print("End ElasticNetH2O")

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

    def test_glm_sklearn_gpu_data0(self): TestGlmSklearn. fun(whichdata=0)

    @pytest.mark.skip("No direct handling of compressed data yet (can convert, but want to add this feature later)")
    def test_glm_sklearn_gpu_data1(self): TestGlmSklearn. fun(whichdata=1)

    # data2 test takes excessive iterations per glm model (i.e. per alpha/lambda).
    # Perhaps adapative rho improvements will help.
    @pytest.mark.skip("Too slow for testing, but should figure out why slow.")
    def test_glm_sklearn_gpu_data2(self): TestGlmSklearn. fun(whichdata=2)

    @pytest.mark.skip("Unknown")
    def test_glm_sklearn_gpu_data3(self): TestGlmSklearn. fun(whichdata=3)

    @pytest.mark.skip("No multi-class yet")
    def test_glm_sklearn_gpu_data4(self): TestGlmSklearn. fun(whichdata=4)

    @pytest.mark.skip("Currently response is constant, h2o-3 complains")
    def test_glm_sklearn_gpu_data5(self): TestGlmSklearn. fun(whichdata=5)

    def test_glm_sklearn_gpu_data6(self): TestGlmSklearn. fun(whichdata=6)

    def test_glm_sklearn_gpu_data7(self): TestGlmSklearn. fun(whichdata=7)

    def test_glm_sklearn_gpu_data8(self): TestGlmSklearn. fun(whichdata=8)

    @pytest.mark.skip("Unknown")
    def test_glm_sklearn_gpu_data9(self): TestGlmSklearn. fun(whichdata=9)

    def test_glm_sklearn_gpu_data10(self): TestGlmSklearn. fun(whichdata=10)

    @pytest.mark.skip("No direct handling of compressed data yet (can convert, but want to add this feature later)")
    def test_glm_sklearn_gpu_data11(self): TestGlmSklearn. fun(whichdata=11)

    @pytest.mark.skip("Unknown")
    def test_glm_sklearn_gpu_data12(self): TestGlmSklearn. fun(whichdata=12)


if __name__ == '__main__':
    TestGlmSklearn.test_glm_sklearn_gpu_data0()
    TestGlmSklearn.test_glm_sklearn_gpu_data1()
    TestGlmSklearn.test_glm_sklearn_gpu_data2()
    TestGlmSklearn.test_glm_sklearn_gpu_data3()
    TestGlmSklearn.test_glm_sklearn_gpu_data4()
    TestGlmSklearn.test_glm_sklearn_gpu_data5()
    TestGlmSklearn.test_glm_sklearn_gpu_data6()
    TestGlmSklearn.test_glm_sklearn_gpu_data7()
    TestGlmSklearn.test_glm_sklearn_gpu_data8()
    TestGlmSklearn.test_glm_sklearn_gpu_data9()
    TestGlmSklearn.test_glm_sklearn_gpu_data10()
    TestGlmSklearn.test_glm_sklearn_gpu_data11()
    TestGlmSklearn.test_glm_sklearn_gpu_data12()
