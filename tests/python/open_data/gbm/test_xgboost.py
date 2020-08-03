# -*- encoding: utf-8 -*-
import pytest
import platform
"""
XGBoost solver tests using Kaggle datasets.

:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import time
import sys
import os
import logging
import fcntl
import os

print(sys.path)
print(sys.getdefaultencoding())

logging.basicConfig(level=logging.DEBUG)


def fetch_data():
    from sklearn.datasets import fetch_covtype
    import fcntl
    with open("sklearn_download.lock", mode="ab") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        data = fetch_covtype()
        fcntl.lockf(f, fcntl.LOCK_UN)
        return data


@pytest.mark.parametrize("n_gpus", [0, 1])
def test_xgboost_covtype(n_gpus):
    import xgboost as xgb
    import numpy as np
    from sklearn.model_selection import train_test_split
    import time

    # Fetch dataset using sklearn
    cov = fetch_data()
    X = cov.data
    y = cov.target

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                        random_state=42)

    # Specify sufficient boosting iterations to reach a minimum
    num_round = 10

    # Leave most parameters as default
    param = {'objective': 'multi:softmax',  # Specify multiclass classification
             'num_class': 8,  # Number of possible output classes
             'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
             }
    if n_gpus is not None:
        param['n_gpus'] = n_gpus

    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)

    gpu_res = {}  # Store accuracy result
    tmp = time.time()
    # Train model
    xgb.train(param, dtrain, num_round, evals=[
              (dtest, 'test')], evals_result=gpu_res)
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

    # TODO: https://github.com/dmlc/xgboost/issues/4518
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
    # Repeat for CPU algorithm
    tmp = time.time()
    param['tree_method'] = 'hist'
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[
              (dtest, 'test')], evals_result=cpu_res)
    print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))


@pytest.mark.multi_gpu
def test_xgboost_covtype_multi_gpu():
    import xgboost as xgb
    import numpy as np
    from sklearn.model_selection import train_test_split
    import time
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from dask import array as da
    import xgboost as xgb
    from xgboost.dask import DaskDMatrix
    from dask import array as da

    # Fetch dataset using sklearn
    cov = fetch_data()
    X = cov.data
    y = cov.target

    print(X.shape, y.shape)

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                        random_state=42)

    # Specify sufficient boosting iterations to reach a minimum
    num_round = 10

    # Leave most parameters as default
    param = {'objective': 'multi:softmax',  # Specify multiclass classification
             'num_class': 8,  # Number of possible output classes
             'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
             }

    from h2o4gpu.util.gpu import device_count
    n_gpus, devices = device_count(-1)

    with LocalCUDACluster(n_workers=n_gpus, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            # Convert input data from numpy to XGBoost format
            partition_size = 100000

            # remove when https://github.com/dmlc/xgboost/issues/4987 is fixed
            dask_X_train = da.from_array(X_train, partition_size)
            dask_X_train = dask_X_train.persist()
            client.rebalance(dask_X_train)
            dask_label_train = da.from_array(y_train, partition_size)
            dask_label_train = dask_label_train.persist()
            client.rebalance(dask_label_train)

            dtrain = DaskDMatrix(
                client=client, data=dask_X_train, label=dask_label_train)

            dask_X_test = da.from_array(X_test, partition_size)
            dask_X_test = dask_X_test.persist()
            client.rebalance(dask_X_test)
            dask_label_test = da.from_array(y_test, partition_size)
            dask_label_test = dask_label_test.persist()
            client.rebalance(dask_label_test)

            dtest = DaskDMatrix(
                client=client, data=dask_X_test, label=dask_label_test)

            gpu_res = {}  # Store accuracy result
            tmp = time.time()
            # Train model
            xgb.dask.train(client, param, dtrain, num_boost_round=num_round, evals=[
                (dtest, 'test')])
            print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

            # TODO: https://github.com/dmlc/xgboost/issues/4518
            dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
            dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
            # Repeat for CPU algorithm
            tmp = time.time()
            param['tree_method'] = 'hist'
            cpu_res = {}
            xgb.train(param, dtrain, num_round, evals=[
                (dtest, 'test')], evals_result=cpu_res)
            print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))


@pytest.mark.multi_gpu
def test_xgboost_airlines():
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    from sklearn.model_selection import train_test_split
    import time
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from dask import array as da
    import xgboost as xgb
    from xgboost.dask import DaskDMatrix
    from dask import array as da

    X = pd.read_csv('./open_data/allyears.1987.2013.zip',
                    dtype={'UniqueCarrier': 'category', 'Origin': 'category', 'Dest': 'category',
                           'TailNum': 'category', 'CancellationCode': 'category',
                           'IsArrDelayed': 'category', 'IsDepDelayed': 'category',
                           'DepTime': np.float32, 'CRSDepTime': np.float32, 'ArrTime': np.float32,
                           'CRSArrTime': np.float32, 'ActualElapsedTime': np.float32,
                           'CRSElapsedTime': np.float32, 'AirTime': np.float32,
                           'ArrDelay': np.float32, 'DepDelay': np.float32, 'Distance': np.float32,
                           'TaxiIn': np.float32, 'TaxiOut': np.float32, 'Diverted': np.float32,
                           'Year': np.int32, 'Month': np.int32, 'DayOfWeek': np.int32,
                           'DayofMonth': np.int32, 'Cancelled': 'category',
                           'CarrierDelay': np.float32, 'WeatherDelay': np.float32,
                           'NASDelay': np.float32, 'SecurityDelay': np.float32,
                           'LateAircraftDelay': np.float32})

    y = X["IsArrDelayed"].cat.codes
    X = X[['UniqueCarrier', 'Origin', 'Dest', 'IsDepDelayed', 'Year', 'Month',
           'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
           'ArrTime', 'CRSArrTime', 'FlightNum', 'TailNum',
           'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay',
           'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
           'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay',
           'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

    print(X.shape, y.shape)

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                        random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Specify sufficient boosting iterations to reach a minimum
    num_round = 1000

    # Leave most parameters as default
    param = {'objective': 'reg:logistic',
             'tree_method': 'gpu_hist',
             }

    from h2o4gpu.util.gpu import device_count
    n_gpus, devices = device_count(-1)

    with LocalCUDACluster(n_workers=n_gpus, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            dask_X_train = da.from_array(X_train)
            dask_label_train = da.from_array(y_train)

            dtrain = DaskDMatrix(
                client=client, data=dask_X_train, label=dask_label_train)

            dask_X_test = da.from_array(X_test)
            dask_label_test = da.from_array(y_test)

            dtest = DaskDMatrix(
                client=client, data=dask_X_test, label=dask_label_test)

            gpu_res = {}  # Store accuracy result
            tmp = time.time()
            # Train model
            xgb.dask.train(client, param, dtrain, num_boost_round=num_round, evals=[
                (dtest, 'test')])
            print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))


if __name__ == "__main__":
    pass
    # test_xgboost_covtype_multi_gpu()
