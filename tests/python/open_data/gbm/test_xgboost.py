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

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

@pytest.mark.multi_gpu
@pytest.mark.parametrize("n_gpus", [0, 1])
def test_xgboost_covtype(n_gpus):
    import xgboost as xgb
    import numpy as np
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    import time

    # Fetch dataset using sklearn
    cov = fetch_covtype()
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
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    import time
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from dask import array as da
    import xgboost as xgb
    from xgboost.dask import DaskDMatrix
    from dask import array as da

    # Fetch dataset using sklearn
    cov = fetch_covtype()
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
            dask_X_train = da.from_array(X_train, partition_size)
            dask_y_label = da.from_array(y_train, partition_size)
            dtrain = DaskDMatrix(client=client, data=dask_X_train, label=dask_y_label)
            dtest = DaskDMatrix(client=client, data=da.from_array(X_test, partition_size), label=da.from_array(y_test, partition_size))

            gpu_res = {}  # Store accuracy result
            tmp = time.time()
            # Train model
            xgb.dask.train(client, param, dtrain, num_boost_round = num_round, evals=[
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


if __name__ == "__main__":
    test_xgboost_covtype_multi_gpu()
