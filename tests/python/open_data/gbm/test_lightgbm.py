import time
import sys
import logging

import pytest

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def test_lightgbm_gpu():
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    X1= np.repeat(np.arange(10), 1000)
    X2= np.repeat(np.arange(10), 1000)
    np.random.shuffle(X2)

    y = (X1 + np.random.randn(10000)) * (X2  + np.random.randn(10000))
    data = pd.DataFrame({'y': y, 'X1': X1, 'X2': X2})

    lgb_params = {'learning_rate'    : 0.1,
                  'boosting'         : 'dart',
                  'objective'        : 'regression',
                  'metric'           : 'rmse',
                  'feature_fraction' : 0.9,
                  'bagging_fraction' : 0.75,
                  'num_leaves'       : 31,
                   'bagging_freq'     : 1,
                  'min_data_per_leaf': 250, 'device_type': 'gpu', 'gpu_device_id': 0}
    lgb_train = lgb.Dataset(data=data[['X1', 'X2']], label=data.y)
    cv = lgb.cv(lgb_params,
                  lgb_train,
                  num_boost_round=100,
                  early_stopping_rounds=15,
                  stratified=False,
                  verbose_eval=50)


def test_lightgbm_cpu():
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    X1 = np.repeat(np.arange(10), 1000)
    X2 = np.repeat(np.arange(10), 1000)
    np.random.shuffle(X2)

    y = (X1 + np.random.randn(10000)) * (X2 + np.random.randn(10000))
    data = pd.DataFrame({'y': y, 'X1': X1, 'X2': X2})

    lgb_params = {'learning_rate': 0.1,
                  'boosting': 'dart',
                  'objective': 'regression',
                  'metric': 'rmse',
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.75,
                  'num_leaves': 31,
                  'bagging_freq': 1,
                  'min_data_per_leaf': 250}
    lgb_train = lgb.Dataset(data=data[['X1', 'X2']], label=data.y)
    cv = lgb.cv(lgb_params,
                lgb_train,
                num_boost_round=100,
                early_stopping_rounds=15,
                stratified=False,
                verbose_eval=50)

if __name__ == '__main__':
    test_lightgbm_cpu()
    test_lightgbm_gpu()

