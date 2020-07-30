import time
import sys
import platform
import logging

import pytest

print(sys.path)

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.skipif(platform.machine().startswith("ppc64le"), reason="lightgbm on gpu is not supported yet")
@pytest.mark.parametrize('booster,', ["dart", "gbdt"])
def test_lightgbm_gpu(booster):
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
                  'boosting': booster,
                  'objective': 'regression',
                  'metric': 'rmse',
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.75,
                  'num_leaves': 31,
                  'bagging_freq': 1,
                  'min_data_per_leaf': 250, 'device_type': 'gpu', 'gpu_device_id': 0}
    lgb_train = lgb.Dataset(data=data[['X1', 'X2']], label=data.y)
    cv = lgb.cv(lgb_params,
                lgb_train,
                num_boost_round=100,
                early_stopping_rounds=15,
                stratified=False,
                verbose_eval=50)


@pytest.mark.parametrize('booster,', ["dart", "gbdt"])
def test_lightgbm_cpu(booster):
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
                  'boosting': booster,
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


@pytest.mark.skip(reason="Huge memory footprint")
@pytest.mark.parametrize('booster,', ["dart", "gbdt"])
@pytest.mark.timeout(3600)
def test_lightgbm_cpu_airlines_full(booster):
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    data = pd.read_csv('./open_data/allyears.1987.2013.zip',
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

    y = data["IsArrDelayed"].cat.codes
    data = data[['UniqueCarrier', 'Origin', 'Dest', 'IsDepDelayed', 'Year', 'Month',
                 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
                 'ArrTime', 'CRSArrTime', 'FlightNum', 'TailNum',
                 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay',
                 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
                 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

    lgb_params = {'learning_rate': 0.1,
                  'boosting': booster,
                  'objective': 'binary',
                  'metric': 'rmse',
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.75,
                  'num_leaves': 31,
                  'bagging_freq': 1,
                  'min_data_per_leaf': 250}
    lgb_train = lgb.Dataset(data=data, label=y)
    cv = lgb.cv(lgb_params,
                lgb_train,
                num_boost_round=50,
                early_stopping_rounds=5,
                stratified=False,
                verbose_eval=10)


@pytest.mark.parametrize('booster,', ["dart", "gbdt"])
@pytest.mark.parametrize('year,', ["1987",
                                   # "1988", "1989", "1990", "1991", "1992", "1993", "1994",
                                   #                                    "1995", "1996", "1997", "1998", "1999", "2000", "2001",
                                   #                                    "2002", "2003", "2004", "2005", "2006", "2007"
                                   ])
@pytest.mark.timeout(3600)
def test_lightgbm_cpu_airlines_year(booster, year):
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    data = pd.read_csv('./open_data/airlines/year{0}.zip'.format(year),
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

    y = data["IsArrDelayed"].cat.codes
    data = data[['UniqueCarrier', 'Origin', 'Dest', 'IsDepDelayed', 'Year', 'Month',
                 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
                 'ArrTime', 'CRSArrTime', 'FlightNum', 'TailNum',
                 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay',
                 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
                 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

    lgb_params = {'learning_rate': 0.1,
                  'boosting': booster,
                  'objective': 'binary',
                  'metric': 'rmse',
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.75,
                  'num_leaves': 31,
                  'bagging_freq': 1,
                  'min_data_per_leaf': 250}
    lgb_train = lgb.Dataset(data=data, label=y)
    cv = lgb.cv(lgb_params,
                lgb_train,
                num_boost_round=50,
                early_stopping_rounds=5,
                stratified=False,
                verbose_eval=10)


@pytest.mark.timeout(3600)
@pytest.mark.skipif(platform.machine().startswith("ppc64le"), reason="lightgbm on gpu is not supported yet")
@pytest.mark.parametrize('booster,', ["dart", "gbdt"])
@pytest.mark.parametrize('year,', ["1987",
                                   # "1988", "1989", "1990", "1991", "1992", "1993", "1994",
                                   #                                    "1995", "1996", "1997", "1998", "1999", "2000", "2001",
                                   #                                    "2002", "2003", "2004", "2005", "2006", "2007"
                                   ])
@pytest.mark.timeout(3600)
def test_lightgbm_gpu_airlines_year(booster, year):
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    data = pd.read_csv('./open_data/airlines/year{0}.zip'.format(year),
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

    y = data["IsArrDelayed"].cat.codes
    data = data[['UniqueCarrier', 'Origin', 'Dest', 'IsDepDelayed', 'Year', 'Month',
                 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
                 'ArrTime', 'CRSArrTime', 'FlightNum', 'TailNum',
                 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay',
                 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
                 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

    lgb_params = {'learning_rate': 0.1,
                  'boosting': booster,
                  'objective': 'binary',
                  'metric': 'rmse',
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.75,
                  'num_leaves': 31,
                  'bagging_freq': 1,
                  'min_data_per_leaf': 250,
                  'device_type': 'gpu',
                  'gpu_device_id': 0}
    lgb_train = lgb.Dataset(data=data, label=y)
    cv = lgb.cv(lgb_params,
                lgb_train,
                num_boost_round=50,
                early_stopping_rounds=5,
                stratified=False,
                verbose_eval=10)


def test_lightgbm_accuracy_cpu():
    import pickle
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    (model, lightgbm_params, args_fit) = pickle.load(
        open("./open_data/lgb_prefit_8c1a2d9f-4477-4abc-aec6-502ce335e551.pkl", "rb"))
    X, y, sample_weight, init_score, eval_set, valid_X_features, eval_sample_weight, init_score, eval_metric, early_stopping_rounds, X_features, verbose_fit = args_fit
    print(model.get_params().keys())
    print(model.get_params()['device_type'])
    model.set_params(device_type='cpu')
    print(model.get_params()['device_type'])
    model.fit(X, y, sample_weight=sample_weight, init_score=init_score, eval_set=eval_set, eval_names=valid_X_features, eval_sample_weight=eval_sample_weight,
              eval_init_score=init_score, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, feature_name=X_features, verbose=verbose_fit)

    eval_loss = model.evals_result_['name0']['multi_logloss']
    assert len(eval_loss) > 90
    assert eval_loss[-1] < 0.33


@pytest.mark.skipif(platform.machine().startswith("ppc64le"), reason="lightgbm on gpu is not supported yet")
def test_lightgbm_accuracy_gpu():
    import pickle
    import numpy as np
    import pandas as pd
    from h2o4gpu.util.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
    import lightgbm as lgb

    (model, lightgbm_params, args_fit) = pickle.load(
        open("./open_data/lgb_prefit_8c1a2d9f-4477-4abc-aec6-502ce335e551.pkl", "rb"))
    X, y, sample_weight, init_score, eval_set, valid_X_features, eval_sample_weight, init_score, eval_metric, early_stopping_rounds, X_features, verbose_fit = args_fit
    print(model.get_params().keys())
    print(model.get_params()['device_type'])
    model.set_params(device_type='gpu')
    print(model.get_params()['device_type'])
    model.fit(X, y, sample_weight=sample_weight, init_score=init_score, eval_set=eval_set, eval_names=valid_X_features, eval_sample_weight=eval_sample_weight,
              eval_init_score=init_score, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, feature_name=X_features, verbose=verbose_fit)

    eval_loss = model.evals_result_['name0']['multi_logloss']
    assert len(eval_loss) > 90
    assert eval_loss[-1] < 0.33


if __name__ == '__main__':
    pass
    # test_lightgbm_cpu()
    # test_lightgbm_gpu()
    # test_lightgbm_cpu_airlines()
