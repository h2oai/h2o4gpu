# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

try:
    import datatable as dt
    HAVE_DT = True
except:
    HAVE_DT = False
    pass

# instructions
# pip uninstall xgboost # rm -rf any residual site-packages/xgboost* in your environment
# git clone https://github.com/h2oai/xgboost
# git checkout h2oai_dt
# cd xgboost ; mkdir -p build ; cd build ; cmake .. -DUSE_CUDA=ON ; make -j ; cd ..
# cd python-package ; python setup.py install ; cd .. # installs as egg instead of like when doing wheel
# python tests/benchmark/testdt.py --algorithm=gpu_hist # use hist if you don't have a gpu


def non_increasing(L, tolerance):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


#Check result is always decreasing and final accuracy is within tolerance
def assert_accuracy(res1, res2, tolerance=0.02):
    assert non_increasing(res1, tolerance)
    assert np.allclose(res1[-1], res2[-1], 1e-3, 1e-2)

def run_benchmark(algorithm='gpu_hist', rows=1000000, columns=50, iterations=5, test_size=0.25):
    
    print("Generating dataset: {} rows * {} columns".format(rows, columns))
    print("{}/{} test/train split".format(test_size, 1.0 - test_size))
    tmp = time.time()
    X, y = make_classification(rows, n_features=columns, random_state=7)
    aa = np.random.rand(X.shape[0],X.shape[1])
    fraction_missing = 0.1
    X[aa<fraction_missing]=np.NaN
    print("Number of Nans: %d" % (np.isnan(X).sum()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    print ("Generate Time: %s seconds" % (str(time.time() - tmp)))

    param = {'objective': 'binary:logistic',
             'max_depth': 6,
             'silent': 0,
             'n_gpus': 1,
             'gpu_id': 0,
             'eval_metric': 'error',
             'debug_verbose': 0,
             }

    param['tree_method'] = algorithm

    do_dt = True
    do_dt_likeDAI = True
    do_ccont = False
    do_nondt = True

    do_check_accuracy = True

    tmp = time.time()
    if do_ccont:
        X_train_cc = X_train
        X_test_cc = X_test
        y_train_cc = y_train
        y_test_cc = y_test
    else:
        # convert to dt as test
        X_train_cc = np.asfortranarray(X_train)
        X_test_cc = np.asfortranarray(X_test)
        y_train_cc = np.asfortranarray(y_train)
        y_test_cc = np.asfortranarray(y_test)

        if not (X_train_cc.flags['F_CONTIGUOUS'] and X_test_cc.flags['F_CONTIGUOUS'] \
                        and y_train_cc.flags['F_CONTIGUOUS'] and y_test_cc.flags['F_CONTIGUOUS']):
            ValueError("Need data to be Fortran (i.e. column-major) contiguous")
    print("dt prepare1 Time: %s seconds" % (str(time.time() - tmp)))

    res={}
    if do_nondt:
        print("np->DMatrix Start")
        # omp way
        tmp = time.time()
        # below takes about 2.826s if do_ccont=False
        # below takes about 0.248s if do_ccont=True
        dtrain = xgb.DMatrix(X_train_cc, y_train_cc, nthread=-1)
        print ("np->DMatrix1 Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        dtest = xgb.DMatrix(X_test_cc, y_test_cc, nthread=-1)
        print ("np->DMatrix2 Time: %s seconds" % (str(time.time() - tmp)))

        print("Training with '%s'" % param['tree_method'])
        tmp = time.time()
        res_tmp = {}
        xgb.train(param, dtrain, iterations, evals=[(dtrain, "train"),(dtest, "test")], evals_result=res_tmp)
        res['1'] = res_tmp['train']['error']
        print("Train Time: %s seconds" % (str(time.time() - tmp)))
    if HAVE_DT and do_dt:

        # convert to column-major contiguous in memory to mimic persistent column-major state
        # do_cccont = True leads to prepare2 time of about 1.4s for 1000000 rows * 50 columns
        # do_cccont = False leads to prepare2 time of about 0.000548 for 1000000 rows * 50 columns
        tmp = time.time()
        dtdata_X_train = dt.DataTable(X_train_cc)
        dtdata_X_test = dt.DataTable(X_test_cc)
        dtdata_y_train = dt.DataTable(y_train_cc)
        dtdata_y_test = dt.DataTable(y_test_cc)
        print ("dt prepare2 Time: %s seconds" % (str(time.time() - tmp)))

        #test = dtdata_X_train.tonumpy()
        #print(test)

        print ("dt->DMatrix Start")
        # omp way
        tmp = time.time()
        # below takes about 0.47s - 0.53s independent of do_ccont
        dtrain = xgb.DMatrix(dtdata_X_train, dtdata_y_train, nthread=-1)
        print ("dt->DMatrix1 Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        dtest = xgb.DMatrix(dtdata_X_test, dtdata_y_test, nthread=-1)
        print ("dt->DMatrix2 Time: %s seconds" % (str(time.time() - tmp)))

        print("Training with '%s'" % param['tree_method'])
        tmp = time.time()
        res_tmp = {}
        xgb.train(param, dtrain, iterations, evals=[(dtrain, "train"),(dtest, "test")], evals_result=res_tmp)
        res['2'] = res_tmp['train']['error']
        print ("Train Time: %s seconds" % (str(time.time() - tmp)))
    if HAVE_DT and do_dt_likeDAI:

        # convert to column-major contiguous in memory to mimic persistent column-major state
        # do_cccont = True leads to prepare2 time of about 1.4s for 1000000 rows * 50 columns
        # do_cccont = False leads to prepare2 time of about 0.000548 for 1000000 rows * 50 columns
        tmp = time.time()
        dtdata_X_train = dt.DataTable(X_train_cc)
        dtdata_X_test = dt.DataTable(X_test_cc)
        dtdata_y_train = dt.DataTable(y_train_cc)
        dtdata_y_test = dt.DataTable(y_test_cc)
        print ("dt prepare2 Time: %s seconds" % (str(time.time() - tmp)))

        #test = dtdata_X_train.tonumpy()
        #print(test)

        print ("dt->DMatrix Start")
        # omp way
        tmp = time.time()
        dtrain = xgb.DMatrix(dtdata_X_train.tonumpy(), dtdata_y_train.tonumpy(), nthread=-1)
        print ("dt->DMatrix1 Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        dtest = xgb.DMatrix(dtdata_X_test.tonumpy(), dtdata_y_test.tonumpy(), nthread=-1)
        print ("dt->DMatrix2 Time: %s seconds" % (str(time.time() - tmp)))

        print("Training with '%s'" % param['tree_method'])
        tmp = time.time()
        res_tmp = {}
        xgb.train(param, dtrain, iterations, evals=[(dtrain, "train"),(dtest, "test")], evals_result=res_tmp)
        res['3'] = res_tmp['train']['error']
        print ("Train Time: %s seconds" % (str(time.time() - tmp)))
    if HAVE_DT and do_check_accuracy:
        assert_accuracy(res['1'],res['2'])
        assert_accuracy(res['1'],res['3'])


def test_dt_integration_xgboost_hist():
    run_benchmark(algorithm='hist')

def test_dt_integration_xgboost_gpu_hist(): 
    run_benchmark(algorithm='gpu_hist')

if __name__ == '__main__':
    test_dt_integration_xgboost_hist()
    test_dt_integration_xgboost_gpu_hist()
