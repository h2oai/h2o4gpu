from __future__ import print_function

import numpy as np
import unittest
import xgboost as xgb
import os, sys
import time

try:
    from nose.plugins.attrib import attr
except:
    pass

rng = np.random.RandomState(1994)

import _pickle as pickle
def save_obj(obj, name):
    # print("Saving %s" % name)
    with open(name, 'wb') as f:
        pickle.dump(obj=obj, file=f)
        # os.sync()


def load_obj(name):
    # print("Loading %s" % name)
    with open(name, 'rb') as f:
        return pickle.load(f)

num_rows = 5000
num_cols = 500
n_estimators = 1

def makeXy():
    np.random.seed(1)
    X = np.random.randn(num_rows, num_cols)
    y = [0, 1] * int(num_rows / 2)
    return X,y

def makeXtest():
    np.random.seed(1)
    Xtest = np.random.randn(num_rows, num_cols)
    return Xtest


#@attr('gpu')
class TestGPUPredict(unittest.TestCase):
    def test_predict_nopickle(self):
        X,y = makeXy()

        dm = xgb.DMatrix(X, label=y)
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        print("Before model.predict on GPU")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = bst.predict(dm, output_margin=True)
        print(gpu_pred)
        print("A1 non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("A1 GPU Time to predict = %g" % (time.time() - tmp))
        print("A1 Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        tmp = time.time()
        cpu_pred = bst.predict(dm, output_margin=True)
        print(cpu_pred)
        print("A2 non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("A2 CPU Time to predict = %g" % (time.time() - tmp))
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_nodm(self):

        tmp = time.time()
        X,y = makeXy()
        Xtest = makeXtest()
        print("Time to Make Data = %g" % (time.time() - tmp))

        tmp = time.time()
        dm = xgb.DMatrix(X, label=y)
        dm_test = xgb.DMatrix(Xtest)
        print("Time to DMatrix = %g" % (time.time() - tmp))

        tmp = time.time()
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        print("Time to Train = %g" % (time.time() - tmp))
        assert self.non_decreasing(res["train"]["auc"])

        tmp = time.time()
        print("Before model.predict on GPU")
        sys.stdout.flush()
        gpu_pred = bst.predict(dm_test, output_margin=True)
        print(gpu_pred)
        print("B1 non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("B1 GPU Time to predict = %g" % (time.time() - tmp))

        tmp = time.time()
        print("B1 Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        cpu_pred = bst.predict(dm_test, output_margin=True)
        print(cpu_pred)
        print("B2 non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("B2 CPU Time to predict = %g" % (time.time() - tmp))

        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_pickle(self):
        X,y = makeXy()

        dm = xgb.DMatrix(X, label=y)
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        bst = xgb.train(param, dm, n_estimators, evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        # pickle model
        save_obj(bst,"bst.pkl")
        # delete model
        del bst
        # load model
        bst = load_obj("bst.pkl")
        os.remove("bst.pkl")

        # continue as before
        print("Before model.predict on GPU")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = bst.predict(dm, output_margin=True)
        print(gpu_pred)
        print("C1 non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("C1 GPU Time to predict = %g" % (time.time() - tmp))
        print("C1 Before model.predict on CPU")
        sys.stdout.flush()
        bst.set_param({"predictor": "cpu_predictor"})
        tmp = time.time()
        cpu_pred = bst.predict(dm, output_margin=True)
        print(cpu_pred)
        print("C2 non-zeroes: %d:" % (np.count_nonzero(cpu_pred)))
        print("C2 CPU Time to predict = %g" % (time.time() - tmp))
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_sklearn_nopickle(self):
        X,y = makeXy()
        Xtest = makeXtest()

        from xgboost import XGBClassifier
        kwargs={}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['silent'] = 0
        kwargs['objective'] = 'binary:logistic'

        model = XGBClassifier(n_estimators=n_estimators, **kwargs)
        model.fit(X,y)
        print(model)

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("D non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("D GPU Time to predict = %g" % (time.time() - tmp))
        # MAJOR issue: gpu predictions wrong  -- all zeros
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_predict_sklearn_pickle(self):
        X,y = makeXy()
        Xtest = makeXtest()

        from xgboost import XGBClassifier
        kwargs={}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['silent'] = 0
        kwargs['objective'] = 'binary:logistic'

        model = XGBClassifier(**kwargs)
        model.fit(X,y)
        print(model)

        # pickle model
        save_obj(model,"model.pkl")
        # delete model
        del model
        # load model
        model = load_obj("model.pkl")
        os.remove("model.pkl")

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("E non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("E GPU Time to predict = %g" % (time.time() - tmp))
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)


    # only run the below after the above
    def test_predict_sklearn_frompickle(self):
        Xtest = makeXtest()

        # load model
        model = load_obj("./tests/python/open_data/gbm/model_saved.pkl")

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("F non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("F GPU Time to predict = %g" % (time.time() - tmp))
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        #np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))
