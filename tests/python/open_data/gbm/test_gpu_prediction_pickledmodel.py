import _pickle as pickle

import numpy as np
import unittest
import xgboost as xgb
import os
import sys
import time
import pytest
import os

try:
    from nose.plugins.attrib import attr
except:
    pass

# TODO: remove when nccl works on ppc


rng = np.random.RandomState(1994)


def save_obj(obj, name):
    # print("Saving %s" % name)
    with open(name, 'wb') as f:
        pickle.dump(obj=obj, file=f)
        # os.sync()


def load_obj(name):
    from h2o4gpu.util.xgboost_migration import load_pkl
    return load_pkl(name)


num_rows = 5000
num_cols = 500
n_estimators = 1


def makeXy():
    np.random.seed(1)
    X = np.random.randn(num_rows, num_cols)
    y = [0, 1] * int(num_rows / 2)
    return X, y


def makeXtest():
    np.random.seed(1)
    Xtest = np.random.randn(num_rows, num_cols)
    return Xtest


@pytest.mark.parametrize("n_gpus", [1, 0, None])
class TestGPUPredict(object):
    def test_predict_nopickle(self, n_gpus):
        X, y = makeXy()

        dm = xgb.DMatrix(X, label=y)
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
        }
        param = self.set_n_gpus(param, n_gpus)
        bst = xgb.train(param, dm, n_estimators,
                        evals=watchlist, evals_result=res)
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

    def test_predict_nodm(self, n_gpus):

        tmp = time.time()
        X, y = makeXy()
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
        param = self.set_n_gpus(param, n_gpus)
        bst = xgb.train(param, dm, n_estimators,
                        evals=watchlist, evals_result=res)
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

    def test_predict_pickle(self, n_gpus):
        X, y = makeXy()

        dm = xgb.DMatrix(X, label=y)
        watchlist = [(dm, 'train')]
        res = {}
        param = {
            "objective": "binary:logistic",
            "predictor": "gpu_predictor",
            'eval_metric': 'auc',
            'gpu_id': 0,
        }
        param = self.set_n_gpus(param, n_gpus)

        bst = xgb.train(param, dm, n_estimators,
                        evals=watchlist, evals_result=res)
        assert self.non_decreasing(res["train"]["auc"])

        # pickle model
        save_obj(bst, "bst-{0}.pkl".format(os.getpid()))
        # delete model
        del bst
        # load model
        bst = load_obj("bst-{0}.pkl".format(os.getpid()))
        os.remove("bst-{0}.pkl".format(os.getpid()))

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

    def test_predict_sklearn_nopickle(self, n_gpus):
        X, y = makeXy()
        Xtest = makeXtest()

        from xgboost import XGBClassifier
        kwargs = {}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['objective'] = 'binary:logistic'
        kwargs = self.set_n_gpus(kwargs, n_gpus)

        model = XGBClassifier(n_estimators=n_estimators, **kwargs)
        model.fit(X, y)
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
        # np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    @pytest.mark.skip("Temporary skipped as data dir is not synced")
    def test_predict_sklearn_pickle(self, n_gpus):
        X, y = makeXy()
        Xtest = makeXtest()

        from xgboost import XGBClassifier
        kwargs = {}
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        kwargs['objective'] = 'binary:logistic'
        kwargs = self.set_n_gpus(kwargs, n_gpus)

        model = XGBClassifier(**kwargs)
        model.fit(X, y)

        # pickle model
        save_obj(model, "model.pkl")
        # delete model
        del model
        # load model
        model = load_obj("model.pkl")
        os.remove("model.pkl")

        # continue as before
        print("Before model.predict")
        sys.stdout.flush()
        tmp = time.time()
        Xtest = makeXtest()
        gpu_pred = model.predict(Xtest, output_margin=True)
        print(gpu_pred)
        print("E non-zeroes: %d:" % (np.count_nonzero(gpu_pred)))
        print("E GPU Time to predict = %g" % (time.time() - tmp))
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        # np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    # only run the below after the above

    @pytest.mark.skip("Temporary skipped as data dir is not synced")
    def test_predict_sklearn_frompickle(self, n_gpus):
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
        _ = model.get_booster().copy()
        # ISSUE1: Doesn't use gpu_predictor.
        # ISSUE2: Also, no way to switch to cpu_predictor?
        # np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))

    def set_n_gpus(self, params, n_gpus):
        if n_gpus is not None:
            params['n_gpus'] = n_gpus
        return params


if __name__ == "__main__":
    pass
    # TestGPUPredict().test_predict_sklearn_frompickle(1)
