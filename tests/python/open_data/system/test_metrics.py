import os
import time
from sklearn.datasets import load_iris
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import v_measure_score

import h2o4gpu.util.metrics as daicx
np.set_printoptions(precision=18)

def test_metrics():
    actual=np.array([1,2,3,4])
    predicted = np.array([1.1, 2, 3.5, 4])

    score = daicx.ll(actual, predicted)
    assert np.allclose(score, np.array([ 9.9920072216264148e-16, -3.4539575992340879e+01, -6.9079151984681758e+01, -1.0361872797702264e+02])), "ll failed"

    score = daicx.log_loss(actual, predicted)
    assert np.allclose(score, -51.80936398851132), "logloss failed"

    score = daicx.se(actual, predicted)
    assert np.allclose(score, np.array([0.010000000000000018, 0.,  0.25, 0.])), "se failed"

    score = daicx.mse(actual, predicted)
    assert np.allclose(score, 0.065), "mse failed"

    score = daicx.rmse(actual, predicted)
    assert np.allclose(score, 0.25495097567963926), "rmse failed"

    score = daicx.ce(actual, predicted)
    assert np.allclose(score, 0.5), "ce failed"

    score = daicx.ae(actual, predicted)
    assert np.allclose(score, np.array([0.10000000000000009, 0., 0.5, 0.])), "ae failed"

    score = daicx.mae(actual, predicted)
    assert np.allclose(score, 0.15000000000000002), "mae failed"

    score = daicx.sle(actual, predicted)
    assert np.allclose(score, np.array([0.002380480119680131, 0., 0.013872843488432929, 0.])), "sle failed"

    score = daicx.msle(actual, predicted)
    assert np.allclose(score, 0.004063330902028265), "msle failed"

    score = daicx.rmsle(actual, predicted)
    assert np.allclose(score, 0.06374426171843443), "rmsle failed"

    score = daicx.auc(actual, predicted)
    assert np.allclose(score, 0.0), "auc failed"

    rank = daicx.tied_rank(actual)
    assert np.allclose(rank, np.array([1.0, 2.0, 3.0, 4.0])), "rank failed"

    score = daicx.f05_opt(actual, predicted)
    assert np.allclose(score, 3.2142857142857144), "f05_opt failed"

    score = daicx.f1_opt(actual, predicted)
    assert np.allclose(score, 1.6363636363636365), "f1_opt failed"

    score = daicx.f2_opt(actual, predicted)
    assert np.allclose(score, 1.1363636363636365), "f2_opt failed"

    score = daicx.mcc_opt(actual, predicted)
    assert np.allclose(score, 0.0), "mcc_opt failed"

    score = daicx.acc_opt(actual, predicted)
    assert np.allclose(score, 2.75), "acc_opt failed"

    matrix = daicx.confusion_matrices(actual, predicted)
    # assert np.allclose(matrix, 1.0), "mcc_opt failed"