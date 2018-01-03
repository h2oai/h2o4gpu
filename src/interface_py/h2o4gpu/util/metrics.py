#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np


def ll(actual, predicted):
    """
    Computes the log likelihood.

    This function computes the log likelihood between two numbers,
    or for element between a pair of lists or numpy arrays.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double or list of doubles
             The log likelihood error between actual and predicted
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    for i in range(0, predicted.shape[0]):
        predicted[i] = min(max(1e-15, predicted[i]), 1 - 1e-15)
    err = np.seterr(all='ignore')
    score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    np.seterr(
        divide=err['divide'],
        over=err['over'],
        under=err['under'],
        invalid=err['invalid'])
    if isinstance(score, np.ndarray):
        score[np.isnan(score)] = 0
    else:
        if np.isnan(score):
            score = 0
    return score


def log_loss(actual, predicted):
    """
    Computes the log loss.

    This function computes the log loss between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
             The log loss between actual and predicted
    """
    return np.mean(ll(actual, predicted))


def se(actual, predicted):
    """
    Computes the squared error.

    This function computes the squared error between two numbers,
    or for element between a pair of lists or numpy arrays.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double or list of doubles
            The squared error between actual and predicted
    """
    return np.power(np.array(actual) - np.array(predicted), 2)


def mse(actual, predicted):
    """
    Computes the mean squared error.

    This function computes the mean squared error between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
             The mean squared error between actual and predicted
    """
    return np.mean(se(actual, predicted))


def rmse(actual, predicted):
    """
    Computes the root mean squared error.

    This function computes the root mean squared error between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
            The root mean squared error between actual and predicted
    """
    return np.sqrt(mse(actual, predicted))


def ce(actual, predicted):
    """
    Computes the classification error.

    This function computes the classification error between two lists

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
            The classification error between actual and predicted
    """
    return (
        sum([1.0 for x, y in zip(actual, predicted) if x != y]) / len(actual))


def ae(actual, predicted):
    """
    Computes the absolute error.

    This function computes the absolute error between two numbers,
    or for element between a pair of lists or numpy arrays.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double or list of doubles
            The absolute error between actual and predicted
    """
    return np.abs(np.array(actual) - np.array(predicted))


def mae(actual, predicted):
    """
    Computes the mean absolute error.

    This function computes the mean absolute error between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
            The mean absolute error between actual and predicted
    """
    return np.mean(ae(actual, predicted))


def sle(actual, predicted):
    """
    Computes the squared log error.

    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double or list of doubles
             The squared log error between actual and predicted
    """
    return (np.power(
        np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1), 2))


def msle(actual, predicted):
    """
    Computes the mean squared log error.

    This function computes the mean squared log error between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
            The mean squared log error between actual and predicted
    """
    return np.mean(sle(actual, predicted))


def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.

    This function computes the root mean squared log error between two lists
    of numbers.

    :param actual : int, float, list of numbers, numpy array
                    The ground truth value
    :param predicted : same type as actual
                     The predicted value

    :returns double
            The root mean squared log error between actual and predicted
    """
    return np.sqrt(msle(actual, predicted))


def tied_rank(x):
    """
    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    :param x : list of numbers, numpy array

    :returns list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0] * len(x)
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i, e in enumerate(sorted_x):
        if cur_val != e[0]:
            cur_val = e[0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[e[1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    :param actual : list of binary numbers, numpy array
                    The ground truth value
    :param posterior : same type as actual
                       Defines a ranking on the binary numbers,
                       from most likely to be positive to least
                       likely to be positive.

    :returns double
             The AUC between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x == 1])
    num_negative = len(actual) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    area_under_curve = ((sum_positive - num_positive *
                         (num_positive + 1) / 2.0) /
                        (num_negative * num_positive))
    return area_under_curve


def f1_opt(actual, predicted, sample_weight=None):
    """
    Computes the f1-score after optimal predictions thresholding.

    This function maximizes the f1-score by means of
    optimal predictions thresholding.

    :param actual : list of binary numbers, numpy array
                    The ground truth value
    :param predicted : int, float, list of numbers, numpy array
                     The predicted value

    :returns double
             The optimal f1-score.
    """
    import h2o4gpu.util.roc_opt as roc_opt
    if sample_weight is None:
        return roc_opt.f1_opt(actual, predicted)
    return roc_opt.f1_opt(actual, predicted, sample_weight)


def mcc_opt(actual, predicted, sample_weight=None):
    """
    Computes the MCC after optimal predictions thresholding.

    This function maximizes the Matthews Correlation Coefficient (MCC)
    by means of optimal predictions thresholding.

    :param actual : list of binary numbers, numpy array
                    The ground truth value
    :param predicted : int, float, list of numbers, numpy array
                     The predicted value

    :returns double
             The optimal MCC.
    """
    import h2o4gpu.util.roc_opt as roc_opt
    if sample_weight is None:
        return roc_opt.mcc_opt(actual, predicted)
    return roc_opt.mcc_opt(actual, predicted, sample_weight)
