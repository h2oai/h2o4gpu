# -*- encoding: utf-8 -*-
"""
:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np


def ll(actual, predicted):
    """
    Computes the log likelihood.
    This function computes the log likelihood between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The log likelihood error between actual and predicted
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    for i in range(0, predicted.shape[0]):
        predicted[i] = min(max(1e-15, predicted[i]), 1 - 1e-15)
    err = np.seterr(all='ignore')
    score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    np.seterr(divide=err['divide'], over=err['over'],
              under=err['under'], invalid=err['invalid'])
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
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The log loss between actual and predicted
    """
    return np.mean(ll(actual, predicted))


def se(actual, predicted):
    """
    Computes the squared error.
    This function computes the squared error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared error between actual and predicted
    """
    return np.power(np.array(actual) - np.array(predicted), 2)


def mse(actual, predicted):
    """
    Computes the mean squared error.
    This function computes the mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared error between actual and predicted
    """
    return np.mean(se(actual, predicted))


def rmse(actual, predicted):
    """
    Computes the root mean squared error.
    This function computes the root mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared error between actual and predicted
    """
    return np.sqrt(mse(actual, predicted))
