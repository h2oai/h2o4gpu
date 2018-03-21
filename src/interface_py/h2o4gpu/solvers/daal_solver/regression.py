#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import warnings
from enum import Enum
import numpy as np
from daal.algorithms.linear_regression import training as linear_training
from daal.algorithms.linear_regression import prediction as linear_prediction
from daal.algorithms.ridge_regression import training as ridge_training
from daal.algorithms.ridge_regression import prediction as ridge_prediction
from daal.data_management import HomogenNumericTable, NumericTable
from h2o4gpu.solvers.daal_solver.daal_data import getNumpyArray
from .utils import printNumericTable
from .daal_data import IInput

class Method(Enum):
    '''
    Method solver for IntelDAAL
    data: {(x1,y1),...,(xm,ym)} and a tentative function (model) to
    fit the data against in form of : f(x) = c1f1(x)+...+cnfn(x), to
    find the response vector, here are used two methods: normal equation and
    QR decomposition - used by default for its numerical stability
    and performance - intelDaal QR decomposition written in Fortran.
    '''
    qr_dense = linear_training.qrDense
    normal_equation = linear_training.normEqDense

class LinearRegression(object):
    '''Linear Regression based on DAAL
    library
    '''

    def __init__(self, fit_intercept=True, normalize=False, **kwargs):
        '''
        :param fit_intercept: calculate all betas by default
        :param normalize: z-score used for independent variables
        :param in kwargs: method: normalEquation, QR
        '''
        if 'method' in kwargs.keys() and kwargs['method'] in Method:
            self.method = kwargs['method'].value
            if self.method == Method.qr_dense.value:
                fit_intercept = False
        else:
            self.method = None
        self.normalize = normalize
        self.model = None
        self.parameters = ['intercept'] if fit_intercept else []
        self.train_data_array = None
        self.response_data_array = None

    def fit(self, X, y=None):

        '''
        masquerade function for train()
        '''
        return self.train(X, y)

    def train(self, X, y=None):
        '''
        :param X: training data
        :param y: dependent variables (responses)
        :return: Linear Regression model object
        '''

        # Training data and responses
        Input = IInput.HomogenousDaalData(X).getNumericTable()
        Responses = IInput.HomogenousDaalData(y).getNumericTable()

        # Training object with/without normalization
        if self.method:
            linear_training_algorithm = linear_training.Batch(
                method=self.method
            )
        else:
            linear_training_algorithm = linear_training.Batch()

        # set input values
        linear_training_algorithm.input.set(linear_training.data, Input)
        linear_training_algorithm.input.set(linear_training.dependentVariables,
                                            Responses)
        # check if intercept flag is set
        linear_training_algorithm.parameter.interceptFlag = True \
            if 'intercept' in self.parameters else False
        # calculate
        res = linear_training_algorithm.compute()
        # return trained model
        self.model = res.get(linear_training.model)
        return self.model

    def get_beta(self):

        '''
        :return: Linear Regression coefficients
        '''

        if self.model is None:
            warnings.warn("The training model is not calculated yet.",
                          UserWarning)
        return self.model.getBeta()

    def predict(self, X):
        '''
        Make prediction for X - unseen data using a trained model
        :param X:new data
        intercept: from parameters, a boolean indicating
        if calculate Beta0 (intercept)
        '''

        Data = IInput.HomogenousDaalData(X).getNumericTable()
        linear_prediction_algorithm = \
            linear_prediction.Batch()
        # set input
        linear_prediction_algorithm.input.setModel(
            linear_prediction.model, self.model)
        linear_prediction_algorithm.input.setTable(
            linear_prediction.data, Data)

        # TODO
        #if 'intercept' in self.parameters:
        #    linear_prediction_algorithm.parameter.interceptFlag = True

        res = linear_prediction_algorithm.compute()
        return getNumpyArray(res.get(linear_prediction.prediction))

    def _score(self,
               predicted_response,
               true_responses=None):
        '''
        :param X: not needed for DAAL, only dependent variables
        for calculation used
        :param y:
        :param smaple_weight:
        '''

        if true_responses is None:
            true_responses = self.response_data_array

        rs = ((true_responses - predicted_response) ** 2).sum(axis=0)
        ts = ((true_responses - true_responses.mean(axis=0)) ** 2).sum(axis=0)
        return 1-rs/ts

    def get_params(self):
        return self.parameters

    def set_params(self, **params):
        '''
        params for prediction
        '''
        # TODO: list of parameters for Linear Regression
        if 'intercept' in params and 'intercept' not in self.parameters:
            self.parameters.append('intercept')

    def set_attributes(self):
        warnings.warn("DAAL Linear Regression doesn't have any attributes",\
                      UserWarning)

    @staticmethod
    def print_table(daal_table, message='',
                    num_printed_rows=0,
                    num_printed_cols=0,
                    interval=10):
        '''Print results from daal
        :param daal_table:
        :param message:
        :param num_printed_rows:
        :param num_printed_cols:
        :param interval:
        '''

        assert isinstance(daal_table, NumericTable), \
                        "requires a daal.data_management.NumericTable"
        printNumericTable(daal_table, message, num_printed_rows,
                          num_printed_cols, interval)


class RidgeRegression(object):
    '''Ridge Regression based on DAAL
    library
    '''

    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize=False,
                 **_):
        '''
        :param kwargs: alpha: Regularization parameter, a small positive
        value with default 1.0
        :param fit_intercept:
        :param normalize:
        '''
        self.alpha = alpha
        self.normalize = normalize
        self.model = None
        self.parameters = ['intercept'] if fit_intercept else []
        self.train_data_array = None
        self.response_data_array = None

    def fit(self, X, y=None):

        '''
        masquerade function for train()
        '''
        return self.train(X, y)

    def train(self, X, y=None):
        '''
        :param X: training data
        :param y: dependent variables (responses)
        :return: Ridge Regression model object
        '''

        # Training data and responses
        Input = IInput.HomogenousDaalData(X).getNumericTable()
        Responses = IInput.HomogenousDaalData(y).getNumericTable()

        # Training object with normalization
        ridge_training_algorithm = ridge_training.Batch()

        # set input values
        ridge_training_algorithm.input.set(ridge_training.data, Input)
        ridge_training_algorithm.input.set(ridge_training.dependentVariables,
                                           Responses)
        # check if intercept flag is set
        ridge_training_algorithm.parameter.interceptFlag = True \
            if 'intercept' in self.parameters else True
        # set parameter
        alpha_nt = HomogenNumericTable(np.array([self.alpha], ndmin=2))
        ridge_training_algorithm.parameter.ridgeParameters = alpha_nt
        # calculate
        res = ridge_training_algorithm.compute()
        # return trained model
        self.model = res.get(ridge_training.model)
        return self.model

    def get_beta(self):

        '''
        :return: Linear Regression coefficients
        '''

        if self.model is None:
            warnings.warn("The training model is not calculated yet.",
                          UserWarning)
        return self.model.getBeta()

    def predict(self, X):
        '''
        Make prediction for X - unseen data using a trained model
        :param X:new data
        intercept: from parameters, a boolean indicating
        if calculate Beta0 (intercept)
        '''

        Data = IInput.HomogenousDaalData(X).getNumericTable()
        ridge_prediction_algorithm = \
            ridge_prediction.Batch()
        # set input
        ridge_prediction_algorithm.input.setModel(
            ridge_prediction.model, self.model)
        ridge_prediction_algorithm.input.setTable(
            ridge_prediction.data, Data)

        if 'intercept' in self.parameters:
            beta_coeff = self.get_beta()
            np_beta = getNumpyArray(beta_coeff)
            self.intercept_ = [np_beta[0, 0]]
        # calculate
        res = ridge_prediction_algorithm.compute()
        return getNumpyArray(res.get(ridge_prediction.prediction))

    def _score(self,
               predicted_response,
               true_responses=None):
        '''
        :param X: not needed for DAAL, only dependent variables
        for calculation used
        :param y:
        :param smaple_weight:
        '''

        if true_responses is None:
            true_responses = self.response_data_array

        rs = ((true_responses - predicted_response) ** 2).sum(axis=0)
        ts = ((true_responses - true_responses.mean(axis=0)) ** 2).sum(axis=0)
        return 1-rs/ts

    def get_params(self):
        return self.parameters

    def set_params(self, **params):
        '''
        params for prediction
        '''
        # TODO: list of parameters for Linear Regression
        if 'intercept' in params and 'intercept' not in self.parameters:
            self.parameters.append('intercept')

    def set_attributes(self):
        warnings.warn("DAAL Linear Regression doesn't have any attributes",\
                      UserWarning)
