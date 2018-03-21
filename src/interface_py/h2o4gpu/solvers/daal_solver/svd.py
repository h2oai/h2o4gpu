#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

from __future__ import print_function
import sys
from enum import Enum
import numpy as np
from daal.algorithms import svd
from .daal_data import IInput

__all__ = ['SingularValueParameter', 'SVD']

class SingularValueParameter(Enum):
    '''
    Algorithm Parameter: for leftSingularMatrix and
    rightSingularMatrix specifies whether the matrix
    of vectors is required.
    '''
    requiredInPackedForm = svd.requiredInPackedForm
    notRequired = svd.notRequired

class SVD(object):
    '''
    Computes result of the SVD algorithm
    '''

    def __init__(self, n_components=2, verbose=0):
        '''
        Constructor
        '''
        self.n_components = n_components
        self.verbose = verbose
        self.components_ = None
        self.singular_values_ = None
        self.parameters = {'method': 'defaultDense',
                           'leftSingularMatrix': \
                           SingularValueParameter.requiredInPackedForm.value,
                           'rightSingularMatrix': \
                           SingularValueParameter.requiredInPackedForm.value}

    def fit(self, X, y=None):
        '''
        Calculate SVD
        :param X:
        :param y:
        '''
        _ = y
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        '''
        Fit SVD to X
        :param X: array-like shape n_samples x n_features(n_components)
        TODO@monika: sparse matrix
        :param y: None
        :return: self object, returns the transformer object
        '''
        _ = y
        hdd = IInput.HomogenousDaalData(X)
        input_type = hdd.informat
        column_lambda = lambda input_, components: input_[:, 0:components] if \
            components <= input_.shape[1] else input_
        if input_type == 'numpy':
            X = column_lambda(X, self.n_components)
        elif input_type == 'pandas':
            X = column_lambda(X.as_matrix(), self.n_components)
        else:
            pass #CSV column size is not supported

        Input = hdd.getNumericTable()

        algorithm = svd.Batch(method=svd.defaultDense,
                              leftSingularMatrix=
                              self.parameters['leftSingularMatrix'],
                              rightSingularMatrix=
                              self.parameters['rightSingularMatrix'])
        algorithm.input.set(svd.data, Input)

        # compute SVD decomposition
        result = algorithm.compute()
        U, Sigma, VT = result.get(svd.leftSingularMatrix), \
                        result.get(svd.singularValues), \
                        result.get(svd.rightSingularMatrix)

        # transform result to numpy array
        self._U = IInput.getNumpyArray(nT=U)
        self._Q = IInput.getNumpyArray(nT=VT)

        sigma = IInput.getNumpyArray(nT=Sigma)
        _, cols = sigma.shape
        self._w = sigma.reshape(cols,)

        # Calculate explained variance & explained variance ratio
        X_transformed = self._U * self._w
        self.explained_variance = exp_var = np.var(X_transformed, axis=0)
        #todo @Monika: support csr, crs
        full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        fit = self.fit(X)
        X_new = fit.U * fit.singular_values_
        return X_new

    def inverse_transform(self, X):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        return np.dot(X, self.components_)

    def _check_double(self, data, convert=True): #@UnusedVariable
        '''
        No check is needed, PyDAAL works already with embedded validation
        :param data:
        :param convert:
        '''
        _ = convert
        return data

    def _print_verbose(self, level, msg):
        if self.verbose > level:
            print(msg)
            sys.stdout.flush()

    @property
    def components_(self):
        return self._Q

    @property
    def explained_variance_(self):
        return self.explained_variance

    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ratio

    @property
    def singular_values_(self):
        return self._w

    @property
    def U(self):
        return self._U

    def get_params(self, *_):
        '''
        Get parameters for the estimator
        :param could be whatever, it's ignored, since there are known
            parameters
        :returns dict params: Parameter names mapped to their values
        '''
        return self.parameters

    def set_params(self, **params):
        """Set the parameters of this solver.
        :return: self
        """
        if not params:
            return self

        valid_params = self.get_params().keys()
        for key, value in params:
            if key not in valid_params:
                raise ValueError('Invalid parameters {} for estimator {}. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.'.
                                 format(key, self.__class__.__name__))
            elif key in valid_params:
                if key in ['leftSingularMatrix', 'rightSingularMatrix']:
                    valid_values = [x.value for x in SingularValueParameter]
                    if value.value not in valid_values:
                        raise ValueError('Invalid parameter {} for estimator '
                                         '{}. The valid values are: {}'.
                                         format(key, self.__class__.__name__,
                                                ', '.join(valid_params)))
                    else:
                        self.parameters[key] = value.value
            elif key == 'method':
                if value != 0:
                    raise ValueError('Invalid parameter {} for estimator {}. ' \
                                     'The only valid method is: 0.'.
                                     format(key, self.__class__.__name__))
        return self
