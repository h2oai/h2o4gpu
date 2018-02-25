#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.base import BaseEstimator
from src.interface_py.h2o4gpu.solvers.daal_solver.daal_data.IInput import HomogenousDaalData,\
    IInput
from daal.algorithms import svd
import numpy as np
from h2o4gpu.utils.validation import check_array

__all__ = ['SVD']


class SVD(BaseEstimator):
    '''
    Computes result of the SVD algorithm
    '''


    def __init__(self, n_components=2, params):
        '''
        Constructor
        '''
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
    
    def fit(self, X, y=None):
        
        self.fit_transform(X)
        return self
    
    def fit_transform(self, X, y=None):
        '''
        Fit SVD to X
        :param X: array-like shape n_samples x n_features(n_components) TODO@monika: sparse matrix
        :param y: None
        :return: self object, returns the transformer object
        '''
        
        Input = HomogenousDaalData(X).getNumericTable()
        
        algorithm = svd.Batch(method=svd.defaultDense)
        algorithm.input.set(svd.data, Input)
        
        # compute SVD decomposition
        result = algorithm.compute()
        U, Sigma, VT = result.get(svd.leftSingularMatrix), result.get(svd.singularValues), result.get(svd.rightSingularMatrix)
        
        # transform result to numpy array
        lU = IInput.getNumpyArray(nT=U)
        self.components_ = IInput.getNumpyArray(nT=VT)
        self.singular_values_ = IInput.getNumpyArray(nT=Sigma)
        
        # Calculate explaiend variance & explained variance ratio
        X_transformed = lU * self.singular_values_
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
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T)

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
        X = check_array(X)
        return np.dot(X, self.components_)   
