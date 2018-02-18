#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.base import BaseEstimator
from src.interface_py.h2o4gpu.solvers.daal_solver.data.IInput import HomogenousDaalData,\
    IInput
from daal.algorithms import svd
import numpy as np

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
                                           
