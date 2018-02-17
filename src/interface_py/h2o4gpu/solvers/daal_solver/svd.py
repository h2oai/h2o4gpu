#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.base import BaseEstimator
from src.interface_py.h2o4gpu.solvers.daal_solver.data.IInput import HomogenousDaalData
from daal.data_management import FileDataSource, DataSourceIface
from daal.algorithms import svd

__all__ = ['SVD']


class SVD(BaseEstimator):
    '''
    Computes result of the SVD algorithm
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    def fit_transform(self, X, y=None):
        
        Input = HomogenousDaalData(X).getNumericTable()
        
        algorithm = svd.Batch(method=svd.defaultDense)
        algorithm.input.set(svd.data, Input)
        
        # compute SVD decomposition
        result = algorithm.compute()
        