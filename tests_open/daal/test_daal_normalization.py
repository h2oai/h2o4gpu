# -*- encoding: utf-8 -*-
"""
ElasticNetH2O solver tests using Kaggle datasets.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np
import h2o4gpu
import logging
from scipy import stats
from daal.data_management import HomogenNumericTable
from h2o4gpu.solvers.daal_solver.normalize import zscore as z_score
from h2o4gpu.solvers.daal_solver.daal_data import IInput
from numpy.ma.testutils import assert_array_almost_equal

logging.basicConfig(level=logging.DEBUG)

def test_zscore_single():

    input = np.random.rand(10,1)
    sc_zscore = stats.zscore(input, axis=0, ddof=1)
    
    da_input = HomogenNumericTable(input)
    da_zscore = z_score(da_input)
    np_da_zscore = IInput.getNumpyArray(da_zscore)
    
    assert_array_almost_equal(sc_zscore, np_da_zscore)

def test_zscore_multicolumns():

    input = np.random.rand(10,3)
    sc_zscore = stats.zscore(input, axis=0, ddof=1)
    
    da_input = HomogenNumericTable(input)
    da_zscore = z_score(da_input)
    np_da_zscore = IInput.getNumpyArray(da_zscore)
    
    assert_array_almost_equal(sc_zscore, np_da_zscore)

def test_zscore_vector(): test_zscore_single()
def test_zscore_matrix(): test_zscore_multicolumns()

if __name__ == '__main__':
    test_zscore()
