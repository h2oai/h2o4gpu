#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from daal.algorithms.normalization import minmax
from daal.algorithms.normalization import zscore as daal_zscore

def min_max(nT, lBound=-1.0, uBound=1.0):
    '''
    returns scaled data
    :param nT: numericTable data format
    :param lBound:
    :param uBound:
    :return normalized data
    '''

    algorithm = minmax.Batch(method=minmax.defaultDense)
    algorithm.parameter.lowerBound = lBound
    algorithm.parameter.upperBound = uBound
    algorithm.input.set(minmax.data, nT)
    result = algorithm.compute()
    return result.get(minmax.normalizedData)


def zscore(nT):
    '''
    :param nT: nuemricTable data format
    :return normalized data
    '''

    algorithm = daal_zscore.Batch(method=daal_zscore.defaultDense)
    algorithm.input.set(daal_zscore.data, nT)
    result = algorithm.compute()
    return result.get(daal_zscore.normalizedData)
