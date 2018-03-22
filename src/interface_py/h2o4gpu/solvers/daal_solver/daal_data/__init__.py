#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

import numpy as np
from daal.data_management import (BlockDescriptor, readOnly, NumericTable)

def getNumpyArray(nT):  # @DontTrace
    '''
    returns Numpy array
    :param nT: daal numericTable as input
    :return: numpy array
    '''
    if not isinstance(nT, NumericTable):
        raise ValueError("getNumpyError, nT is not Numeric table, but {}".
                         format(str(type(nT))))

    block = BlockDescriptor()
    nT.getBlockOfRows(0, nT.getNumberOfRows(), readOnly, block)
    np_array = block.getArray()
    nT.releaseBlockOfRows(block)
    return np_array

def getNumpyShape(nP):
    '''
    returns Numpy shape
    :param nP:
    :return: shape
    '''
    try:
        return (nP.shape[0], nP.shape[1])
    except IndexError:
        return (1, nP.shape[0])
