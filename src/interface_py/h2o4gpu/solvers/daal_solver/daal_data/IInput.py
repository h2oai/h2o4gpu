#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

#import abc
import os
import numpy as np
import pandas as pd
from daal.data_management import (AOSNumericTable, FileDataSource,
                                  DataSource, HomogenNumericTable)

class IInput(object):
    '''
    Abstract class for generic input data in daal library
    '''

    #@abc.abstractclassmethod
    def getNumericTable(self, **kwargs):
        raise NotImplementedError()


class HomogenousDaalData(IInput):
    '''
    Converts numpy, pandas, csv-file to daal_solver NumericTables
    e.g. np.array([[1,2,3],[4,5,6]])
         pd.DataFrame(values)
         'example.csv'
    '''

    def __init__(self, indata=None):
        self.indata = indata
        if self.indata is not None:
            self._categorize(indata)

    def __call__(self, indata):
        if indata is not None and indata is not self.indata:
            self._categorize(indata)

    def _categorize(self, indata):
        '''
        decide what data type is input
        :param indata:
        '''

        if isinstance(indata, np.ndarray):
            self.informat = 'numpy'
        elif isinstance(indata, pd.DataFrame):
            self.informat = 'pandas'
        elif isinstance(indata, str):
            if os.path.isfile(input):
                self.informat = 'csv'
            else:
                raise ValueError("DaalData error in intialization,\
                no valid format given.")
        else:
            raise ValueError("DaalData error in intialization,\
            no valid format given.")
        self.indata = indata

    def getNumericTable(self, **kwargs):
        if self.informat == 'numpy':
            return HomogenNumericTable(self.indata)
        elif self.informat == 'pandas':
            array = self.indata.as_matrix()
            return HomogenNumericTable(array)
        elif self.informat == 'csv':
            dataSource =  \
            FileDataSource(self.indata,
                           DataSource.doAllocateNumericTable,
                           DataSource.doDictionaryFormContext)
            dataSource.loadDataBlock()
            return dataSource.getNumericTable()
        else:
            raise ValueError("Cannot identify input type.")


class HeterogenousDaalData(HomogenousDaalData):
    '''
    Heterogenous data with numpy:
    np.array([(1,2.3),(2,-1,-0.9)],
    dtype=('x',np.float32), ('y', np.float64)])
    '''

    def __init__(self, indata=None):
        HomogenousDaalData.__init__(indata)

    def __call__(self, indata):
        HomogenousDaalData.__call__(self, indata)

    def _getStructureArray(self, dataframe, dtypes):
        '''
        :param dataframe:
        :param dtypes:
        :output structured numpy array
        '''

        dataList = []
        for i in range(dataframe.shape[0]):
            dataList.append(tuple(dataframe.loc[i]))
        decDtype = list(zip(dataframe.columns.tolist(), dtypes))
        array = np.array(dataList, dtype=decDtype)
        return array

    def getNumericTable(self, **kwargs):
        if self.informat == 'numpy':
            return AOSNumericTable(self.indata)
        elif self.informat == 'pandas':
            array = self._getStructureArray(
                self.indata,
                dtypes=self.indata.dtypes)
            return AOSNumericTable(array)
        elif self.informat == 'csv':
            dataSource = FileDataSource(
                self.indata,
                DataSource.notAllocateNumericTable,
                DataSource.doDictionaryFromContext)

            if 'nRows' not in kwargs and 'dtype' not in kwargs:
                raise ValueError("HeterogenousDaalData, for csv file, \
                'nrows' and 'dtypes' must be specified.")
            nRows = kwargs['nRows']
            dtype = kwargs['dtype']
            array = np.empty([nRows,], dtype=dtype)
            nT = AOSNumericTable(array)
            return dataSource.loadDataBlock(nRows, nT)
        else:
            return None
