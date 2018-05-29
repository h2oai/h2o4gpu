import numpy as np
from daal_conf import (c_double_p,daal_lib,realpath,PrintNTP)

class DaalNT(object):
    def __init__(self, array, rows=0, columns=0):
        if isinstance(array, (np.ndarray, np.generic)):
            p_array = array.ctypes.data_as(c_double_p)
            self.array = array
            self.rows = rows
            self.columns = columns
            self.obj = daal_lib.CreateDaalInput(p_array, rows, columns)
        elif isinstance(array, str):
            b_str = bytes(array, 'UTF-8')
            self.obj = daal_lib.CreateDaalInputFile(b_str)
        else:
            print("Unsupported constructor for numeric table!")
    
    def __del__(self):
        return daal_lib.DeleteDaalInput(self.obj)
    
    def print_NT(self, msg="", rows = 0, columns = 0):
        daal_lib.PrintDaalNumericTablePtr(self.obj, bytes(msg, 'UTF-8'), rows, columns)
        
# with features and dependent data
class DaalNTs(object):
    def __init__(self, features, f_rows, f_cols, dependent=None, d_rows=0, d_cols=0):
        if dependent is not None:
            if isinstance(features, (np.ndarray, np.generic)) and isinstance(dependent, (np.ndarray, np.generic)):
                self.features = features
                self.dependent = dependent
                self.features_rows = f_rows
                self.features_cols = f_cols
                self.dependent_rows = d_rows
                self.dependent_cols = d_cols
                p_features = features.ctypes.data_as(c_double_p)
                p_deps = dependent.ctypes.data_as(c_double_p)
                
                self.obj = daal_lib.CreateDaalInputFeaturesDependent(p_features, f_rows, f_cols, p_deps, d_rows, d_cols)
            else:
                raise("Invalid DaalNTs constructor")
        else:
            if isinstance(features, str):
                b_str = bytes(features, "UTF-8")
                self.obj = daal_lib.CreateDaalInputFileFeaturesDependent(b_str, f_rows, f_cols)
        
    def __del__(self):
        return daal_lib.DeleteDaalInput(self.obj)

    def getFeaturesData(self):
        return daal_lib.GetFeaturesData(self.obj)
    
    def getDependentData(self):
        return daal_lib.GetDependentTable(self.obj)


if __name__ == '__main__':
    print("1 Testing DaalNT class:")
    print("1.1 Testing Numeric Table Object")
    array = np.array([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]], dtype = np.double)
   
    NT = DaalNT(array, array.shape[0], array.shape[1])
    NT.print_NT("Numeric table object from numpy Array")
    
    filename = '{}/../datasets/data/svd.csv'.format(realpath)
    print("parsed filename: {}".format(filename))
    NT_f = DaalNT(filename)
    NT_f.print_NT("Numeric Table object from file", 10, 10)
    
    print("1.2 Testing Numeric Tables with Features and Dependent data")
    array_d = np.array([[1.,1.5],[1.5,1.5],[2,2.5]], dtype = np.double)
    
    NT2 = DaalNTs(array, array.shape[0], array.shape[1], array_d, array_d.shape[0], array_d.shape[1])
    features = NT2.getFeaturesData()
    PrintNTP(features, "features data")
    dependent = NT2.getDependentData()
    PrintNTP(dependent, "dependent data")
    
    print("1.3 Testing Numeric Tables with CSV file")
    training_data = '{}/../datasets/data/linear_regression_train.csv'.format(realpath)
    NT3 = DaalNTs(training_data, 10, 2)
    PrintNTP(NT3.getDependentData(), "dependent data from csv - 2 cols")
    PrintNTP(NT3.getFeaturesData(), "features data from csv - 10 cols")
    
    
    