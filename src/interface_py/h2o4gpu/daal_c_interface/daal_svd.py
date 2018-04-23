import numpy as np
from daal_conf import (daal_lib,realpath,PrintNTP)
from daal_nt import DaalNT

class DaalSVD(object):
    def __init__(self, array, rows = 0, columns = 0):
        print("type array : ", type(array))
        if isinstance(array, (DaalNT)):
            self.NT = array
            self.obj = daal_lib.CreateDaalSVD(self.NT.obj)
        elif isinstance(array, (np.ndarray, np.generic)):
            print("numeric table is it SVD")
            self.NT = DaalNT(array, rows, columns)
            print("type of self.NT: ", type(self.NT))
            self.obj = daal_lib.CreateDaalSVD(self.NT.obj)
        elif isinstance(array, str):
            NT = DaalNT(array)
            self.NT = NT
            self.obj = daal_lib.CreateDaalSVD(self.NT.obj)
        else:
            print("Unsupported constructor for numeric table!")
    
    def __del__(self):
        return daal_lib.DeleteDaalSVD(self.obj)

    def fit(self):
        daal_lib.FitDaalSVD(self.obj)
        
    def getSigma(self):
        return daal_lib.GetDaalSVDSigma(self.obj)
    
    def getRightSingularMatrix(self):
        return daal_lib.GetDaalRightSingularMatrix(self.obj)
    
    def getLeftSingularMatrix(self):
        return daal_lib.GetDaalLeftSingularMatrix(self.obj)
        
if __name__ == '__main__':
    print("Testing Numeric SVD")
    
    array = np.array([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]], dtype= np.double)
    SVD = DaalSVD(array, array.shape[0], array.shape[1])
    filename = '{}/../datasets/data/svd.csv'.format(realpath)
    print("parsed filename: {}".format(filename))
    SVD_f = DaalSVD(filename)
    SVD_f.fit()
    sigmas = SVD_f.getSigma()
    PrintNTP(sigmas, "Calculated Sigma")
    right_singular_matrix = SVD_f.getRightSingularMatrix()
    PrintNTP(right_singular_matrix, "Right Singular Matrix")
    left_singular_matrix = SVD_f.getLeftSingularMatrix()
    PrintNTP(left_singular_matrix, "Left Singular Matrix")
    
    