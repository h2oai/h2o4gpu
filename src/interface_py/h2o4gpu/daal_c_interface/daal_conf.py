from ctypes import cdll, POINTER, c_int, c_double, c_void_p, c_char_p
import os
c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)


realpath = os.path.dirname(__file__)
libpath = "{}/{}".format(realpath, 'libh2o4gpu_daal.so')
print("libpath: ", libpath)
daal_lib = cdll.LoadLibrary(libpath)

daal_lib.CreateDaalInput.restype = c_void_p
daal_lib.CreateDaalInput.argtypes = [c_double_p, c_int, c_int]
daal_lib.CreateDaalInputFeaturesDependent.restype = c_void_p
daal_lib.CreateDaalInputFeaturesDependent.argtypes = [c_double_p, c_int, c_int, c_double_p, c_int, c_int]
daal_lib.GetFeaturesData.restype = c_void_p
daal_lib.GetFeaturesData.argtypes = [c_void_p]
daal_lib.GetDependentTable.restype = c_void_p
daal_lib.GetDependentTable.argtypes = [c_void_p]
daal_lib.DeleteDaalInput.argtypes = [c_void_p]
daal_lib.PrintDaalNumericTablePtr.argtypes = [c_void_p, c_char_p, c_int, c_int]
daal_lib.CreateDaalInputFile.restype = c_void_p
daal_lib.CreateDaalInputFile.argtypes = [c_char_p]
daal_lib.CreateDaalInputFileFeaturesDependent.restype = c_void_p
daal_lib.CreateDaalInputFileFeaturesDependent.argtypes = [c_char_p, c_int, c_int]
daal_lib.PrintNTP.argtypes = [c_void_p, c_char_p, c_int, c_int]
# SVD
daal_lib.CreateDaalSVD.restype = c_void_p
daal_lib.CreateDaalSVD.argtypes = [c_void_p]
daal_lib.DeleteDaalSVD.argtypes = [c_void_p]
daal_lib.FitDaalSVD.argtypes = [c_void_p]
daal_lib.GetDaalSVDSigma.restype = c_void_p
daal_lib.GetDaalSVDSigma.argtypes = [c_void_p]
daal_lib.GetDaalRightSingularMatrix.restype = c_void_p
daal_lib.GetDaalRightSingularMatrix.argtypes = [c_void_p]
daal_lib.GetDaalLeftSingularMatrix.restype = c_void_p
daal_lib.GetDaalLeftSingularMatrix.argtypes = [c_void_p]
# Ridge Rebression
daal_lib.CreateDaalRidgeRegression.restype = c_void_p
daal_lib.CreateDaalRidgeRegression.argtypes = [c_void_p]
daal_lib.DeleteDaalRidgeRegression.argtypes = [c_void_p]
daal_lib.TrainDaalRidgeRegression.argtypes = [c_void_p]
daal_lib.GetDaalRidgeRegressionBeta.restype = c_void_p
daal_lib.GetDaalRidgeRegressionBeta.argtypes = [c_void_p]
daal_lib.PredictDaalRidgeRegression.argtypes = [c_void_p, c_void_p]
daal_lib.GetDaalRidgeRegressionPredictionData.restype = c_void_p
daal_lib.GetDaalRidgeRegressionPredictionData.argtypes = [c_void_p]
# Linear Regression
daal_lib.CreateDaalLinearRegression.restype = c_void_p
daal_lib.CreateDaalLinearRegression.argtypes = [c_void_p]
daal_lib.DeleteDaalLinearRegression.argtypes = [c_void_p]
daal_lib.TrainDaalLinearRegression.argtypes = [c_void_p]
daal_lib.GetDaalLinearRegressionBeta.argtypes = [c_void_p]
daal_lib.GetDaalLinearRegressionBeta.restype = c_void_p
daal_lib.PredictDaalLinearRegression.argtypes = [c_void_p, c_void_p]
daal_lib.GetDaalLinearRegressionPredictionData.restype = c_void_p
daal_lib.GetDaalLinearRegressionPredictionData.argtypes = [c_void_p]

def PrintNTP(NTP, msg = "", rows = 0, cols = 0):
    daal_lib.PrintNTP(NTP, bytes(msg, "UTF-8"), rows, cols)

__all__ = ['c_double_p','daal_lib','realpath', 'PrintNTP']