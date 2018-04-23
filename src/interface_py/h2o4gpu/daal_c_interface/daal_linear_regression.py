import numpy as np
from daal_conf import (daal_lib,realpath,PrintNTP)
from daal_nt import DaalNTs

class DaalLinearRegression(object):
    def __init__(self, features, f_rows=0, f_columns=0, dependent=None, d_rows=0, d_columns=0):
        if not dependent and features:
            if isinstance(features, (DaalNTs)):
                f_obj = features.obj
                self.features = f_obj.getFeaturesData()
                self.dependent = f_obj.getDependentData()
                self.NT = features
                self.obj = daal_lib.CreateDaalLinearRegression(f_obj)
            elif isinstance(features, str):
                print('handling traing file')
                self.training_file = features
                self.features = f_rows
                self.dependent = f_columns
                NT = DaalNTs(features, f_rows, f_columns)
                self.NT = NT
                self.obj = daal_lib.CreateDaalLinearRegression(NT.obj)
        elif dependent is not None and features is not None:
            if isinstance(dependent, (np.ndarray, np.generic)) and isinstance(features, (np.ndarray, np.generic)):
                NTs = DaalNTs(features, f_rows, f_columns, dependent, d_rows, d_columns)
                f_obj = NTs.obj
                self.NT = NTs
                self.obj = daal_lib.CreateDaalLinearRegression(f_obj)
        else:
                print("Unsupported constructor for numeric table!")
    
    def __del__(self):
        return daal_lib.DeleteDaalLinearRegression(self.obj)

    def train(self):
        daal_lib.TrainDaalLinearRegression(self.obj)
        
    def getBeta(self):
        return daal_lib.GetDaalLinearRegressionBeta(self.obj)
 
    def predict(self, dependent_file, n_features, n_dependent):
        if isinstance(dependent_file, str):
            NT = DaalNTs(dependent_file, n_features, n_dependent)
            daal_lib.PredictDaalLinearRegression(self.obj, NT.obj)
    
    def getPrediction(self):
        return daal_lib.GetDaalLinearRegressionPredictionData(self.obj)
 
if __name__ == '__main__':
    print("Testing Linear Regression")
    
    training_data = '{}/../datasets/data/linear_regression_train.csv'.format(realpath)
    testing_data = '{}/../datasets/data/linear_regression_test.csv'.format(realpath)
    n_features = 10
    n_dependent = 2
    dlr = DaalLinearRegression(training_data, n_features, n_dependent)
    dlr.train()
    beta = dlr.getBeta()
    print(beta)
    PrintNTP(beta, "Calculated Beta")
    dlr.predict(testing_data, n_features, n_dependent)
    prediction_data = dlr.getPrediction()
    PrintNTP(prediction_data, "Predicted Data")
    
    