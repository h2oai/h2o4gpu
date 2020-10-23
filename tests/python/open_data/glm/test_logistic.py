import h2o4gpu
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np


def test_not_labels():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # convert class values to [0,2]
    # y = y * 2

    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # sklearn
    clf_sklearn = linear_model.LogisticRegression(solver='liblinear')
    clf_sklearn.fit(X_train, y_train)
    y_pred_sklearn = clf_sklearn.predict(X_test)

    # h2o
    clf_h2o = h2o4gpu.LogisticRegression()
    clf_h2o.fit(X_train, y_train)
    y_pred_h2o = clf_h2o.predict(X_test)

    assert np.allclose(accuracy_score(y_test, y_pred_sklearn), accuracy_score(y_test, y_pred_h2o.squeeze()))

