import sys
import numpy as np
import logging
import pandas as pd

print(sys.path)


logging.basicConfig(level=logging.DEBUG)


# Function to check fall back to sklearn
def test_drf_regressor_backupsklearn(backend='auto'):
    df = pd.read_csv("./open_data/simple.txt", delim_whitespace=True)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    import h2o4gpu
    Solver = h2o4gpu.RandomForestRegressor

    #Run h2o4gpu version of RandomForest Regression
    drf = Solver(backend=backend, random_state=1234, oob_score=True, n_estimators=10)
    print("h2o4gpu fit()")
    drf.fit(X, y)

    #Run Sklearn version of RandomForest Regression
    from h2o4gpu.ensemble import RandomForestRegressorSklearn
    drf_sk = RandomForestRegressorSklearn(random_state=1234, oob_score=True, max_depth=3, n_estimators=10)
    print("Scikit fit()")
    drf_sk.fit(X, y)

    if backend == "sklearn":
        assert (drf.predict(X) == drf_sk.predict(X)).all() == True
        assert (drf.score(X, y) == drf_sk.score(X, y)).all() == True
        assert (drf.decision_path(X)[1] == drf_sk.decision_path(X)[1]).all() == True
        assert (drf.apply(X) == drf_sk.apply(X)).all() == True

        print("Estimators")
        print(drf.estimators_)
        print(drf_sk.estimators_)

        print("n_features")
        print(drf.n_features_)
        print(drf_sk.n_features_)
        assert drf.n_features_ == drf_sk.n_features_

        print("n_outputs")
        print(drf.n_outputs_)
        print(drf_sk.n_outputs_)
        assert drf.n_outputs_ == drf_sk.n_outputs_

        print("Feature importance")
        print(drf.feature_importances_)
        print(drf_sk.feature_importances_)
        assert (drf.feature_importances_ == drf_sk.feature_importances_).all() == True

        print("oob_score")
        print(drf.oob_score_)
        print(drf_sk.oob_score_)
        assert drf.oob_score_ == drf_sk.oob_score_

        print("oob_prediction")
        print(drf.oob_prediction_)
        print(drf_sk.oob_prediction_)
        assert (drf.oob_prediction_ == drf_sk.oob_prediction_).all() == True

def test_drf_classifier_backupsklearn(backend='auto'):
    df = pd.read_csv("./open_data/creditcard.csv")
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    import h2o4gpu
    Solver = h2o4gpu.RandomForestClassifier

    #Run h2o4gpu version of RandomForest Regression
    drf = Solver(backend=backend, random_state=1234, oob_score=True, n_estimators=10)
    print("h2o4gpu fit()")
    drf.fit(X, y)

    #Run Sklearn version of RandomForest Regression
    from h2o4gpu.ensemble import RandomForestClassifierSklearn
    drf_sk = RandomForestClassifierSklearn(random_state=1234, oob_score=True, max_depth=3, n_estimators=10)
    print("Scikit fit()")
    drf_sk.fit(X, y)

    if backend == "sklearn":
        assert (drf.predict(X) == drf_sk.predict(X)).all() == True
        assert (drf.predict_log_proba(X) == drf_sk.predict_log_proba(X)).all() == True
        assert (drf.predict_proba(X) == drf_sk.predict_proba(X)).all() == True
        assert (drf.score(X, y) == drf_sk.score(X, y)).all() == True
        assert (drf.decision_path(X)[1] == drf_sk.decision_path(X)[1]).all() == True
        assert (drf.apply(X) == drf_sk.apply(X)).all() == True

        print("Estimators")
        print(drf.estimators_)
        print(drf_sk.estimators_)

        print("n_features")
        print(drf.n_features_)
        print(drf_sk.n_features_)
        assert drf.n_features_ == drf_sk.n_features_

        print("n_classes_")
        print(drf.n_classes_)
        print(drf_sk.n_classes_)
        assert drf.n_classes_ == drf_sk.n_classes_

        print("n_features")
        print(drf.classes_)
        print(drf_sk.classes_)
        assert (drf.classes_ == drf_sk.classes_).all() == True

        print("n_outputs")
        print(drf.n_outputs_)
        print(drf_sk.n_outputs_)
        assert drf.n_outputs_ == drf_sk.n_outputs_

        print("Feature importance")
        print(drf.feature_importances_)
        print(drf_sk.feature_importances_)
        assert (drf.feature_importances_ == drf_sk.feature_importances_).all() == True

        print("oob_score")
        print(drf.oob_score_)
        print(drf_sk.oob_score_)
        assert drf.oob_score_ == drf_sk.oob_score_

# Function to check fall back to sklearn
def test_gbm_regressor_backupsklearn(backend='auto'):
    df = pd.read_csv("./open_data/simple.txt", delim_whitespace=True)
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    import h2o4gpu
    Solver = h2o4gpu.GradientBoostingRegressor

    #Run h2o4gpu version of RandomForest Regression
    gbm = Solver(backend=backend, random_state=1234)
    print("h2o4gpu fit()")
    gbm.fit(X, y)

    #Run Sklearn version of RandomForest Regression
    from h2o4gpu.ensemble import GradientBoostingRegressorSklearn
    gbm_sk = GradientBoostingRegressorSklearn(random_state=1234, max_depth=3)
    print("Scikit fit()")
    gbm_sk.fit(X, y)

    if backend == "sklearn":
        assert (gbm.predict(X) == gbm_sk.predict(X)).all() == True
        print((a == b for a, b in zip(gbm.staged_predict(X), gbm_sk.staged_predict(X))))
        assert np.allclose(list(gbm.staged_predict(X)), list(gbm_sk.staged_predict(X)))
        assert (gbm.score(X, y) == gbm_sk.score(X, y)).all() == True
        assert (gbm.apply(X) == gbm_sk.apply(X)).all() == True
        
        print("Estimators")
        print(gbm.estimators_)
        print(gbm_sk.estimators_)
        
        print("loss")
        print(gbm.loss_)
        print(gbm_sk.loss_)
        assert gbm.loss_.__dict__ == gbm_sk.loss_.__dict__
        
        print("init_")
        print(gbm.init)
        print(gbm_sk.init)

        print("Feature importance")
        print(gbm.feature_importances_)
        print(gbm_sk.feature_importances_)
        assert (gbm.feature_importances_ == gbm_sk.feature_importances_).all() == True
        
        print("train_score_")
        print(gbm.train_score_)
        print(gbm_sk.train_score_)
        assert (gbm.train_score_ == gbm_sk.train_score_).all() == True


# Function to check fall back to sklearn
def test_gbm_classifier_backupsklearn(backend='auto'):
    df = pd.read_csv("./open_data/creditcard.csv")
    X = np.array(df.iloc[:, :df.shape[1] - 1], dtype='float32', order='C')
    y = np.array(df.iloc[:, df.shape[1] - 1], dtype='float32', order='C')
    import h2o4gpu
    Solver = h2o4gpu.GradientBoostingClassifier

    # Run h2o4gpu version of RandomForest Regression
    gbm = Solver(backend=backend, random_state=1234)
    print("h2o4gpu fit()")
    gbm.fit(X, y)

    # Run Sklearn version of RandomForest Regression
    from h2o4gpu.ensemble import GradientBoostingClassifierSklearn
    gbm_sk = GradientBoostingClassifierSklearn(random_state=1234, max_depth=3)
    print("Scikit fit()")
    gbm_sk.fit(X, y)

    if backend == "sklearn":
        assert (gbm.predict(X) == gbm_sk.predict(X)).all() == True
        assert (gbm.predict_log_proba(X) == gbm_sk.predict_log_proba(X)).all() == True
        assert (gbm.predict_proba(X) == gbm_sk.predict_proba(X)).all() == True
        assert (gbm.score(X, y) == gbm_sk.score(X, y)).all() == True
        assert (gbm.decision_function(X)[1] == gbm_sk.decision_function(X)[1]).all() == True
        assert np.allclose(list(gbm.staged_predict(X)), list(gbm_sk.staged_predict(X)))
        assert np.allclose(list(gbm.staged_predict_proba(X)), list(gbm_sk.staged_predict_proba(X)))
        assert (gbm.apply(X) == gbm_sk.apply(X)).all() == True

        print("Estimators")
        print(gbm.estimators_)
        print(gbm_sk.estimators_)

        print("loss")
        print(gbm.loss_)
        print(gbm_sk.loss_)
        assert gbm.loss_.__dict__ == gbm_sk.loss_.__dict__

        print("init_")
        print(gbm.init)
        print(gbm_sk.init)

        print("Feature importance")
        print(gbm.feature_importances_)
        print(gbm_sk.feature_importances_)
        assert (gbm.feature_importances_ == gbm_sk.feature_importances_).all() == True

        print("train_score_")
        print(gbm.train_score_)
        print(gbm_sk.train_score_)
        assert (gbm.train_score_ == gbm_sk.train_score_).all() == True



def test_sklearn_drf_regression(): test_drf_regressor_backupsklearn()
def test_sklearn_drf_regression_sklearn(): test_drf_regressor_backupsklearn(backend='sklearn')
def test_sklearn_drf_regression_h2o4gpu(): test_drf_regressor_backupsklearn(backend='h2o4gpu')

def test_sklearn_drf_classification(): test_drf_classifier_backupsklearn()
def test_sklearn_drf_classification_sklearn(): test_drf_classifier_backupsklearn(backend='sklearn')
def test_sklearn_drf_regression_h2o4gpu(): test_drf_classifier_backupsklearn(backend='h2o4gpu')

def test_sklearn_gbm_classification(): test_gbm_classifier_backupsklearn()
def test_sklearn_gbm_classification_sklearn(): test_gbm_classifier_backupsklearn(backend='sklearn')
def test_sklearn_gbm_regression_h2o4gpu(): test_gbm_classifier_backupsklearn(backend='h2o4gpu')

def test_sklearn_gbm_regression(): test_gbm_regressor_backupsklearn()
def test_sklearn_gbm_regression_sklearn(): test_gbm_regressor_backupsklearn(backend='sklearn')
def test_sklearn_gbm_regression_h2o4gpu(): test_gbm_regressor_backupsklearn(backend='h2o4gpu')


