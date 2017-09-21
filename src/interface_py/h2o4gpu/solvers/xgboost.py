# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

import xgboost as xgb
import numpy as np
from h2o4gpu.ensemble import RandomForestClassifierSklearn, \
    RandomForestRegressorSklearn, GradientBoostingClassifierSklearn, \
    GradientBoostingRegressorSklearn
from ..typecheck.typechecks import assert_is_type
from ..solvers.utils import _setter


class RandomForestClassifier(object):
    """H2O RandomForestClassifier Solver

    Selects between h2o4gpu.solvers.xgboost.RandomForestClassifier
    and h2o4gpu.ensemble.forest.RandomForestClassifierSklearn
    Documentation:
    import h2o4gpu.solvers ; help(h2o4gpu.xgboost.RandomForestClassifierO)
    help(h2o4gpu.ensemble.forest.RandomForestClassifierSklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(self,
                 n_estimators=10,  # h2o4gpu
                 criterion='gini',
                 max_depth=3,  # h2o4gpu
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,  # h2o4gpu
                 random_state=None,  # h2o4gpu
                 verbose=0,  # h2o4gpu
                 warm_start=False,
                 class_weight=None,
                 # XGBoost specific params
                 subsample=1.0,  # h2o4gpu
                 colsample_bytree=1.0,  # h2o4gpu
                 num_parallel_tree=100,  # h2o4gpu
                 tree_method='gpu_hist',  # h2o4gpu
                 n_gpus=-1,  # h2o4gpu
                 predictor='gpu_predictor',  # h2o4gpu
                 backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend
        assert_is_type(backend, str)

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'max_features',
                'max_leaf_nodes', 'min_impurity_decrease',
                'min_impurity_split', 'bootstrap', 'oob_score',
                'warm_start', 'class_weight'
            ]
            params = [
                criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, max_features,
                max_leaf_nodes, min_impurity_decrease,
                min_impurity_split, bootstrap, oob_score,
                warm_start, class_weight
            ]
            params_default = ['gini', 2, 1, 0.0, 'auto', None, 0.0,
                              None, True, False, 0, False]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    print("WARNING: The sklearn parameter " + params_string[i] +
                          " has been changed from default to " + str(param) +
                          ". Will run Sklearn RandomForestsClassifier.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        self.model_sklearn = RandomForestClassifierSklearn(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        # Parameters for random forest
        silent = False
        if verbose != 0:
            silent = True
        if random_state is None:
            random_state = 0
        self.model_h2o4gpu = xgb.XGBClassifier(
            n_estimators=n_estimators,  # h2o4gpu
            max_depth=max_depth,  # h2o4gpu
            n_jobs=n_jobs,  # h2o4gpu
            random_state=random_state,  # h2o4gpu
            num_parallel_tree=num_parallel_tree,
            tree_method=tree_method,
            n_gpus=n_gpus,
            predictor=predictor,
            silent=silent,
            num_round=1,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

        if self.do_sklearn:
            print("Running sklearn RandomForestClassifier")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu RandomForestClassifier")
            self.model = self.model_h2o4gpu

    def apply(self, X):
        print("WARNING: apply() is using sklearn")
        return self.model_sklearn.apply(X)

    def decision_path(self, X):
        print("WARNING: decision_path() is using sklearn")
        return self.model_sklearn.decision_path(X)

    def fit(self, X, y=None, sample_weight=None):
        res = self.model.fit(X, y, sample_weight)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            res = self.model.predict(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        res[res < 0.5] = 0
        res[res > 0.5] = 1
        self.set_attributes()
        return res.squeeze()

    def predict_log_proba(self, X):
        res = self.predict_proba(X)
        self.set_attributes()
        return np.log(res)

    def predict_proba(self, X):
        if self.do_sklearn:
            res = self.model.predict_proba(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  # Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """ Set attributes for class"""
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.estimators_ = oself.model.estimators_')
        s('oself.classes_ = oself.model.classes_')
        s('oself.n_classes_ = oself.model.n_classes_')
        s('oself.n_features_ = oself.model.n_features_')
        s('oself.n_outputs_ = oself.model.n_outputs_')
        s('oself.feature_importances_ = oself.model.feature_importances_')
        s('oself.oob_score_ = oself.model.oob_score_')
        s('oself.oob_decision_function_ = oself.model.oob_decision_function_')


class RandomForestRegressor(object):
    """H2O RandomForestRegressor Solver

    Selects between h2o4gpu.solvers.xgboost.RandomForestRegressor
    and h2o4gpu.ensemble.forest.RandomForestRegressorSklearn
    Documentation:
    import h2o4gpu.solvers ; help(h2o4gpu.xgboost.RandomForestRegressorO)
    help(h2o4gpu.ensemble.forest.RandomForestRegressorSklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(self,
                 n_estimators=10,  # h2o4gpu
                 criterion='mse',
                 max_depth=3,  # h2o4gpu
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,  # h2o4gpu
                 random_state=None,  # h2o4gpu
                 verbose=0,  # h2o4gpu
                 warm_start=False,
                 # XGBoost specific params
                 subsample=1.0,  # h2o4gpu
                 colsample_bytree=1.0,  # h2o4gpu
                 num_parallel_tree=100,  # h2o4gpu
                 tree_method='gpu_hist',  # h2o4gpu
                 n_gpus=-1,  # h2o4gpu
                 predictor='gpu_predictor',  # h2o4gpu
                 backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend
        assert_is_type(backend, str)

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'max_features',
                'max_leaf_nodes', 'min_impurity_decrease',
                'min_impurity_split', 'bootstrap', 'oob_score',
                'warm_start'
            ]
            params = [
                min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, max_features,
                max_leaf_nodes, min_impurity_decrease,
                min_impurity_split, bootstrap, oob_score,
                warm_start
            ]
            params_default = [2, 1, 0.0, 'auto', None, 0.0,
                              None, True, False, 0, False]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    print("WARNING: The sklearn parameter " + params_string[i] +
                          " has been changed from default to " + str(param) +
                          ". Will run Sklearn RandomForestRegressor.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        self.model_sklearn = RandomForestRegressorSklearn(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        # Parameters for random forest
        silent = False
        if verbose != 0:
            silent = True
        if random_state is None:
            random_state = 0
        self.model_h2o4gpu = xgb.XGBRegressor(
            n_estimators=n_estimators,  # h2o4gpu
            max_depth=max_depth,  # h2o4gpu
            n_jobs=n_jobs,  # h2o4gpu
            random_state=random_state,  # h2o4gpu
            num_parallel_tree=num_parallel_tree,
            tree_method=tree_method,
            n_gpus=n_gpus,
            predictor=predictor,
            silent=silent,
            num_round=1,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

        if self.do_sklearn:
            print("Running sklearn RandomForestRegressor")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu RandomForestRegressor")
            self.model = self.model_h2o4gpu

    def apply(self, X):
        print("WARNING: apply() is using sklearn")
        return self.model_sklearn.apply(X)

    def decision_path(self, X):
        print("WARNING: decision_path() is using sklearn")
        return self.model_sklearn.decision_path(X)

    def fit(self, X, y=None, sample_weight=None):
        res = self.model.fit(X, y, sample_weight)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            res = self.model.predict(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        res[res < 0.5] = 0
        res[res > 0.5] = 1
        self.set_attributes()
        return res.squeeze()

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  # Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """ Set attributes for class"""
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.estimators_ = oself.model.estimators_')
        s('oself.n_features_ = oself.model.n_features_')
        s('oself.n_outputs_ = oself.model.n_outputs_')
        s('oself.feature_importances_ = oself.model.feature_importances_')
        s('oself.oob_score_ = oself.model.oob_score_')
        s('oself.oob_prediction_  = oself.model.oob_prediction_')


class GradientBoostingClassifier(object):
    """H2O GradientBoostingClassifier Solver

    Selects between h2o4gpu.solvers.xgboost.GradientBoostingClassifier
    and h2o4gpu.ensemble.gradient_boosting.GradientBoostingClassifierSklearn
    Documentation:
    import h2o4gpu.solvers ; help(h2o4gpu.xgboost.GradientBoostingClassifierO)
    help(h2o4gpu.ensemble.gradient_boosting.GradientBoostingClassifierSklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(self,
                 loss='deviance',
                 learning_rate=0.1,  # h2o4gpu
                 n_estimators=100,  # h2o4gpu
                 subsample=1.0,  # h2o4gpu
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,  # h2o4gpu
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,  # h2o4gpu
                 max_features=None,
                 verbose=0,  # h2o4gpu
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto',
                 # XGBoost specific params
                 colsample_bytree=1.0,  # h2o4gpu
                 num_parallel_tree=100,  # h2o4gpu
                 tree_method='gpu_hist',  # h2o4gpu
                 n_gpus=-1,  # h2o4gpu
                 predictor='gpu_predictor',  # h2o4gpu
                 backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend
        assert_is_type(backend, str)

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'loss', 'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'min_impurity_decrease',
                'min_impurity_split', 'init',
                'max_features', 'max_leaf_nodes',
                'warm_start', 'presort'
            ]
            params = [
                loss, criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf,
                min_impurity_decrease, min_impurity_split, init,
                max_features, max_leaf_nodes,
                warm_start, presort
            ]
            params_default = ['deviance', 'friedman-mse', 2, 1, 0.0, 0.0, None,
                              None, 'auto', None, False, 'auto']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    print("WARNING: The sklearn parameter " + params_string[i] +
                          " has been changed from default to " + str(param) +
                          ". Will run Sklearn GradientBoostingClassifier.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        self.model_sklearn = GradientBoostingClassifierSklearn(
            loss=loss,
            learning_rate=learning_rate,  # h2o4gpu
            n_estimators=n_estimators,  # h2o4gpu
            subsample=subsample,  # h2o4gpu
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,  # h2o4gpu
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            init=init,
            random_state=random_state,  # h2o4gpu
            max_features=max_features,
            verbose=verbose,  # h2o4gpu
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            presort=presort)  # h2o4gpu)

        # Parameters for gbm
        silent = False
        if verbose != 0:
            silent = True
        if random_state is None:
            random_state = 0
        self.model_h2o4gpu = xgb.XGBClassifier(
            learning_rate=learning_rate,  # h2o4gpu
            n_estimators=n_estimators,  # h2o4gpu
            subsample=subsample,  # h2o4gpu
            max_depth=max_depth,  # h2o4gpu
            random_state=random_state,  # h2o4gpu
            silent=silent,  # h2o4gpu
            colsample_bytree=colsample_bytree,  # h2o4gpu
            num_parallel_tree=num_parallel_tree,  # h2o4gpu
            tree_method=tree_method,  # h2o4gpu
            n_gpus=n_gpus,  # h2o4gpu
            predictor=predictor,  # h2o4gpu
            backend=backend)  # h2o4gpu

        if self.do_sklearn:
            print("Running sklearn GradientBoostingClassifier")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu GradientBoostingClassifier")
            self.model = self.model_h2o4gpu

    def apply(self, X):
        print("WARNING: apply() is using sklearn")
        return self.model_sklearn.apply(X)

    def decision_function(self, X):
        print("WARNING: decision_path() is using sklearn")
        return self.model_sklearn.decision_function(X)

    def fit(self, X, y=None, sample_weight=None):
        res = self.model.fit(X, y, sample_weight)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            res = self.model.predict(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        res[res < 0.5] = 0
        res[res > 0.5] = 1
        self.set_attributes()
        return res.squeeze()

    def predict_log_proba(self, X):
        res = self.predict_proba(X)
        self.set_attributes()
        return np.log(res)

    def predict_proba(self, X):
        if self.do_sklearn:
            res = self.model.predict_proba(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  # Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def staged_decision_function(self, X):
        print("WARNING: staged_decision__function() is using sklearn")
        return self.model_sklearn.staged_decision_function(X)

    def staged_predict(self, X):
        print("WARNING: staged_predict() is using sklearn")
        return self.model_sklearn.staged_predict(X)

    def staged_predict_proba(self, X):
        print("WARNING: staged_predict_proba() is using sklearn")
        return self.model_sklearn.staged_predict_proba(X)

    def set_attributes(self):
        """ Set attributes for class"""
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.feature_importances_ = oself.model.feature_importances_')
        s('oself.oob_improvement_ = oself.model.oob_improvement_')
        s('oself.train_score_ = oself.model.train_score_')
        s('oself.loss_ = oself.model.loss_')
        s('oself.init = oself.model.init')
        s('oself.estimators_ = oself.model.estimators_')


class GradientBoostingRegressor(object):
    """H2O GradientBoostingRegressor Solver

    Selects between h2o4gpu.solvers.xgboost.GradientBoostingRegressor
    and h2o4gpu.ensemble.gradient_boosting.GradientBoostingRegressorSklearn
    Documentation:
    import h2o4gpu.solvers ; help(h2o4gpu.xgboost.GradientBoostingRegressorO)
    help(h2o4gpu.ensemble.gradient_boosting.GradientBoostingRegressorSklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(self,
                 loss='ls',
                 learning_rate=0.1,  # h2o4gpu
                 n_estimators=100,  # h2o4gpu
                 subsample=1.0,  # h2o4gpu
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,  # h2o4gpu
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,  # h2o4gpu
                 max_features=None,
                 alpha=0.9,
                 verbose=0,  # h2o4gpu
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto',
                 # XGBoost specific params
                 colsample_bytree=1.0,  # h2o4gpu
                 num_parallel_tree=100,  # h2o4gpu
                 tree_method='gpu_hist',  # h2o4gpu
                 n_gpus=-1,  # h2o4gpu
                 predictor='gpu_predictor',  # h2o4gpu
                 backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend
        assert_is_type(backend, str)

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'loss', 'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf',
                'min_impurity_decrease', 'min_impurity_split', 'init',
                'max_features', 'alpha', 'max_leaf_nodes',
                'warm_start', 'presort'
            ]
            params = [
                loss, criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf,
                min_impurity_decrease, min_impurity_split, init,
                max_features, alpha, max_leaf_nodes,
                warm_start, presort
            ]
            params_default = ['ls', 'friedman-mse', 2, 1, 0.0, 0.0, None,
                              None, 'auto', 0.9, None, False, 'auto']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    print("WARNING: The sklearn parameter " + params_string[i] +
                          " has been changed from default to " + str(param) +
                          ". Will run Sklearn GradientBoostingRegressor.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        self.model_sklearn = GradientBoostingRegressorSklearn(
            loss=loss,
            learning_rate=learning_rate,  # h2o4gpu
            n_estimators=n_estimators,  # h2o4gpu
            subsample=subsample,  # h2o4gpu
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,  # h2o4gpu
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            init=init,
            random_state=random_state,  # h2o4gpu
            max_features=max_features,
            alpha=alpha,
            verbose=verbose,  # h2o4gpu
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            presort=presort)  # h2o4gpu)

        # Parameters for gbm
        silent = False
        if verbose != 0:
            silent = True
        if random_state is None:
            random_state = 0
        self.model_h2o4gpu = xgb.XGBClassifier(
            learning_rate=learning_rate,  # h2o4gpu
            n_estimators=n_estimators,  # h2o4gpu
            subsample=subsample,  # h2o4gpu
            max_depth=max_depth,  # h2o4gpu
            random_state=random_state,  # h2o4gpu
            silent=silent,  # h2o4gpu
            colsample_bytree=colsample_bytree,  # h2o4gpu
            num_parallel_tree=num_parallel_tree,  # h2o4gpu
            tree_method=tree_method,  # h2o4gpu
            n_gpus=n_gpus,  # h2o4gpu
            predictor=predictor,  # h2o4gpu
            backend=backend)  # h2o4gpu

        if self.do_sklearn:
            print("Running sklearn GradientBoostingRegressor")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu GradientBoostingRegressor")
            self.model = self.model_h2o4gpu

    def apply(self, X):
        print("WARNING: apply() is using sklearn")
        return self.model_sklearn.apply(X)

    def fit(self, X, y=None, sample_weight=None):
        res = self.model.fit(X, y, sample_weight)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            res = self.model.predict(X)
            self.set_attributes()
            return res
        res = self.model.predict(X)
        res[res < 0.5] = 0
        res[res > 0.5] = 1
        self.set_attributes()
        return res.squeeze()

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  # Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def staged_predict(self, X):
        print("WARNING: staged_predict() is using sklearn")
        return self.model_sklearn.staged_predict(X)

    def set_attributes(self):
        """ Set attributes for class"""
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.feature_importances_ = oself.model.feature_importances_')
        s('oself.oob_improvement_ = oself.model.oob_improvement_')
        s('oself.train_score_ = oself.model.train_score_')
        s('oself.loss_ = oself.model.loss_')
        s('oself.init = oself.model.init')
        s('oself.estimators_ = oself.model.estimators_')
