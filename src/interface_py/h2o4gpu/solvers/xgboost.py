#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""


class RandomForestClassifier():

    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        pass


#TODO:
#num_parallel_tree = 100,
#subsample = 0.5,
#colsample_bytree = 0.5,
#nrounds = 1,


class RandomForestRegressor():

    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        #TODO
        pass


class GradientBoostingClassifier():

    def __init__(self,
                 loss='deviance',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto'):
        #TODO
        pass


class GradientBoostingRegressor():

    def __init__(self,
                 loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto'):
        #TODO
        pass
