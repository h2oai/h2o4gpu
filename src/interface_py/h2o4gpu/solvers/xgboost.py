# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""


class RandomForestClassifier(object):
    """H2O RandomForestClassifier Solver

    This estimator selects between h2o4gpu.solvers.xgboost.RandomForestClassifier
    and h2o4gpu.ensemble.forest.RandomForestClassifierSklearn

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    subsample : float
        Subsample ratio of the training instance.

    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.

    num_parallel_tree : int
        Number of trees to grow per round

    tree_method : string [default=’auto’]
        The tree construction algorithm used in XGBoost
        Distributed and external memory version only support approximate algorithm.
        Choices: {‘auto’, ‘exact’, ‘approx’, ‘hist’, ‘gpu_exact’, ‘gpu_hist’}
            ‘auto’: Use heuristic to choose faster one.
                - For small to medium dataset, exact greedy will be used.
                - For very large-dataset, approximate algorithm will be chosen.
                - Because old behavior is always use exact greedy in single machine,
                - user will get a message when approximate algorithm is chosen to
                  notify this choice.
            ‘exact’: Exact greedy algorithm.
            ‘approx’: Approximate greedy algorithm using sketching and histogram.
            ‘hist’: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
            ‘gpu_exact’: GPU implementation of exact algorithm.
            ‘gpu_hist’: GPU implementation of hist algorithm.

    n_gpus : int
        Number of gpu's to use in RandomForestClassifier solver. Default is -1.

    predictor : string [default='gpu_predictor']
        The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
            - 'cpu_predictor': Multicore CPU prediction algorithm.
            - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
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
            num_parallel_tree=1,  # h2o4gpu
            tree_method='gpu_hist',  # h2o4gpu
            n_gpus=-1,  # h2o4gpu
            predictor='gpu_predictor',  # h2o4gpu
            backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes',
                'min_impurity_decrease', 'min_impurity_split', 'bootstrap',
                'oob_score', 'class_weight'
            ]
            params = [
                criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, max_features, max_leaf_nodes,
                min_impurity_decrease, min_impurity_split, bootstrap, oob_score,
                class_weight
            ]
            params_default = [
                'gini', 2, 1, 0.0, 'auto', None, 0.0, None, True, False, None
            ]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose > 0:
                        print(
                            "WARNING: The sklearn parameter " + params_string[i]
                            + " has been changed from default to " + str(param)
                            + ". Will run Sklearn RandomForestsClassifier.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        from h2o4gpu.ensemble import RandomForestClassifierSklearn
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
        silent = True
        if verbose != 0:
            silent = False
        if random_state is None:
            random_state = 0

        import xgboost as xgb
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
            colsample_bytree=colsample_bytree)

        if self.do_sklearn:
            if verbose > 0:
                print("Running sklearn RandomForestClassifier")
            self.model = self.model_sklearn
        else:
            if verbose > 0:
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
        self.set_attributes()
        return res.squeeze()

    def predict_log_proba(self, X):
        res = self.predict_proba(X)
        self.set_attributes()
        import numpy as np
        return np.log(res)

    def predict_proba(self, X):
        if self.do_sklearn:
            res = self.model.predict_proba(X)
            self.set_attributes()
            return res
        res = self.model.predict_proba(X)
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
        from ..solvers.utils import _setter
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

    This estimator selects between h2o4gpu.solvers.xgboost.RandomForestRegressor
    and h2o4gpu.ensemble.forest.RandomForestRegressorSklearn

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    subsample : float
        Subsample ratio of the training instance.

    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.

    num_parallel_tree : int
        Number of trees to grow per round

    tree_method : string [default=’auto’]
        The tree construction algorithm used in XGBoost
        Distributed and external memory version only support approximate algorithm.
        Choices: {‘auto’, ‘exact’, ‘approx’, ‘hist’, ‘gpu_exact’, ‘gpu_hist’}
            ‘auto’: Use heuristic to choose faster one.
                - For small to medium dataset, exact greedy will be used.
                - For very large-dataset, approximate algorithm will be chosen.
                - Because old behavior is always use exact greedy in single machine,
                - user will get a message when approximate algorithm is chosen to
                  notify this choice.
            ‘exact’: Exact greedy algorithm.
            ‘approx’: Approximate greedy algorithm using sketching and histogram.
            ‘hist’: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
            ‘gpu_exact’: GPU implementation of exact algorithm.
            ‘gpu_hist’: GPU implementation of hist algorithm.

    n_gpus : int
        Number of gpu's to use in RandomForestRegressor solver. Default is -1.

    predictor : string [default='gpu_predictor']
        The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
            - 'cpu_predictor': Multicore CPU prediction algorithm.
            - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
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
            num_parallel_tree=1,  # h2o4gpu
            tree_method='gpu_hist',  # h2o4gpu
            n_gpus=-1,  # h2o4gpu
            predictor='gpu_predictor',  # h2o4gpu
            backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes',
                'min_impurity_decrease', 'min_impurity_split', 'bootstrap',
                'oob_score'
            ]
            params = [
                min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                max_features, max_leaf_nodes, min_impurity_decrease,
                min_impurity_split, bootstrap, oob_score
            ]
            params_default = [2, 1, 0.0, 'auto', None, 0.0, None, True, False]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose > 0:
                        print(
                            "WARNING: The sklearn parameter " + params_string[i]
                            + " has been changed from default to " + str(param)
                            + ". Will run Sklearn RandomForestRegressor.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        from h2o4gpu.ensemble import RandomForestRegressorSklearn
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
        silent = True
        if verbose != 0:
            silent = False
        if random_state is None:
            random_state = 0

        import xgboost as xgb
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
            colsample_bytree=colsample_bytree)

        if self.do_sklearn:
            if verbose > 0:
                print("Running sklearn RandomForestRegressor")
            self.model = self.model_sklearn
        else:
            if verbose > 0:
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
        from ..solvers.utils import _setter
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

    GBM builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.

    num_parallel_tree : int
        Number of trees to grow per round

    tree_method : string [default=’auto’]
        The tree construction algorithm used in XGBoost
        Distributed and external memory version only support approximate algorithm.
        Choices: {‘auto’, ‘exact’, ‘approx’, ‘hist’, ‘gpu_exact’, ‘gpu_hist’}
            ‘auto’: Use heuristic to choose faster one.
                - For small to medium dataset, exact greedy will be used.
                - For very large-dataset, approximate algorithm will be chosen.
                - Because old behavior is always use exact greedy in single machine,
                - user will get a message when approximate algorithm is chosen to
                  notify this choice.
            ‘exact’: Exact greedy algorithm.
            ‘approx’: Approximate greedy algorithm using sketching and histogram.
            ‘hist’: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
            ‘gpu_exact’: GPU implementation of exact algorithm.
            ‘gpu_hist’: GPU implementation of hist algorithm.

    n_gpus : int
        Number of gpu's to use in GradientBoostingClassifier solver. Default is -1.

    predictor : string [default='gpu_predictor']
        The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
            - 'cpu_predictor': Multicore CPU prediction algorithm.
            - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
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
            max_features='auto',
            verbose=0,  # h2o4gpu
            max_leaf_nodes=None,
            warm_start=False,
            presort='auto',
            # XGBoost specific params
            colsample_bytree=1.0,  # h2o4gpu
            num_parallel_tree=1,  # h2o4gpu
            tree_method='gpu_hist',  # h2o4gpu
            n_gpus=-1,  # h2o4gpu
            predictor='gpu_predictor',  # h2o4gpu
            backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'loss', 'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'min_impurity_decrease',
                'min_impurity_split', 'init', 'max_features', 'max_leaf_nodes',
                'presort'
            ]
            params = [
                loss, criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, min_impurity_decrease,
                min_impurity_split, init, max_features, max_leaf_nodes, presort
            ]
            params_default = [
                'deviance', 'friedman-mse', 2, 1, 0.0, 0.0, None, None, 'auto',
                None, 'auto'
            ]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose > 0:
                        print(
                            "WARNING: The sklearn parameter " + params_string[i]
                            + " has been changed from default to " + str(param)
                            + ". Will run Sklearn GradientBoostingClassifier.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        from h2o4gpu.ensemble import GradientBoostingClassifierSklearn
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
        silent = True
        if verbose != 0:
            silent = False
        if random_state is None:
            random_state = 0

        import xgboost as xgb
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
            if verbose > 0:
                print("Running sklearn GradientBoostingClassifier")
            self.model = self.model_sklearn
        else:
            if verbose > 0:
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
        import numpy as np
        return np.log(res)

    def predict_proba(self, X):
        if self.do_sklearn:
            res = self.model.predict_proba(X)
            self.set_attributes()
            return res
        res = self.model.predict_proba(X)
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
        from ..solvers.utils import _setter
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

    GBM builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.

    num_parallel_tree : int
        Number of trees to grow per round

    tree_method : string [default=’auto’]
        The tree construction algorithm used in XGBoost
        Distributed and external memory version only support approximate algorithm.
        Choices: {‘auto’, ‘exact’, ‘approx’, ‘hist’, ‘gpu_exact’, ‘gpu_hist’}
            ‘auto’: Use heuristic to choose faster one.
                - For small to medium dataset, exact greedy will be used.
                - For very large-dataset, approximate algorithm will be chosen.
                - Because old behavior is always use exact greedy in single machine,
                - user will get a message when approximate algorithm is chosen to
                  notify this choice.
            ‘exact’: Exact greedy algorithm.
            ‘approx’: Approximate greedy algorithm using sketching and histogram.
            ‘hist’: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
            ‘gpu_exact’: GPU implementation of exact algorithm.
            ‘gpu_hist’: GPU implementation of hist algorithm.

    n_gpus : int
        Number of gpu's to use in GradientBoostingRegressor solver. Default is -1.

    predictor : string [default='gpu_predictor']
        The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
            - 'cpu_predictor': Multicore CPU prediction algorithm.
            - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
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
            max_features='auto',
            alpha=0.9,
            verbose=0,  # h2o4gpu
            max_leaf_nodes=None,
            warm_start=False,
            presort='auto',
            # XGBoost specific params
            colsample_bytree=1.0,  # h2o4gpu
            num_parallel_tree=1,  # h2o4gpu
            tree_method='gpu_hist',  # h2o4gpu
            n_gpus=-1,  # h2o4gpu
            predictor='gpu_predictor',  # h2o4gpu
            backend='auto'):  # h2o4gpu
        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':

            params_string = [
                'loss', 'criterion', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'min_impurity_decrease',
                'min_impurity_split', 'init', 'max_features', 'alpha',
                'max_leaf_nodes', 'presort'
            ]
            params = [
                loss, criterion, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, min_impurity_decrease,
                min_impurity_split, init, max_features, alpha, max_leaf_nodes,
                presort
            ]
            params_default = [
                'ls', 'friedman-mse', 2, 1, 0.0, 0.0, None, None, 'auto', 0.9,
                None, 'auto'
            ]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose > 0:
                        print(
                            "WARNING: The sklearn parameter " + params_string[i]
                            + " has been changed from default to " + str(param)
                            + ". Will run Sklearn GradientBoostingRegressor.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        self.backend = backend

        from h2o4gpu.ensemble import GradientBoostingRegressorSklearn
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
        silent = True
        if verbose != 0:
            silent = False
        if random_state is None:
            random_state = 0

        import xgboost as xgb
        self.model_h2o4gpu = xgb.XGBRegressor(
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
            if verbose > 0:
                print("Running sklearn GradientBoostingRegressor")
            self.model = self.model_sklearn
        else:
            if verbose > 0:
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
        from ..solvers.utils import _setter
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.feature_importances_ = oself.model.feature_importances_')
        s('oself.oob_improvement_ = oself.model.oob_improvement_')
        s('oself.train_score_ = oself.model.train_score_')
        s('oself.loss_ = oself.model.loss_')
        s('oself.init = oself.model.init')
        s('oself.estimators_ = oself.model.estimators_')
