#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import logistic as sk
from ..typecheck.typechecks import assert_is_type
import numpy as np

class LogisticRegression(object):
    """H2O Logistic Regression Solver

    :param penalty : str, 'l1' or 'l2', default: 'l2'
        Details for sklearn: Used to specify the norm used in the penalization.
        The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties.
        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    :param dual : bool, default: False
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    :param tol : float, default: 1e-4
        Tolerance for stopping criteria.

    :param C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    :param fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    :param intercept_scaling : float, default 1.
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    :param class_weight : dict or 'balanced', default: None
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        .. versionadded:: 0.17
           *class_weight='balanced'*

    :param random_state : int, RandomState instance or None, optional, default: None
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    :param solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'liblinear'
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Algorithm to use in the optimization problem.
        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
            'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
            handle multinomial loss; 'liblinear' is limited to one-versus-rest
            schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
            'liblinear' and 'saga' handle L1 penalty.
        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.
        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    :param max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    :param multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for liblinear
        solver.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    :param verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    :param warm_start : bool, default: False
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver.
        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    :param n_jobs : int, default: 1
        Only used for sklearn. If set to non-default backend will run sklearn logistic regression solver.
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    :param n_gpus: int, default is -1
        Number of GPUs to use in GLM solver.

    :param glm_stop_early: bool, default is True
        Stop early when there is no more relative improvement in the primary and
        dual residuals for ADMM. Default is True

    :param glm_stop_early_error_fraction: float, default is 1.0
        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at
        least this much). Default is 1.0.
    """

    def __init__(self,
                 penalty='l2', #h2o4gpu
                 dual=False,
                 tol=1E-2, #h2o4gpu
                 C=1.0, #h2o4gpu
                 fit_intercept=True, #h2o4gpu
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='liblinear',
                 max_iter=5000, #h2o4gpu
                 multi_class='ovr',
                 verbose=0, #h2o4gpu
                 warm_start=False,
                 n_jobs=1,
                 n_gpus=-1, #h2o4gpu
                 glm_stop_early=True, #h2o4gpu
                 glm_stop_early_error_fraction=1.0): #h2o4gpu

        #Fall back to Sklearn
        #Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        
        params_string=['dual', 'intercept_scaling', 'class_weight', 'random_state', 'solver', 'multi_class', 'warm_start', 'n_jobs']
        params=[dual ,intercept_scaling, class_weight,random_state, solver, multi_class, warm_start, n_jobs]
        params_default = [False, 1, None, None, 'liblinear', 'ovr', False, 1]

        i=0
        self.do_sklearn = False
        for param in params:
            if param != params_default[i]:
                self.do_sklearn = True
                print("WARNING: The sklearn parameter " + params_string[i] + " has been changed from default to " + str(
                    param) + ". Will run Sklearn Logistic Regression.")
                self.do_sklearn = True
            i=i+1

        self.model_sklearn=sk.LogisticRegression_sklearn(
                                      penalty=penalty,
                                      dual=dual,
                                      tol=tol,
                                      C=C,
                                      fit_intercept=fit_intercept,
                                      intercept_scaling=intercept_scaling,
                                      class_weight=class_weight,
                                      random_state=random_state,
                                      solver=solver,
                                      max_iter=max_iter,
                                      multi_class=multi_class,
                                      verbose=verbose,
                                      warm_start=warm_start,
                                      n_jobs=n_jobs)
        n_threads = None
        n_alphas = 1 #n_alphas is always 1 since we are only dealing with l1 or l2 regularization
        n_lambdas = 1 #n_lambdas is always 1 since we are only dealing with l1 or l2 regularization
        n_folds = 1 #Might set aside for separate CV class?
        lambda_max = 1.0/C #Inverse of lambda is C
        lambda_min_ratio = 1.0
        lambda_stop_early = False
        store_full_path = 0
        alphas = None
        lambdas = None

        #Utilize penalty parameter to setup alphas
        if penalty is 'l2':
            alpha_min = 0.0
            alpha_max = 0.0
        elif penalty is 'l1':
            alpha_min = 1.0
            alpha_max = 1.0
        else:
            assert ValueError, "penalty should be either l1 or l2 but got " + penalty

        self.model_h2o4gpu=elastic_net.GLM(
            n_threads=n_threads,
            n_gpus=n_gpus,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=n_lambdas,
            n_folds=n_folds,
            n_alphas=n_alphas,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
            family='logistic',
            store_full_path=store_full_path,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            order=None)

        if self.do_sklearn:
            print("Running sklearn Logistic Regression")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu Logistic Regression")
            self.model = self.model_h2o4gpu

    def fit(self, X, y=None, sample_weight=None):
        model_fit = self.model.fit(X,y,sample_weight)
        self.coef_ = model_fit.coef_
        return model_fit

    def predict_proba(self, X):
        if self.do_sklearn:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def decision_function(self,X):
        print("WARNING: decision_function() is using sklearn")
        return self.model_sklearn.decision_function(X)
    
    def densify(self):
        if self.do_sklearn:
            return self.model.densify()
        else:
            pass

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        if self.do_sklearn:
            return self.model.predict(X)
        else:
            preds = self.model.predict(X)
            preds[preds<0.5] = 0
            preds[preds>0.5] = 1

    def predict_log_proba(self, X):
        preds = self.predict_proba(X)
        return np.log(preds)

    def score(self, X, y, sample_weight = None):
        #TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        return self.model_sklearn.score(X, y, sample_weight)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def sparsify(self):
        if self.do_sklearn:
            return self.model.sparsify()
        else:
            assert ValueError, "sparsify() is not yet supporte for h2o4gpu"
